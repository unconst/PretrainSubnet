# Imports
import os
import sys
import math
import torch
import wandb
import argparse
import traceback
import bittensor as bt
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup

# Pull in training reduce.
import __init__
import reduce

# If turned on, restarts begin from the last checkpoint model.
# Otherwise they start from the initial randomized weights of the model.
LOAD = True 

# Parse arguments
def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument( '--lr', type=float, default = 1e-4, help = 'Training learning rate.')
    parser.add_argument( '--bs', type=int, default = 4, help = 'Training batch size.')
    parser.add_argument( '--sl', type=int, default = 1024, help = 'Training sequence length.')
    parser.add_argument( '--dp', type=float, default = 0.1, help = 'Model dropout probability.')
    parser.add_argument( '--n_head', type=int, default = 12, help = 'Model number of attention heads')
    parser.add_argument( '--n_layer', type=int, default = 12, help = 'Number of gpt2 model layers')
    parser.add_argument( '--load', action="store_true", default = False, help = 'Load local model instead of sync.')
    parser.add_argument( '--local', action="store_true", default = True, help = 'Turn on local training.')
    parser.add_argument( '--wandb', action="store_true", default = False, help = 'Turn on wandb')
    parser.add_argument( '--wandb_run_id', type = str, default = None, help="Set the wandb run for carry forward." )
    parser.add_argument( '--no_initial_sync', action="store_true", default = False, help = 'Turn off initial model sync.')
    parser.add_argument( '--max_steps', type=int, default = 50000, help = 'Max training steps.')
    parser.add_argument( '--accs_per_step', type=int, default = 10, help = 'Number of training accumulation steps.')
    parser.add_argument( '--steps_per_log', type=int, default = 1, help = 'Number of steps per log.')
    parser.add_argument( '--steps_per_sync', type=int, default = 100, help = 'Number of steps per chain sync.')
    parser.add_argument( '--steps_per_eval', type=int, default = 50, help = 'Number of steps per eval.')
    parser.add_argument( '--blocks_per_reduce', type=int, default = 10_000, help = 'Number of steps reduce.')
    parser.add_argument( '--blocks_per_set_weights', type=int, default = 100, help = 'Number of blocks before we set weights.')
    parser.add_argument( '--num_warmup', type=int, default = 2000, help = 'Scheduler warm up steps.')
    parser.add_argument( '--netuid', type = int, default = 1, help = "The chain subnet uid." )
    parser.add_argument( '--name', type = str, default = 'pretrain', help = "Name of run." )
    parser.add_argument( '--device', type = str, default = "cuda" if torch.cuda.is_available() else "cpu", help="Device to train on." )
    parser.add_argument( '--dataset_name', type = str, default = "pile", help="Dataset to use." )
    parser.add_argument( '--chain_endpoint', type = str, default = "wss://test.finney.opentensor.ai", help="The chain endpoint to connect with." )
    bt.subtensor.add_args( parser )
    bt.wallet.add_args( parser )
    bt.axon.add_args( parser )
    bt.logging.add_args( parser )
    config = bt.config( parser )

    # Construct save directory.
    config.full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,
            config.wallet.name,
            config.wallet.hotkey,
            config.netuid,
            config.name,
        )
    )
    if not os.path.exists(config.full_path):
        os.makedirs(config.full_path, exist_ok=True)

    return config

def main ( config ):
    bt.logging( config = config, logging_dir = config.full_path )
    bt.logging.info( config )
    pass

    # Setup model and tokenizer
    bt.logging.info( "setting up model" )
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel(GPT2Config(n_layer = config.n_layer, n_head = config.n_head))
    model.to(config.device).train()
    pass

    # Save + load model.
    def save_model( model ):
        bt.logging.info( f"saving model to {config.full_path}/model.pt" )
        torch.save(model.state_dict(), config.full_path + '/model.pt')
    def load_model():
        bt.logging.info( f"loading model from {config.full_path}/model.pt" )
        model = GPT2LMHeadModel(GPT2Config(
            n_layer = config.n_layer, 
            n_head = config.n_head,
            resid_pdrop = config.dp,
            embd_pdrop = config.dp
        ))
        model.load_state_dict(torch.load(config.full_path + '/model.pt'))
        model.to(config.device).train()
        return model

    # Optionally load model from disk.
    if LOAD:
        model = load_model().to(config.device).train()

    class Dataset(IterableDataset):
        def __init__(self, dataset_name:str, tokenizer, sequence_length ):
            self.dataset_name = dataset_name
            self.tokenizer = tokenizer
            self.sequence_length = sequence_length
        def __iter__(self):
            # Select the dataset.
            if self.dataset_name == 'pile':
                data_streamer = load_dataset( "EleutherAI/pile", name="all", split="train", streaming=True )
            elif self.dataset_name == 'red':
                data_streamer = load_dataset("togethercomputer/RedPajama-Data-1T", 'default', split='train', streaming=True)
            else:
                raise RuntimeError( f'{self.dataset_name} is unknown.' )
            # Iterate and tokenize the dataset.
            buffer = []
            for sample in data_streamer.shuffle(buffer_size=10_000):
                buffer += self.tokenizer(sample["text"])["input_ids"]
                buffer += [self.tokenizer.eos_token_id]
                while len(buffer) > self.sequence_length:
                    yield torch.tensor(buffer[: self.sequence_length])
                    buffer = buffer[self.sequence_length :]

    bt.logging.info( "setting up dataloader" )
    dataset = Dataset( dataset_name = config.dataset_name, tokenizer = tokenizer, sequence_length = config.sl )
    dataloader = DataLoader( dataset, batch_size = config.bs, num_workers = 8 )
    pass

    # Get optimized and scheduler
    bt.logging.info( "setting up optimizer" )
    optimizer = torch.optim.AdamW (model.parameters(), lr = config.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.num_warmup, num_training_steps=config.max_steps)  # assuming total steps
    pass

    # Set up Bittensor
    bt.logging.info( "setting up bittensor" )
    wallet = bt.wallet( config = config ).create_if_non_existent()
    subtensor = bt.subtensor( chain_endpoint = config.chain_endpoint  )
    dendrite = bt.dendrite( wallet = wallet )
    axon = bt.axon( wallet = wallet, config = config )
    metagraph = subtensor.metagraph( config.netuid )
    last_set_weights = subtensor.block

    # Set up wandb
    if config.wandb:
        import wandb
        bt.logging.info( "setting up wandb" )
        wandb = wandb.init(
            anonymous = "allow",
            project = "openpretrain",
            entity = "opentensor-dev",
            config = config,
            mode = "online",
            tags = [wallet.hotkey.ss58_address, wallet.coldkeypub.ss58_address],
            dir = config.full_path,
            id = None if not config.wandb_run_id else config.wandb_run_id,
            resume = "allow" if config.wandb_run_id else False
        )

    # Register our wallet, serve our axon, get our uid.
    if not config.local:
        bt.logging.info( f"registering on netuid: {config.netuid}" )
        subtensor.register( wallet = wallet, netuid = config.netuid )
        bt.logging.info( f"serving axon on netuid: {config.netuid}" )
        axon.serve( netuid = config.netuid, subtensor = subtensor )
        metagraph = subtensor.metagraph( config.netuid )    
        my_uid = metagraph.hotkeys.index( wallet.hotkey.ss58_address )
        bt.logging.info( f"registered and served with uid: {my_uid}" )

    # Set up chain connection.
    def chain_sync():
        bt.logging.info( "Syncing chain state." )
        global metagraph
        global subtensor
        global my_uid
        metagraph = subtensor.metagraph( config.netuid )
        my_uid = metagraph.hotkeys.index( wallet.hotkey.ss58_address )
        if config.wandb: wandb.log( { "R": metagraph.R[my_uid], 'S': metagraph.S[my_uid], 'E': metagraph.E[my_uid], 'D': metagraph.D[my_uid], 'I':  metagraph.I[my_uid]} )
    if not config.local:
        chain_sync()

    # Pull latest weights.
    last_merge_axon = None
    if not config.local and not config.no_initial_sync and not config.load:
        bt.logging.info( "Pulling initial weights from others." )
        is_first = True
        tries = 0
        while tries < 5:
            # Reduce model weights with random.
            success, last_merge_axon = reduce.reduce( model, dendrite, metagraph, replace = True, allow_self = not is_first )
            if success:
                bt.logging.success( f"Pulling weights from: {last_merge_axon}" )
                break
            else: 
                is_first = False
                tries += 1 
                # Sync chain state and try again.
                chain_sync()
                continue

    # Record the last sync block.
    last_sync_block = subtensor.block

    # Set up synapse.
    def get_params( synapse: reduce.GetParams ) -> reduce.GetParams:
        best_model = load_model()
        synapse.serialize( model = best_model )
        return synapse
    axon.attach( get_params ).start()

    # Training vars.
    step = 0 # Global step.
    tokens = 0 # total tokens seen.
    alpha = 0.9 # Moving average coefficient for weights.
    best_eval = math.inf # Best loss seen so far.
    accumulation_counter = 0 # Counter for gradient accumulation.
    moving_average_scores = {} # Map from hotkey to loss.

    # Main training loop.
    for epoch in range(3):
        bt.logging.info( f'Epoch {epoch + 1}/{3}' )
        for batch in dataloader:
            try:
                # Forward pass.
                outputs = model(
                    input_ids = batch.to(config.device), 
                    # attention_mask = batch["attention_mask"].to(device),
                    labels = batch.to(config.device)
                ) 
            
                # Count tokens.
                tokens += batch.numel()

                
                # Backward pass
                loss = outputs.loss / config.accs_per_step
                loss.backward()

                # Update weights for miner.
                if last_merge_axon:
                    if last_merge_axon.hotkey in moving_average_scores:
                        moving_average_scores[ last_merge_axon.hotkey ] = alpha * loss.item() + (1 - alpha) * moving_average_scores[ last_merge_axon.hotkey ]
                    else:
                        moving_average_scores[ last_merge_axon.hotkey ] = loss.item()
                    bt.logging.trace( f"Updated weights for {last_merge_axon.hotkey} to {moving_average_scores[ last_merge_axon.hotkey ]}" )

                # Accumulate across batches.
                accumulation_counter += 1
                if accumulation_counter % config.accs_per_step == 0:

                    # Apply gradient step.
                    optimizer.step()
                    scheduler.step() 
                    optimizer.zero_grad()
                    
                    # Log state to terminal and wandb.
                    if step % config.steps_per_log == 0:
                        perplexity = torch.exp(loss * config.accs_per_step).item()
                        loss = loss * config.accs_per_step
                        bt.logging.info(f'Step {step}, Loss {loss}, Perplexity {perplexity}, Tokens {tokens}, spec {__init__.__spec_version__ }, run {None if not config.wandb_run_id else config.wandb_run_id}')
                        if config.wandb: wandb.log( {
                            'step': step, 
                            'loss': loss, 
                            'perplexity': perplexity, 
                            'tokens': tokens, 
                            'spec': __init__.__spec_version__, 
                            'lr': config.lr, 
                            'sl': config.sl, 
                            'bs': config.bs, 
                            'ac': config.accs_per_step
                        })

                    # Sync chain state.
                    if step % config.steps_per_sync == 0 and not config.local:
                        chain_sync()

                    # Increment step.
                    step += 1

                    # Check if we need to sync based on blocks passed since last sync.
                    current_block = subtensor.block
                    if current_block - last_sync_block > config.blocks_per_reduce and not config.local:
                        bt.logging.info( f"Reducing model at block {current_block}" )
                        # Perform the reduction
                        success, last_merge_axon = reduce.reduce(model, dendrite, metagraph)
                        last_sync_block = current_block
                        bt.logging.info( f"Reduced with axon {last_merge_axon}" )

                    # Check if we should set weights after this point.
                    if current_block - last_set_weights > config.blocks_per_set_weights and not config.local:
                        bt.logging.info( f"Setting weights on chain at block {current_block}" )
                        # Create weights tensor.
                        weights = torch.zeros_like( metagraph.uids, dtype = torch.float32 )
                        for uid in metagraph.uids:
                            if metagraph.hotkeys[uid.item()] in moving_average_scores:
                                weights[uid] = moving_average_scores[ metagraph.hotkeys[uid] ]

                        # Normalize weights across uids.
                        weights = torch.nn.functional.normalize( weights, p = 1.0, dim = 0, out = weights )
                        bt.logging.info( f"weights: {weights}" )

                        # Set absolute weights
                        subtensor.set_weights( 
                            netuid = config.netuid, 
                            wallet = wallet, 
                            uids = metagraph.uids, 
                            weights = weights
                        )
                        last_set_weights = current_block
                        bt.logging.info( f"Set weights on chain at block {current_block}" )


                    # Run eval online.
                    if step % config.steps_per_eval == 0:

                        bt.logging.info(f'Running eval')

                        # Load wiki test test.
                        validation_dataset = load_dataset('wikitext', "wikitext-2-raw-v1", split="test")
                        encodings = tokenizer("\n\n".join(validation_dataset["text"]), return_tensors="pt")
                            
                        # Compute perplexity.
                        max_length = config.sl
                        stride = config.sl
                        seq_len = encodings.input_ids.size(1)
                        nlls = []
                        prev_end_loc = 0
                        for begin_loc in tqdm(range(0, seq_len, stride)):
                            end_loc = min(begin_loc + max_length, seq_len)
                            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
                            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(config.device)
                            target_ids = input_ids.clone()
                            target_ids[:, :-trg_len] = -100

                            with torch.no_grad():
                                outputs = model(input_ids, labels=target_ids)

                                # loss is calculated using CrossEntropyLoss which averages over valid labels
                                # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
                                # to the left by 1.
                                neg_log_likelihood = outputs.loss

                            nlls.append(neg_log_likelihood)

                            prev_end_loc = end_loc
                            if end_loc == seq_len:
                                break
                        
                        # Log perplexity.
                        eval_perplexity = torch.exp(torch.stack(nlls).mean())
                        bt.logging.success(f'Eval perplexity: {eval_perplexity.item()}')
                        if config.wandb: wandb.log( {'eval_perplexity': eval_perplexity.item() } )
                        if eval_perplexity < best_eval:
                            best_eval = eval_perplexity
                            save_model( model )

            # Catch unknown errors.
            except RuntimeError as e:
                bt.logging.error(e)
                traceback.print_exc()
        
            # Catch keyboard interrupt.
            except KeyboardInterrupt:
                bt.logging.info("Keyboard interrupt detected. Saving model and exiting.")
                if config.wandb:
                    wandb.finish()
                exit()

if __name__ == "__main__":
    main( get_config() )