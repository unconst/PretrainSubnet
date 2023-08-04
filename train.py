# Imports
import os
import sys
import math
import torch
import wandb
import random
import asyncio
import threading
import argparse
import traceback
import torch.nn as nn
import bittensor as bt
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup

# Pull in training reduce.
import reduce

# Parse arguments
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument( '--lr', type=float, default = 3e-4, help = 'Training learning rate.')
    parser.add_argument( '--bs', type=int, default = 8, help = 'Training batch size.')
    parser.add_argument( '--sl', type=int, default = 1024, help = 'Training sequence length.')
    parser.add_argument( '--n_head', type=int, default = 12, help = 'Model number of attention heads')
    parser.add_argument( '--n_layer', type=int, default = 12, help = 'Number of gpt2 model layers')
    parser.add_argument( '--local', action="store_true", default = False, help = 'Turn on local training.')
    parser.add_argument( '--wandb', action="store_true", default = False, help = 'Turn on wandb')
    parser.add_argument( '--validator', action="store_true", default = False, help = 'Turn on validating')
    parser.add_argument( '--no_initial_sync', action="store_true", default = False, help = 'Turn off initial model sync.')
    parser.add_argument( '--mock', action="store_true", default = False, help = 'Turn on mocking.')
    parser.add_argument( '--self_query', action="store_true", default = False, help = 'Query only yourself.')
    parser.add_argument( '--max_k', type=int, default = 1, help = 'Max number of gradients to merge.')
    parser.add_argument( '--max_steps', type=int, default = 50000, help = 'Max training steps.')
    parser.add_argument( '--accs_per_step', type=int, default = 5, help = 'Number of training accumulation steps.')
    parser.add_argument( '--steps_per_log', type=int, default = 1, help = 'Number of steps per log.')
    parser.add_argument( '--steps_per_sync', type=int, default = 100, help = 'Number of steps per chain sync.')
    parser.add_argument( '--steps_per_eval', type=int, default = 300, help = 'Number of steps per eval.')
    parser.add_argument( '--blocks_per_reduce', type=int, default = 22, help = 'Number of steps reduce.')
    parser.add_argument( '--blocks_per_set_weights', type=int, default = 100, help = 'Number of blocks before we set weights.')
    parser.add_argument( '--num_warmup', type=int, default = 2000, help = 'Scheduler warm up steps.')
    parser.add_argument( '--netuid', type = int, default = 1, help = "The chain subnet uid." )
    parser.add_argument( '--name', type = str, default = 'pretrain', help = "Name of run." )
    parser.add_argument( '--chain_endpoint', type = str, default = "wss://test.finney.opentensor.ai", help="The chain endpoint to connect with." )
    bt.subtensor.add_args( parser )
    bt.wallet.add_args( parser )
    bt.axon.add_args( parser )
    bt.logging.add_args( parser )
    return bt.config( parser )

config = parse_arguments()

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
bt.logging( config = config, logging_dir = config.full_path )
bt.logging.info( config )
pass

# Setup model and tokenizer
def setup_model_and_tokenizer():
    if not config.mock:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel(GPT2Config(n_layer = config.n_layer, n_head = config.n_head))
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel(GPT2Config(n_layer = 1, n_head = 1))
    return model, tokenizer, device

bt.logging.info( "setting up model" )
model, tokenizer, device = setup_model_and_tokenizer()
pass

# Save + load model.
def save_model( model ):
    bt.logging.info( f"saving model to {config.full_path}/model.pt" )
    torch.save(model.state_dict(), config.full_path + '/model.pt')
def load_model():
    bt.logging.info( f"loading model from {config.full_path}/model.pt" )
    model, _, _ = setup_model_and_tokenizer()
    model.load_state_dict(torch.load(config.full_path + '/model.pt'))
    return model
save_model( model )
model = load_model().to(device).train()

# Load dataloader
def load_dataloader():
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation = True, padding = "max_length", max_length = config.sl, return_tensors = "pt")
    if not config.mock:
        dataset = load_dataset("togethercomputer/RedPajama-Data-1T", 'default', split='train', streaming=True)
        dataset = dataset.shuffle(buffer_size = config.bs * 4, seed = random.randint(0, 1000))
        tokenized_dataset = dataset.map( tokenize_function, batched=True )
        dataloader = DataLoader( tokenized_dataset, batch_size = config.bs)
    else:
        texts = ["mock sentence "+str(i) for i in range(100)]  # creating 100 mock sentences
        encoded_texts = [tokenize_function({"text": txt}) for txt in texts]
        dataloader = DataLoader(encoded_texts, batch_size = config.bs)
    return dataloader

bt.logging.info( "setting up dataloader" )
dataloader = load_dataloader()
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
    bt.logging.info( "setting up wandb" )
    wandb = wandb.init(
        anonymous = "allow",
        project = "openpretrain",
        entity = "opentensor-dev",
        config = config,
        mode = "online",
        tags=[wallet.hotkey.ss58_address, wallet.coldkeypub.ss58_address],
        dir = config.full_path,
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
if not config.local and not config.no_initial_sync:
    is_first = True
    tries = 0
    while tries < 5:
        # Reduce model weights with random.
        success, last_merge_axon = reduce.reduce( model, dendrite, metagraph, replace = True, allow_self = not is_first )
        if success:
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
alpha = 0.9 # Moving average coefficient for weights.
best_loss = math.inf # Best loss seen so far.
accumulation_counter = 0 # Counter for gradient accumulation.
moving_average_scores = {} # Map from hotkey to loss.

# Main training loop.
for epoch in range(3):
    bt.logging.info( f'Epoch {epoch + 1}/{3}' )
    for batch in dataloader:
        try:
            # Forward pass.
            outputs = model(
                input_ids = batch["input_ids"].to(device), 
                attention_mask = batch["attention_mask"].to(device),
                labels = batch["input_ids"].to(device)
            ) 
            
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
                    bt.logging.info(f'Step {step}, Loss {loss}, Perplexity {perplexity}')
                    if config.wandb: wandb.log( {'step': step, 'loss': loss, 'perplexity': perplexity } )

                # Sync chain state.
                if step % config.steps_per_sync == 0 and not config.local:
                    chain_sync()

                # Increment step.
                step += 1

                # Check if our model has beaten the current best.
                if loss < best_loss:
                    bt.logging.debug( f"New best loss: {loss}" )
                    # Save the model as the best we have.
                    save_model( model )

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
                    validation_datasets = load_dataset('wikitext', "wikitext-2-raw-v1", split="test")
                    encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")
                        
                    # Compute perplexity.
                    max_length = config.sl
                    stride = config.sl
                    seq_len = encodings.input_ids.size(1)
                    nlls = []
                    prev_end_loc = 0
                    for begin_loc in tqdm(range(0, seq_len, stride)):
                        end_loc = min(begin_loc + max_length, seq_len)
                        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
                        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
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
                    bt.logging.success(f'Eval perplexity: {perplexity.item()}')
                    if config.wandb: wandb.log( {'eval_perplexity': eval_perplexity.item() } )

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

