# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

__spec_version__ = 6

# Imports
import os
import sys
import math
import torch
import argparse
import traceback
import bittensor as bt
from datasets import load_dataset
from transformers import GPT2TokenizerFast
from torch.utils.data import DataLoader, IterableDataset

# Pull in training reduce.
import reduce as reduce
import models as models
import benchmark as benchmark

# Exception handling for sigterm.
import signal
class SigTermException( Exception ):pass
def handler_sigterm( signum, frame ): raise SigTermException("Received SIGTERM")
signal.signal(signal.SIGTERM, handler_sigterm)

# Parse arguments
def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument( '--lr', type=float, default = 3e-5, help = 'Training learning rate.')
    parser.add_argument( '--wd', type=float, default = 1e-1, help = 'Training weight decay.')
    parser.add_argument( '--bs', type=int, default = 4, help = 'Training batch size.')
    parser.add_argument( '--sl', type=int, default = 1024, help = 'Training sequence length.')
    parser.add_argument( '--model_type', type = str, default = 'gpt2', help = "Model type to train")
    parser.add_argument( '--n_head', type=int, default = 12, help = 'Model number of attention heads')
    parser.add_argument( '--n_layer', type=int, default = 12, help = 'Number of gpt2 model layers')
    parser.add_argument( '--load', action="store_true", default = False, help = 'Load local model instead of sync.')
    parser.add_argument( '--local', action="store_true", default = False, help = 'Turn on local training.')
    parser.add_argument( '--wandb', action="store_true", default = False, help = 'Turn on wandb')
    parser.add_argument( '--wandb_run_id', type = str, default = None, help="Set the wandb run for carry forward." )
    parser.add_argument( '--validator', action="store_true", default = False, help = 'Turn on validating')
    parser.add_argument( '--no_initial_sync', action="store_true", default = False, help = 'Turn off initial model sync.')
    parser.add_argument( '--mock', action="store_true", default = False, help = 'Turn on mocking.')
    parser.add_argument( '--self_query', action="store_true", default = False, help = 'Query only yourself.')
    parser.add_argument( '--max_k', type=int, default = 1, help = 'Max number of gradients to merge.')
    parser.add_argument( '--accs_per_step', type=int, default = 6, help = 'Number of training accumulation steps.')
    parser.add_argument( '--epochs', type=int, default = 3, help = 'Number of training epochs.')
    parser.add_argument( '--steps_per_log', type=int, default = 1, help = 'Number of steps per log.')
    parser.add_argument( '--steps_per_sync', type=int, default = 100, help = 'Number of steps per chain sync.')
    parser.add_argument( '--steps_per_eval', type=int, default = 125, help = 'Number of steps per eval.')
    parser.add_argument( '--steps_per_reduce', type=int, default = 100, help = 'Number of steps reduce.')
    parser.add_argument( '--steps_per_set_weights', type=int, default = 400, help = 'Number of blocks before we set weights.')
    parser.add_argument( '--netuid', type = int, default = 1, help = "The chain subnet uid." )
    parser.add_argument( '--name', type = str, default = 'pretrain', help = "Name of run." )
    parser.add_argument( '--chain_endpoint', type = str, default = "wss://test.finney.opentensor.ai", help="The chain endpoint to connect with." )
    parser.add_argument( '--loader_script_path', type = str, default = "load_redpajama_random.py", help="Path to dataloader custom script." )
    parser.add_argument( '--shuffle_seed', type = int, default = 1337, help="Seed for shuffling dataset." )
    parser.add_argument( '--device', type = str, default = "cuda" if torch.cuda.is_available() else "cpu", help="Device to train on." )
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

def main( config ):
    # Setup logging.
    bt.logging( config = config, logging_dir = config.full_path )
    bt.logging.info( config )

    # Load model.
    bt.logging.info( "setting up model" )
    device = torch.device(config.device)
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = models.make_model( config )
    model.to(device).train()
    pass

    # Save + load model.
    def save_model( model ):
        bt.logging.info( f"saving model to {config.full_path}/model.pt" )
        torch.save(model.state_dict(), config.full_path + '/model.pt')
    def load_model():
        bt.logging.info( f"loading model from {config.full_path}/model.pt" )
        model = models.make_model( config )
        model.load_state_dict(torch.load(config.full_path + '/model.pt'))
        return model

    # Optionally load model from disk.
    if config.load:
        try:
            model = load_model().to(device).train()
        except Exception as e:
            bt.logging.error( f"Failed to load model with error: {e}" )

    class PileDataset(IterableDataset):
        def __init__(self, tokenizer, sequence_length ):
            self.tokenizer = tokenizer
            self.sequence_length = sequence_length
        def __iter__(self):
            buffer = []
            for sample in load_dataset( "EleutherAI/pile", name="all", split="train", streaming=True ).shuffle(buffer_size=10_000):
                buffer += self.tokenizer(sample["text"])["input_ids"]
                buffer += [self.tokenizer.eos_token_id]
                while len(buffer) > self.sequence_length:
                    yield torch.tensor(buffer[: self.sequence_length])
                    buffer = buffer[self.sequence_length :]

    # Load the dataloader.
    bt.logging.info( "setting up dataloader" )
    pile_dataset = PileDataset( tokenizer = tokenizer, sequence_length = config.sl )
    dataloader = DataLoader( pile_dataset, batch_size = config.bs, num_workers = 8 )
    pass

    # Get optimizer
    bt.logging.info( "setting up optimizer" )
    optimizer = torch.optim.Adam(
        params = model.parameters(),
        lr = config.lr,
        weight_decay = config.wd,
        betas = (0.9, 0.95),
        fused = True if 'cuda' in config.device else False,
    )
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
            tags=[wallet.hotkey.ss58_address, wallet.coldkeypub.ss58_address],
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

    # Pull latest weights.
    last_merge_axon = None
    if not config.local and not config.no_initial_sync and not config.load:
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
    for epoch in range(config.epochs):
        bt.logging.info( f'Epoch {epoch + 1}/{config.epochs}' )
        for batch in dataloader:
            batch = batch.to(device)
            with torch.autocast( device_type="cuda", enabled=True ):
                try:
                    # Forward pass.
                    outputs = model( batch.to(device), labels = batch.to(device) ) 
                    
                    # Backward pass
                    loss = outputs.loss / config.accs_per_step
                    loss.backward()

                    # Increment token count.
                    tokens += batch.numel()

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
                        optimizer.zero_grad()

                        # Increment step.
                        step += 1
                        
                        # Log state to terminal and wandb.
                        if step % config.steps_per_log == 0:
                            perplexity = torch.exp(loss * config.accs_per_step).item()
                            loss = loss * config.accs_per_step
                            bt.logging.info(f'Step {step}, Loss {loss}, Perplexity {perplexity}, Tokens {tokens}, __spec_version__: {__spec_version__}  ')
                            if config.wandb: wandb.log( {'step': step, 'loss': loss, 'perplexity': perplexity, 'tokens': tokens, '__spec_version__': __spec_version__ } )

                        # Sync chain state.
                        if step % config.steps_per_sync == 0 and not config.local:
                            # Pull the latest metagraph.
                            metagraph = subtensor.metagraph( config.netuid )
                            my_uid = metagraph.hotkeys.index( wallet.hotkey.ss58_address )
                            if config.wandb: wandb.log( { "R": metagraph.R[my_uid], 'S': metagraph.S[my_uid], 'E': metagraph.E[my_uid], 'D': metagraph.D[my_uid], 'I':  metagraph.I[my_uid]} )

                        # Check if we need to sync based on blocks passed since last sync.
                        current_block = subtensor.block
                        if step % config.steps_per_reduce == 0 and not config.local:
                            bt.logging.info( f"Reducing model at block {current_block}" )
                            # Perform the reduction
                            success, last_merge_axon = reduce.reduce(model, dendrite, metagraph)
                            last_sync_block = current_block
                            bt.logging.info( f"Reduced with axon {last_merge_axon}" )
                            if config.wandb: wandb.log( {'reduce': metagraph.hotkeys.index( last_merge_axon.hotkey ) } )

                        # Check if we should set weights after this point.
                        if step % config.steps_per_set_weights == 0 and not config.local:
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
                            eval_perplexity = benchmark.calculate_wikitext_perplexity( model, tokenizer, device, config.sl )
                            bt.logging.success(f'Eval perplexity: {eval_perplexity}')
                            if config.wandb: wandb.log( {'eval_perplexity': eval_perplexity } )
                            if eval_perplexity < best_eval:
                                best_eval = eval_perplexity
                                save_model( model )


                # Catch unknown errors.
                except RuntimeError as e:
                    bt.logging.error(e)
                    traceback.print_exc()

                # Catch SigTermException
                except SigTermException:
                    bt.logging.info("Caught SIGTERM")
                    if config.wandb:
                        wandb.finish()
                    exit()
            
                # Catch keyboard interrupt.
                except KeyboardInterrupt:
                    bt.logging.info("Keyboard interrupt detected. Saving model and exiting.")
                    if config.wandb:
                        wandb.finish()
                    exit()


if __name__ == "__main__":
    main( get_config() )