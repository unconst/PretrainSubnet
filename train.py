# Imports
import os
import sys
import torch
import wandb
import random
import asyncio
import threading
import argparse
import torch.nn as nn
import bittensor as bt
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup

# Pull in training utils.
import utils

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument( '--lr', type=float, default = 5e-5, help = 'Training learning rate.')
    parser.add_argument( '--bs', type=int, default = 1, help = 'Training batch size')
    parser.add_argument( '--sl', type=int, default = 512, help = 'Training sequence length')
    parser.add_argument( '--n_head', type=int, default = 12, help = 'Model number of attention heads')
    parser.add_argument( '--n_layer', type=int, default = 12, help = 'Number of gpt2 model layers')
    parser.add_argument( '--local', action="store_true", default = False, help = 'Turn on local training.')
    parser.add_argument( '--wandb', action="store_true", default = False, help = 'Turn on wandb')
    parser.add_argument( '--no_self_query', dest = 'self_query', action="store_false", default = False, help = 'Turn off querying yourself.')
    parser.add_argument( '--max_k', type=int, default = 1, help = 'Max number of gradients to merge.')
    parser.add_argument( '--max_steps', type=int, default = 50000, help = 'Max training steps.')
    parser.add_argument( '--steps_per_log', type=int, default = 1, help = 'Number of steps per log.')
    parser.add_argument( '--steps_per_sync', type=int, default = 10, help = 'Number of steps per chain sync.')
    parser.add_argument( '--num_warmup', type=int, default = 2000, help = 'Scheduler warm up steps.')
    parser.add_argument( '--accs_per_step', type=int, default= 3, help = 'Number of training accumulation steps.')
    parser.add_argument( '--netuid', type = int, default = 1, help = "The chain subnet uid." )
    parser.add_argument( '--chain_endpoint', type = str, default = "wss://test.finney.opentensor.ai", help="The chain endpoint to connect with." )
    bt.subtensor.add_args( parser )
    bt.wallet.add_args( parser )
    bt.axon.add_args( parser )
    bt.logging.add_args( parser )
    return bt.config( parser )

config = parse_arguments()
bt.logging( config = config )
bt.logging.info( config )
pass


# Setup model and tokenizer
def setup_model_and_tokenizer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel(GPT2Config(n_layer = config.n_layer, n_head = config.n_head)).to(device)
    model.train()
    return model, tokenizer, device

bt.logging.info( "setting up model" )
model, tokenizer, device = setup_model_and_tokenizer()
pass

# Load dataloader
def load_dataloader():
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation = True, padding = "max_length", max_length = config.sl, return_tensors = "pt")
    dataset = load_dataset("togethercomputer/RedPajama-Data-1T", 'default', split='train', streaming=True)
    dataset = dataset.shuffle(buffer_size = config.bs * 4, seed=42)
    tokenized_dataset = dataset.map( tokenize_function, batched=True )
    dataloader = DataLoader( tokenized_dataset, batch_size = config.bs)
    return dataloader

bt.logging.info( "setting up dataloader" )
dataloader = load_dataloader()
pass


# Get optimized and scheduler
bt.logging.info( "setting up optimizer" )
optimizer = torch.optim.AdamW (model.parameters(), lr = config.lr)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.num_warmup, num_training_steps=config.max_steps)  # assuming total steps
pass

# Set up wandb
if config.wandb:
    bt.logging.info( "setting up wandb" )
    wandb = wandb.init(
        anonymous = "allow",
        project = "openpretrain",
        entity = "opentensor-dev",
        config = config,
        mode = "online"
    )

# Set up Bittensor
bt.logging.info( "setting up bittensor" )
wallet = bt.wallet( config = config ).create_if_non_existent()
subtensor = bt.subtensor( chain_endpoint = config.chain_endpoint  )
dendrite = bt.dendrite( wallet = wallet )
axon = bt.axon( wallet = wallet, config = config )

# Register our wallet, serve our axon, get our uid.
if not config.local:
    bt.logging.info( f"connecting to subtensor: {subtensor}" )
    subtensor.register( wallet = wallet, netuid = config.netuid )
    bt.logging.info( f"serving axon on netuid: {config.netuid}" )
    axon.serve( netuid = config.netuid, subtensor = subtensor )
    metagraph = subtensor.metagraph( config.netuid )    
    my_uid = metagraph.hotkeys.index( wallet.hotkey.ss58_address )
    bt.logging.info( f"served on uid: {my_uid}" )

# Set up chain connection.
def chain_sync():
    bt.logging.info( "Syncing chain state." )
    global metagraph
    global subtensor
    if subtensor.block - metagraph.last_update[my_uid] > 50:
        bt.logging.info( f"Setting weights on chain at block {subtensor.block}" )
        subtensor.set_weights( netuid = config.netuid, wallet = wallet, uids = [my_uid], weights = [1.0] )
        metagraph = subtensor.metagraph( config.netuid )
if not config.local:
    chain_sync()

# Set up synapse.
def get_grads( synapse: utils.GetGrads ) -> utils.GetGrads:
    global model
    synapse.serialize( model = model )
    return synapse
axon.attach( get_grads ).start()

# Set up dendrite get grads.
async def run_gradient_merging():

    # Function which merges gradients with one random axon.
    async def merge_random():
        global metagraph
        global dendrite
        global subtensor
        # Query random available axon.
        available = [ metagraph.axons[uid] for uid in metagraph.uids if subtensor.block - metagraph.last_update[uid] < 1000 ]

        # Check if we should query self.
        if config.self_query: available.append( metagraph.axons[my_uid] )

        # Check no available.
        if len( available ) == 0: bt.logging.debug( f"No available axons to query." ); return

        axon = random.choice( available )
        bt.logging.debug( f"Merging gradients with axon {axon}" )

        # Query axon and get grads.
        grad_dict = await dendrite( axon, utils.GetGrads() )

        # Check if it is valid.
        if utils.is_valid_grad_dict( model, grad_dict ):

            # Apply grad to model.
            utils.apply_grads_to_model( model, grad_dict )
            bt.logging.debug(f'merged gradients with axon: {axon}')

    while True:
        await merge_random()
        await asyncio.sleep( 1 )

def start_async_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run_gradient_merging())

def run_merge():
    new_loop = asyncio.new_event_loop()
    t = threading.Thread(target=start_async_loop, args=(new_loop,))
    t.start()

run_merge()

# training loop
step = 0
accumulation_counter = 0
for epoch in range(3):
    bt.logging.info( f'Epoch {epoch + 1}/{3}' )
    for batch in dataloader:
        
        # Forward pass.
        outputs = model(
            input_ids = batch["input_ids"].to(device), 
            attention_mask = batch["attention_mask"].to(device),
            labels = batch["input_ids"].to(device)
        ) 
        
        # Backward pass
        loss = outputs.loss / config.accs_per_step
        loss.backward()
        
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
