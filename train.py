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

# Pull in training reduce.
import reduce

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument( '--lr', type=float, default = 3e-4, help = 'Training learning rate.')
    parser.add_argument( '--bs', type=int, default = 1, help = 'Training batch size')
    parser.add_argument( '--sl', type=int, default = 512, help = 'Training sequence length')
    parser.add_argument( '--n_head', type=int, default = 12, help = 'Model number of attention heads')
    parser.add_argument( '--n_layer', type=int, default = 12, help = 'Number of gpt2 model layers')
    parser.add_argument( '--local', action="store_true", default = False, help = 'Turn on local training.')
    parser.add_argument( '--wandb', action="store_true", default = False, help = 'Turn on wandb')
    parser.add_argument( '--mock', action="store_true", default = False, help = 'Turn on mocking.')
    parser.add_argument( '--self_query', action="store_true", default = False, help = 'Query only yourself.')
    parser.add_argument( '--max_k', type=int, default = 1, help = 'Max number of gradients to merge.')
    parser.add_argument( '--max_steps', type=int, default = 50000, help = 'Max training steps.')
    parser.add_argument( '--accs_per_step', type=int, default = 5, help = 'Number of training accumulation steps.')
    parser.add_argument( '--steps_per_log', type=int, default = 1, help = 'Number of steps per log.')
    parser.add_argument( '--steps_per_sync', type=int, default = 200, help = 'Number of steps per chain sync.')
    parser.add_argument( '--steps_per_reduce', type=int, default = 100, help = 'Number of steps reduce.')
    parser.add_argument( '--num_warmup', type=int, default = 2000, help = 'Scheduler warm up steps.')
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
    if not config.mock:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel(GPT2Config(n_layer = config.n_layer, n_head = config.n_head)).to(device)
        model.train()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel(GPT2Config(n_layer = 1, n_head = 1)).to(device)
        model.train()
    return model, tokenizer, device

bt.logging.info( "setting up model" )
model, tokenizer, device = setup_model_and_tokenizer()
pass

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
metagraph = subtensor.metagraph( config.netuid )

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
    if subtensor.block - metagraph.last_update[my_uid] > 50:
        bt.logging.info( f"Setting weights on chain at block {subtensor.block}" )
        subtensor.set_weights( netuid = config.netuid, wallet = wallet, uids = [my_uid], weights = [1.0] )
        metagraph = subtensor.metagraph( config.netuid )
if not config.local:
    chain_sync()

# Set up synapse.
def get_params( synapse: reduce.GetParams ) -> reduce.GetParams:
    global model
    synapse.serialize( model = model )
    return synapse
axon.attach( get_params ).start()


if not config.local:
    is_first = True
    while True:
        # Reduce model weights with random.
        success, model = reduce.reduce( model, dendrite, metagraph, replace = True, allow_self = not is_first )
        if success:
            break
        else: 
            is_first = False
            # Sync chain state and try again.
            chain_sync()
            continue

# training loop
step = 0
accumulation_counter = 0
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

                if step % config.steps_per_reduce == 0 and not config.local:
                    # Perform the reduction
                    success, model = reduce.reduce(model, dendrite, metagraph)


        except RuntimeError as e:
            bt.logging.error(e)
        
        except KeyboardInterrupt:
            bt.logging.info("Keyboard interrupt detected. Saving model and exiting.")
            if config.wandb:
                wandb.finish()
            exit()

