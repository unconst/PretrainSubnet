
import sys
import time
import wandb
import torch
import typing
import random
import argparse
import torch.nn as nn
import bittensor as bt
import torch.optim as optim
import torch.nn.functional as F
from collections import Counter
from typing import List, Optional
from types import SimpleNamespace

# GPT2 Specific.
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling
from transformers import GPT2LMHeadModel, AdamW
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Define the constant chain endpoint and network unique ID
CHAIN_ENDPOINT = "wss://test.finney.opentensor.ai"
NETUID = 97


# Protocol Definition to get Gradients
class GetGrads( bt.Synapse ):
    """
    The GetGrads class is used to get the gradients of the model.
    It subclasses the bittensor Synapse.
    """
    # Gradients per variable in the model.
    grads: typing.Optional[typing.Dict[str, bt.Tensor]] = None

    # Define deserialization function
    def deserialize( self ) -> typing.Dict[ str, torch.FloatTensor ]:
        """
        Deserialize method converts the Bittensor gradients to Pytorch tensors.

        Returns:
        Dictionary with gradient tensors.
        """
        if self.grads: 
            return { name: g.tensor() for name, g in self.grads.items() }
        else: 
            return {}

# Protocol Definition to get Weights
class GetWeights( bt.Synapse ):
    """
    The GetWeights class is used to get the weights of the model.
    It subclasses the bittensor Synapse.
    """
    # Weights per variable in the model.
    weights: Optional[ typing.Dict[ str, bt.Tensor ] ] = None

    # Define deserialization function
    def deserialize( self ) -> typing.Dict[ str, torch.FloatTensor ]:
        """
        Deserialize method converts the Bittensor weights to Pytorch tensors.

        Returns:
        Dictionary with weight tensors.
        """
        if self.weights:
            return { name: w.tensor() for name, w in self.weights.items() }
        else: 
            return {}

# Miner Definition
class DMiner:
    """
    The DMiner class is the decentralized miner, which interacts with the subtensor chain, dendrite, axon, and the model.
    """
    @classmethod
    def config(cls) -> bt.config:
        """
        Configuration method for DMiner class.

        Returns:
        Bittensor configuration object with default values set.
        """
        parser = argparse.ArgumentParser()
        bt.wallet.add_args( parser )
        bt.axon.add_args( parser )
        bt.logging.add_args( parser )
        config = bt.config( parser )
        return config

    def __init__( self ):
        """
        Initialization method for DMiner class.
        """
        # Create config
        self.config = DMiner.config()

        # Turn on bittensor logging
        bt.logging( config = self.config )

        # Create wallet hotkey and coldkey.
        self.wallet = bt.wallet( config = self.config )
        self.wallet.create_if_non_existent()
        bt.logging.debug( 'wallet:', self.wallet.hotkey.ss58_address )

        # Initialize Bittensor objects.
        self.subtensor = bt.subtensor( chain_endpoint = CHAIN_ENDPOINT )
        self.axon = bt.axon( wallet = self.wallet, config = self.config )
        self.dendrite = bt.dendrite( wallet = self.wallet )
        bt.logging.debug( 'subtensor:', self.subtensor )
        bt.logging.debug( 'axon: ', self.axon )
        bt.logging.debug( 'dendrite: ', self.dendrite )

        # Set the batch size and the number of epochs
        self.batch_size = 16

        # Load the 'wikitext' dataset
        self.wikitext_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')['train']

        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        # Add a padding token and set it to the same as the EOS token
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Tokenize the training_dataset
        def encode(examples): return self.tokenizer(examples['text'])
        self.train_dataset = self.wikitext_dataset.map( encode, batched = True, remove_columns = ['text']) 

        # Build the dataset collator.
        self.data_collator = DataCollatorForLanguageModeling( tokenizer = self.tokenizer, mlm = False, pad_to_multiple_of = 128 )

        # Create a DataLoader
        self.dataloader = DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle = True, collate_fn = self.data_collator)

        # Load pre-trained model (weights)
        self.model = GPT2LMHeadModel.from_pretrained('gpt2', pad_token_id=self.tokenizer.eos_token_id)

        # Move the model to the GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Initialize the optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=1e-5)

        # Init wandb
        tags = [self.wallet.hotkey.ss58_address]
        self.wandb = wandb.init(
            anonymous="allow",
            reinit = True,
            project = "open-pretrain",
            entity = "opentensor-dev",
            config = self.config,
            tags = [self.wallet.hotkey.ss58_address],
        )
        bt.logging.success(
            prefix="Started a new wandb run",
            sufix=f"<blue> {self.wandb.name} </blue>",
        )

    def get_grads( self, synapse: GetGrads ) -> GetGrads:
        """
        Method to get the current gradients from the model.

        Args:
            synapse: GetGrads object that holds the gradients.

        Returns:
            GetGrads object with updated gradients.
        """
        synapse.grads = self.saved_grads
        return synapse

    def get_weights( self, synapse: GetWeights ) -> GetWeights:
        """
        Method to get weights from the model.

        Args:
            synapse: GetWeights object that holds the weights.

        Returns:
            GetWeights object with updated weights.
        """
        synapse.weights = { name: bt.tensor( weight ) for name, weight in self.model.state_dict().items() }
        return synapse
    
    def get_online_uids( self ) -> List[int]:
        current_block = self.subtensor.block
        return [ uid for uid, update in enumerate( self.metagraph.last_update ) if current_block - update < 100 ]

    def apply_remote_grads( self ):
        """
        Apply the gradients received from remote miners to the model.
        """
        # import pdb; pdb.set_trace()
        # bt.logging.info( 'Reducing gradients.')
        wandb.log({ 'reduce_gradients_event': 1.0 })

        # Get all online axons..
        online_axons = [self.metagraph.axons[uid] for uid in self.get_online_uids() ]
        if len( online_axons ) == 0: return
        bt.logging.info(f'Reducing grads with uids: {self.get_online_uids()}')

        # Get the saved grads from everyone.
        grad_dicts = self.dendrite.query( online_axons, GetGrads(), timeout = 5 )
        if not isinstance(grad_dicts, list ): grad_dicts = [grad_dicts]

        # Create a new state dictionary for the averaged grads
        avg_valid_grads_dict = {}
        for key in self.model.state_dict().keys():

            all_grads = [grad_dict[key] for grad_dict in grad_dicts if key in grad_dict]
            if len(all_grads) == 0: continue

            # stack the weights along a new dimension, and take their mean
            avg_grad = torch.stack( all_grads ).mean(dim=0).to(self.device)

            # assign the average weight to the new state dictionary
            avg_valid_grads_dict[key] = avg_grad

        # Apply avg valid grads to model.
        for name, param in self.model.named_parameters():
            if name in avg_valid_grads_dict:
                if param.grad is not None:
                    param.grad += avg_valid_grads_dict[name]
                else:
                    param.grad = avg_valid_grads_dict[name].clone()

        bt.logging.success(f'Successfully reduced {len(grad_dicts)} grads.') 
        wandb.log({ 'successfully_average_gradients': 1.0 })

    def average_weights_across_miners( self ):
        """
        This method retrieves the model weights from all miners in the network and then averages 
        these weights. The averaged weights are then used to update the local model.

        First, it sends a GetWeights query to all the axons (miners) in the metagraph (network). 
        The result is a list of state_dicts, each representing the state of a model from a 
        different miner. It discards any None responses, which may be due to failures or timeouts.

        Then, it creates a new state_dict where each weight is the average of the corresponding 
        weights from all miners. This is done by iterating over the keys (parameter names) in 
        the first state_dict and for each key, computing the average of the corresponding values 
        from all state_dicts. 

        Finally, it updates the state of the local model using the averaged state_dict. This is 
        done using the PyTorch method load_state_dict.

        Note that this averaging process is straightforward because all models are assumed to 
        have the exact same architecture and therefore the exact same parameter names and shapes.
        """
        # import pdb; pdb.set_trace()
        bt.logging.info( 'Avergating weights')
        wandb.log({ 'average_weights_event': 1.0 })

        # Get all online axons..
        online_axons = [self.metagraph.axons[uid] for uid in self.get_online_uids() ]
        if len( online_axons ) == 0: return
        bt.logging.info(f'Averaging weights with uids: {self.get_online_uids()}')

        # Query all miners for their model weights.
        state_dicts = self.dendrite.query( online_axons, GetWeights(), timeout = 5)
        if not isinstance( state_dicts, list ): state_dicts = [state_dicts]
        
        # Create a new state dictionary for the averaged weights
        avg_state_dict = {}
        for key in self.model.state_dict().keys():

            all_weights = [state_dict[key] for state_dict in state_dicts if key in state_dict]
            if len(all_weights) == 0: continue

            # stack the weights along a new dimension, and take their mean
            avg_weight = torch.stack(all_weights).mean(dim=0)

            # assign the average weight to the new state dictionary
            avg_state_dict[key] = avg_weight

        # Load the averaged weights into the local model.
        if avg_state_dict != {}:
            self.model.load_state_dict(avg_state_dict)
        bt.logging.success(f'Successfully averaged {len(state_dicts)} weights.') 
        wandb.log({ 'successfully_average_weights': 1.0 })

    def run( self ):
        """
        This method is the main loop that runs the training for the miner. It first registers
        the miner's wallet, sets up the communication infrastructure (axon), retrieves the
        current network state (metagraph), and averages the weights from the network miners.

        Then, it enters a loop where it retrieves the metagraph at each step, computes
        gradients from the local data batch, saves these gradients, retrieves gradients from
        other miners, applies these gradients to the local model, and steps the optimizer.

        After processing all the batches, the function again averages the weights from the 
        network miners.

        Finally, it evaluates the model on the test set, logs the results, and waits for the 
        next training epoch to start.
        """
        # Register wallet with the Subtensor (blockchain).
        self.subtensor.register(wallet=self.wallet, netuid=NETUID)

        # Build, attach, serve and start Axon for communication with other miners.
        self.axon.attach( forward_fn = self.get_grads ).attach( forward_fn = self.get_weights ).serve( netuid = NETUID, subtensor = self.subtensor ).start()

        # Fetch the current network state (metagraph) from the Subtensor.
        self.metagraph = self.subtensor.metagraph(NETUID)

        # Fetch my uid.
        self.my_uid = self.metagraph.hotkeys.index( self.wallet.hotkey.ss58_address )
        bt.logging.info( 'Set weights ')

          # Set ping weights.
        if self.subtensor.block - self.metagraph.last_update[ self.my_uid ] > 50:
            self.subtensor.set_weights( 
                netuid = NETUID, 
                wallet = self.wallet, 
                uids = [self.my_uid], 
                weights = [1.0],
                wait_for_inclusion = False,
                wait_for_finalization = False,
            )
            bt.logging.info( 'Set weights ')
            wandb.log({ 'set_weights': 1.0 })

        # Average model weights across all miners in the network.
        self.average_weights_across_miners()

        # Initialize step counter
        global_step = 0
        training_step = 0

        # Training loop
        while True:
            bt.logging.info(f'Starting new epoch: {global_step}')

            # Fetch the current network state (metagraph) from Subtensor.
            self.metagraph = self.subtensor.metagraph(NETUID)
            bt.logging.info( 'Online UIDS', self.get_online_uids() )
            wandb.log({ 'n_online': len(self.get_online_uids()) })

            # Set the model to training mode.
            self.model.train()

            # Clear out all existing gradients in the model.
            self.optimizer.zero_grad()

            # Train on epoch.
            for batch in self.dataloader:
                bt.logging.debug(f'Starting new training step: {training_step}')

                # Zero out gradients calculated in the previous iteration.
                # and save them for others to query.
                self.saved_grads = { name: bt.tensor(parameter.grad.clone()) for name, parameter in self.model.named_parameters() if parameter.grad is not None }
                self.optimizer.zero_grad()

                # Move the batch tensors to the same device as the model
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss

                # Backward pass
                loss.backward()

                # Retrieve and apply gradients from other miners.
                self.apply_remote_grads()

                # Update the weights
                self.optimizer.step()

                 # Log the loss value for this batch.
                training_step += 1
                wandb.log({ 'block': self.subtensor.block })
                wandb.log({ 'training_step': training_step })
                wandb.log({ 'train_loss': loss })
                bt.logging.info(f"Loss: {loss.item()}") 

            # Set ping weights.
            if self.subtensor.block - self.metagraph.last_update[ self.my_uid ] > 50:
                self.subtensor.set_weights( 
                    netuid = NETUID, 
                    wallet = self.wallet, 
                    uids = [self.my_uid], 
                    weights = [1.0],
                    wait_for_inclusion = False,
                    wait_for_finalization = False,
                )
                bt.logging.info( 'Set weights ')
                wandb.log({ 'set_weights': 1.0 })

            # Increment step counter.
            global_step += 1
            wandb.log({ 'global_step': global_step })


if __name__ == "__main__":
    miner = DMiner()
    miner.run()

