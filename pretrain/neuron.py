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

# GPT2 Specific.
from transformers import GPT2LMHeadModel, AdamW

from pretrain.config import init_config
from pretrain.dataset import get_dataloader
from pretrain.misc import init_wandb
from pretrain.run import run
from pretrain.get_grads import GetGrads, get_grads
from pretrain.get_weights import GetWeights, get_weights

# Miner Definition
class DMiner:
    """
    The DMiner class is the decentralized miner, which interacts with the subtensor chain, dendrite, axon, and the model.
    """
    @classmethod
    def config(cls) -> bt.config: return init_config( cls )

    def __init__( self, config: bt.config = None):
        """
        Initialization method for DMiner class.
        """
        # Create config
        self.config = config or DMiner.config()

        # Turn on bittensor logging
        bt.logging( config = self.config )

        # Create wallet hotkey and coldkey.
        self.wallet = bt.wallet( config = self.config )
        self.wallet.create_if_non_existent()
        bt.logging.debug( 'wallet:', self.wallet.hotkey.ss58_address )

        # Init the neuron chain connection.
        self.subtensor = bt.subtensor( chain_endpoint = self.config.chain_endpoint )
        bt.logging.debug( 'subtensor:', self.subtensor )

        # Init the neuron dendrite client.
        self.dendrite = bt.dendrite( wallet = self.wallet )
        bt.logging.debug( 'dendrite: ', self.dendrite )

        # Init the axon and attach the forward functions.
        self.axon = bt.axon( 
            wallet = self.wallet, 
            config = self.config 
        ).attach( 
            forward_fn = self.get_grads 
        ).attach( 
            forward_fn = self.get_weights 
        )
        bt.logging.debug( 'axon: ', self.axon )

        # Build the dataloader
        self.dataloader, self.tokenizer = get_dataloader( self )

        # Load pre-trained model (weights)
        self.model = GPT2LMHeadModel.from_pretrained('gpt2', pad_token_id = self.tokenizer.eos_token_id)

        # Initialize the optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=1e-5)

        # Move the model to the GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Init wandb.
        init_wandb( self )    

    def get_grads( self, synapse: GetGrads ) -> GetGrads: get_grads( self, synapse )
    def get_weights( self, synapse: GetWeights ) -> GetWeights: get_weights( self, synapse )
    def run( self ): run( self )

if __name__ == "__main__":
    miner = DMiner()
    miner.run()

