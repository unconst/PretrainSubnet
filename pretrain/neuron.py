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
from pretrain.dataset import Dataset
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
        bt.logging.info( self.config )
        bt.logging.debug( 'logdir:', self.config.full_path )

        # Create wallet hotkey and coldkey.
        self.wallet = bt.wallet( config = self.config )
        self.wallet.create_if_non_existent()
        bt.logging.debug( 'wallet:', self.wallet )

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

        self.device = self.config.device
        bt.logging.debug( 'device: ', self.device )

        # Build the dataloader
        self.dataset = Dataset( self.config ) 
        bt.logging.debug( 'dataset: ', self.dataset )

        # Load pre-trained model (weights)
        self.model = GPT2LMHeadModel.from_pretrained('gpt2', pad_token_id = self.dataset.tokenizer.eos_token_id)
        bt.logging.debug( 'model: ', self.model )

        # Initialize the optimizer
        self.optimizer = torch.optim.AdamW( self.model.parameters(), lr = self.config.lr )
        bt.logging.debug( 'optimizer: ', self.optimizer )

        # Move the model to the GPU if available
        self.model.to( self.device )

        # Init wandb.
        init_wandb( self )    

    def get_grads( self, synapse: GetGrads ) -> GetGrads: 
        return get_grads( self, synapse )

    def get_weights( self, synapse: GetWeights ) -> GetWeights: 
        return get_weights( self, synapse )

    def run( self ): 
        run( self )

if __name__ == "__main__":
    miner = DMiner()
    miner.run()

