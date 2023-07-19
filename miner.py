
import sys
import time
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
from torchvision import datasets, transforms

# bt.debug()

CHAIN_ENDPOINT = "wss://test.finney.opentensor.ai"
NETUID = 97

# Define the model architecture.
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Protocol
class GetGrads( bt.Synapse ):
    # Grads per variable in model.
    grads: Optional[ typing.Dict[ str, bt.Tensor ] ] = None
    # define deserialization function
    def deserialize( self ) -> dict[ str, torch.FloatTensor ]:
        if self.grads: return { name: g.tensor() for name, g in self.grads.items() }
        else: return {}

# Define get weights.
class GetWeights( bt.Synapse ):
    # Grads per variable in model.
    weights: Optional[ typing.Dict[ str, bt.Tensor ] ] = None
    # define deserialization function
    def deserialize( self ) -> dict[ str, torch.FloatTensor ]:
        if self.weights:
            return { name: w.tensor() for name, w in self.weights.items() }
        else: {}

# The miner.
class DMiner:

    @classmethod
    def config(cls) -> bt.config:
        parser = argparse.ArgumentParser()
        bt.wallet.add_args( parser )
        bt.axon.add_args( parser )
        config = bt.config( parser )
        return config

    def __init__( self ):

        # Create config
        self.config = DMiner.config()

        # Create wallet hotkey and coldkey.
        self.wallet = bt.wallet( config = self.config )
        self.wallet.create_if_non_existent()

        # Bittensor objects.
        self.subtensor = bt.subtensor( chain_endpoint = CHAIN_ENDPOINT )
        self.axon = bt.axon( wallet = self.wallet, config = self.config )
        self.dendrite = bt.dendrite( wallet = self.wallet )

        # Model objects.
        self.model = Model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = optim.SGD( self.model.parameters(), lr = 0.01, momentum = 0.5 )
        self.criterion = nn.CrossEntropyLoss()
        self.saved_grads = None

        # Send model to device
        self.model.to(self.device)

        # Training hyper-parameters
        self.epoch_length = 2
        self.batch_size = 1000

        # Training data.
        self.train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
        self.test_data = datasets.MNIST(root='./data', train=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
        self.train_loader = torch.utils.data.DataLoader(self.train_data, batch_size = self.batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_data, batch_size = 1000, shuffle=False)

    def get_saved_grads( self, synapse: GetGrads ) -> GetGrads:
        synapse.grads = self.saved_grads
        return synapse

    def get_weights( self, synapse: GetWeights ) -> GetWeights:
        synapse.weights = { name: bt.tensor( weight ) for name, weight in self.model.state_dict().items() }
        return synapse

    # Returns true if we are in a training state.
    def is_training(self) -> bool:
        return self.subtensor.block % self.epoch_length != 0

    def apply_remote_grads(self):
        # Get the saved grads from everyone.
        grads = self.dendrite.query( self.metagraph.axons, GetGrads(), timeout = 5 )

        # Apply all grads to the model.
        for _, grads_dict in list(zip(self.metagraph.axons, grads)):
            # Loop over named parameters of the model
            for name, param in self.model.named_parameters():
                # If there's a corresponding gradient in your dictionary, apply it
                if name in grads_dict:
                    grads_dict[name] = grads_dict[name].to(self.device)
                    if param.grad is not None:
                        param.grad += grads_dict[name]
                    else:
                        param.grad = grads_dict[name].clone()

    def average_weights_across_miners(self):
        # Get the saved grads from everyone.
        state_dicts = self.dendrite.query( self.metagraph.axons, GetWeights(), timeout = 5 )
        state_dicts = [s for s in state_dicts if s != None]
        self.model.load_state_dict({k: sum(d[k] for d in state_dicts) / len(state_dicts) for k in state_dicts[0] if k != None })

    def run( self ):

        # Register wallet.
        self.subtensor.register( wallet = self.wallet, netuid = NETUID )

        # Build, attach, serve and start axon.
        self.axon.attach( 
            forward_fn = self.get_saved_grads 
        ).attach( 
            forward_fn = self.get_weights 
        ).serve( 
            netuid = NETUID, 
            subtensor = self.subtensor
        ).start()

        # Get the current network state.
        self.metagraph = self.subtensor.metagraph( NETUID )

        # Take network Average weights.
        self.average_weights_across_miners()

        step = 0
        while True:

            # Get the current network state.
            self.metagraph = self.subtensor.metagraph( NETUID )

            self.model.train()
            self.optimizer.zero_grad()

            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)

                # Create grads.
                self.optimizer.zero_grad()
                output = self.model( data )
                loss = self.criterion( output, target )
                loss.backward()

                # Save grads.
                self.saved_grads = { name: bt.tensor( parameter.grad.clone() ) for name, parameter in self.model.named_parameters() if parameter.grad is not None }
                
                # Zero grads.
                self.optimizer.zero_grad()
                
                # Apply remote grads.
                self.apply_remote_grads()

                # Apply step.
                self.optimizer.step()

                bt.logging.info(f"Loss: {loss.item()}")

            # average weights.
            self.average_weights_across_miners()

            # Get the test scoring.
            # if step % 10 == 0:
            print (f'Test...')
            self.model.eval()
            test_loss = 0
            correct = 0
            with torch.no_grad():
                for data, target in self.test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    test_loss += self.criterion(output, target).item()
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
            test_loss /= len(self.test_loader.dataset)
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(self.test_loader.dataset), 100. * correct / len(self.test_loader.dataset)))

            # Wait until it is training time again.
            step += 1
            while not self.is_training():
                print (f'wait..')
                time.sleep(1)


if __name__ == "__main__":
    miner = DMiner()
    miner.run()

