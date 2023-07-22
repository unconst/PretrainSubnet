
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
from torchvision import datasets, transforms

# Define the constant chain endpoint and network unique ID
CHAIN_ENDPOINT = "wss://test.finney.opentensor.ai"
NETUID = 97


# Model Definition
class Model(nn.Module):
    """
    Pytorch Convolutional Neural Network Model definition.

    It comprises of 2 convolutional layers, followed by 2 linear layers. Dropout is used for regularization.
    """
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        """
        Forward method for the model. Defines the computations performed at every call.
        
        Args:
        x: the input data.

        Returns:
        Output after passing input through the model.
        """
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# Protocol Definition to get Gradients
class GetGrads( bt.Synapse ):
    """
    The GetGrads class is used to get the gradients of the model.
    It subclasses the bittensor Synapse.
    """
    # Gradients per variable in the model.
    grads: Optional[typing.Dict[str, bt.Tensor]] = None

    # Define deserialization function
    def deserialize( self ) -> dict[ str, torch.FloatTensor ]:
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
    def deserialize( self ) -> dict[ str, torch.FloatTensor ]:
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

        # Initialize model, optimizer and criterion
        self.model = Model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = optim.SGD( self.model.parameters(), lr = 0.01, momentum = 0.5 )
        self.criterion = nn.CrossEntropyLoss()
        self.saved_grads = None
        bt.logging.debug( 'device:', self.device )
        bt.logging.debug( 'optimizer: ', self.optimizer )
        bt.logging.debug( 'criterion: ', self.criterion )
        bt.logging.debug( 'model: \n', self.model )

        # Send model to device
        self.model.to(self.device)

        # Define training hyperparameters
        self.batch_size = 1000

        # Define training and testing data
        self.train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
        self.test_data = datasets.MNIST(root='./data', train=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
        self.train_loader = torch.utils.data.DataLoader(self.train_data, batch_size = self.batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_data, batch_size = 1000, shuffle=False)
        bt.logging.debug( 'train_data: ', self.train_data )
        bt.logging.debug( 'test_data: ', self.test_data )

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
        bt.logging.info( 'Reducing gradients.')
        wandb.log({ 'reduce_gradients_event': 1.0 })

        # Get all online axons..
        online_axons = [self.metagraph.axons[uid] for uid in self.get_online_uids() ]
        if len( online_axons ) == 0: return
        bt.logging.info(f'Reducing grads with uids: {self.get_online_uids()}')

        # Get the saved grads from everyone.
        grad_dicts = self.dendrite.query( online_axons, GetGrads(), timeout = 5)
        if not isinstance(grad_dicts, list ): grad_dicts = [grad_dicts]

        # Get all valid s_dics.
        valid_grads_dicts = []
        for grad_dict in grad_dicts:
            valid = True
            for key in self.model.state_dict().keys():
                if key not in grad_dict: 
                    bt.logging.warning( f'{key} not in {grad_dict}')
                    valid = False
            if valid: valid_grads_dicts.append( grad_dict )

        # Check for no-valid dicts.
        if len( valid_grads_dicts ) == 0: 
            bt.logging.warning( 'No valid gradient dicts')
            return 
        wandb.log({ 'n_valid_gradients': len(valid_grads_dicts) })

        # Create a new state dictionary for the averaged grads
        avg_valid_grads_dict = {}
        for key in self.model.state_dict().keys():

            # stack the weights along a new dimension, and take their mean
            avg_grad = torch.stack([grad_dict[key] for grad_dict in valid_grads_dicts]).mean(dim=0).to(self.device)

            # assign the average weight to the new state dictionary
            avg_valid_grads_dict[key] = avg_grad

        # Apply avg valid grads to model.
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param.grad += avg_valid_grads_dict[name]
            else:
                param.grad = avg_valid_grads_dict[name].clone()

        bt.logging.success(f'Successfully reduced {len(avg_valid_grads_dict)} grads.') 
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
        bt.logging.info( 'Avergating weights')
        wandb.log({ 'average_weights_event': 1.0 })

        # Get all online axons..
        online_axons = [self.metagraph.axons[uid] for uid in self.get_online_uids() ]
        if len( online_axons ) == 0: return
        bt.logging.info(f'Averaging weights with uids: {self.get_online_uids()}')

        # Query all miners for their model weights.
        state_dicts = self.dendrite.query( online_axons, GetWeights(), timeout = 5 )

        # Get the keys of the state dictionary (i.e., the parameter names)
        keys = self.model.state_dict().keys()

        # Get all valid state_dics.
        valid_state_dics = []
        for state_dict in state_dicts:
            valid = True
            for key in self.model.state_dict().keys():
                if key not in state_dict: valid = False
            if valid: valid_state_dics.append( state_dict )

        if len( valid_state_dics ) == 0:
            bt.logging.warning('No valid weights to average.') 
            return 

        # Create a new state dictionary for the averaged weights
        avg_state_dict = {}
        for key in self.model.state_dict().keys():

            # stack the weights along a new dimension, and take their mean
            avg_weight = torch.stack([state_dict[key] for state_dict in valid_state_dics]).mean(dim=0)

            # assign the average weight to the new state dictionary
            avg_state_dict[key] = avg_weight

        # Load the averaged weights into the local model.
        self.model.load_state_dict(avg_state_dict)
        bt.logging.success(f'Successfully averaged {len(valid_state_dics)} weights.') 
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
        self.subtensor.set_weights( 
            netuid = NETUID, 
            wallet = self.wallet, 
            uids = [self.my_uid], 
            weights = [1.0]
        )
        bt.logging.info( 'Set weights ')

        # Average model weights across all miners in the network.
        self.average_weights_across_miners()

        # Initialize step counter
        global_step = 0
        training_step = 0

        # Training loop
        while True:
            # Fetch the current network state (metagraph) from Subtensor.
            self.metagraph = self.subtensor.metagraph(NETUID)
            bt.logging.info( 'Online UIDS', self.get_online_uids() )
            wandb.log({ 'n_online': len(self.get_online_uids()) })

            # Set the model to training mode.
            self.model.train()

            # Clear out all existing gradients in the model.
            self.optimizer.zero_grad()

            # Iterate over batches of training data.
            for batch_idx, (data, target) in enumerate(self.train_loader):

                # Zero out gradients calculated in the previous iteration.
                # and save them for others to query.
                self.saved_grads = { name: bt.tensor(parameter.grad.clone()) for name, parameter in self.model.named_parameters() if parameter.grad is not None }
                self.optimizer.zero_grad()

                # Move the batch tensors to the same device as the model.
                data, target = data.to(self.device), target.to(self.device)

                # Forward pass through the model to compute outputs.
                output = self.model( data )

                # Compute loss between model output and true targets.
                loss = self.criterion( output, target )

                # Backward pass to compute gradients of the loss with respect to model parameters.
                loss.backward()

                # Retrieve and apply gradients from other miners.
                self.apply_remote_grads()

                # Take an optimization step using the aggregated gradients.
                self.optimizer.step()

                # Log the loss value for this batch.
                training_step += 1
                wandb.log({ 'block': self.subtensor.block })
                wandb.log({ 'training_step': training_step })
                bt.logging.info(f"Loss: {loss.item()}") 
                wandb.log({ 'train_loss': loss })

            # After going through all batches, average the model weights across all miners.
            self.average_weights_across_miners()

            # Evaluate the model on the test dataset.
            # TODO: Make the frequency of testing configurable (i.e., not every epoch).
            self.model.eval()
            test_loss = 0
            correct = 0
            with torch.no_grad():
                for data, target in self.test_loader:
                    # Move the batch tensors to the same device as the model.
                    data, target = data.to(self.device), target.to(self.device)

                    # Forward pass through the model to compute outputs.
                    output = self.model(data)

                    # Compute and aggregate loss on the test dataset.
                    test_loss += self.criterion(output, target).item()

                    # Compute number of correct predictions.
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()

            # Compute average test loss and accuracy.
            test_loss /= len(self.test_loader.dataset)
            test_accuracy = 100. * correct / len(self.test_loader.dataset)
            wandb.log({ 'test_loss': test_loss })
            wandb.log({ 'test_accuracy': test_accuracy })

            # Log test loss and accuracy.
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(self.test_loader.dataset), test_accuracy))

            # Set ping weights.
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
