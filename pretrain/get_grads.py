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

import gc
import torch
import typing
import random
from types import SimpleNamespace
import bittensor as bt
from pretrain.misc import get_online_uids

# Protocol Definition to get Gradients
class GetGrads( bt.Synapse ):
    """
    The GetGrads class is used to get the gradients of the model.
    It subclasses the bittensor Synapse.
    """
    # Gradients per variable in the model.
    grads: typing.Optional[typing.Dict[str, bt.Tensor]] = None

    # Data samples used to train these gradients.
    samples: typing.List[ int ] = None

    # Define deserialization function
    def deserialize( self ) -> SimpleNamespace:
        """
        Deserialize method converts the Bittensor gradients to Pytorch tensors.

        Returns:
        Dictionary with gradient tensors.
        """
        return SimpleNamespace(  
            samples = self.samples, 
            grads = { name: g.tensor() for name, g in self.grads.items() } if self.grads != None else None
        )

def get_grads( self, synapse: GetGrads ) -> GetGrads: 
    """
    Method to get the current gradients from the model.

    Args:
        synapse: GetGrads object that holds the gradients.

    Returns:
        GetGrads object with updated gradients.
    """
    self.total_grads_shared += 1
    self.total_samples_shared += len( self.global_accumulated_ids )
    synapse.samples = self.global_accumulated_ids
    synapse.grads = { name: bt.tensor(parameter.grad.clone()) for name, parameter in self.model.named_parameters() if parameter.grad is not None }
    return synapse

def merge_grads( self ):

    available_axons = []
    # Filter out all uids that we should not merge with.
    for uid in self.metagraph.uids:

        # Filter dead neurons.
        if self.current_block - self.metagraph.last_update[ uid ] > 100: 
            continue
        
        # Filter myself
        elif self.metagraph.hotkeys[ uid ] == self.wallet.hotkey.ss58_address: 
            continue

        # Filter axons I have merged with this round
        elif self.metagraph.hotkeys[ uid ] in self.hotkeys_seen_this_round: 
            continue

        # otherwise this peer is available to merge with
        else:
            available_axons.append( self.metagraph.axons[uid] )

    # Get axons to merge with.
    if len(available_axons) == 0: 
        bt.logging.debug('No available uid to average gradients with, continuing')

    # Merge gradients.
    axon_sample = random.choices( available_axons, k = min( self.config.grads_merge_max_k, len( available_axons ) ) )
    if len(axon_sample) > 0:
        _merge_grads( self, axon_sample )


def _merge_grads( self, axons: typing.List[ bt.axon ] ):
    """
    Function to average the gradients of a model across several online axons.
    """
    # Log that the gradient reduction process is starting
    bt.logging.debug('Reducing gradients.')
    self.wandb.log({'reduce_gradients_event': 1.0})

    # Use the dendrite's query function to retrieve the gradients for the online axons
    # If the query function only returns one dictionary, wrap it in a list for the later iteration
    bt.logging.info(f'Grad Querying: {axons}')
    responses = self.dendrite.query( axons, GetGrads() )
    if not isinstance(responses, list): responses = [responses]

    # Save hotkeys we have merged with this round already.
    for axon in axons: self.hotkeys_seen_this_round.add( axon.hotkey )

    # Filter out invalid grads.
    # Check that there are valid gradient dicts to average.
    valid_resps = [ resp for resp in responses if resp is not None and hasattr(resp, 'grads') and hasattr(resp, 'samples') and is_valid_grad_dict( self, resp.grads ) ]
    if len(valid_resps) == 0: 
        bt.logging.debug('There were no valid grad dicts to average with')
        return
    self.wandb.log( {'n_valid_grad_dicts': len(valid_resps)} )

    # Extend the number of samples accumulates
    new_samples = []
    for resp in valid_resps:
        new_samples.extend( resp.samples )
    self.global_accumulated_ids.extend( new_samples )
    self.remote_samples_accumulated += len( new_samples )

    # Average the grad dicts.
    avg_valid_grads_dict = average_grad_dicts( self, [ resp.grads for resp in valid_resps ] )

    # Apply the averaged gradients to the model's parameters
    apply_averaged_gradients( self, avg_valid_grads_dict )

    # Log the successful reduction of gradients
    bt.logging.debug(f'Successfully reduced {len(valid_resps)} grads with {len(new_samples)} samples.')
    self.wandb.log({'successfully_average_gradients': 1.0})

    # Sanity check delete memory
    del responses
    del avg_valid_grads_dict
    torch.cuda.empty_cache() # Clear cache if existent.
    gc.collect()

def apply_averaged_gradients(self, avg_valid_grads_dict: typing.Dict[str, torch.Tensor] ):
    """
    This function applies the averaged gradients from the input gradient dictionary to 
    the parameters of the model.

    Parameters:
        avg_valid_grads_dict (dict): A gradient dictionary with averaged gradients. This dictionary
            typically contains mappings from parameter names to corresponding gradients.
    """
    # Apply the averaged gradients to the model's parameters
    for name, param in self.model.named_parameters():
        # Only apply the gradients if the parameter exists in the averaged gradients dictionary
        if name in avg_valid_grads_dict:
            # If the parameter already has a gradient, add the averaged gradient to it
            # Otherwise, assign the averaged gradient as the parameter's gradient
            if param.grad is not None:
                param.grad += avg_valid_grads_dict[name].to( self.device )
            else:
                param.grad = avg_valid_grads_dict[name].clone().to( self.device )

def average_grad_dicts(self, valid_grad_dicts: typing.List[typing.Dict[str, torch.Tensor]], average: bool = False) -> typing.Dict[str, torch.Tensor]:
    """
    This function averages the gradients from a list of valid gradient dictionaries and 
    returns a new gradient dictionary with the averaged gradients.

    Parameters:
        valid_grad_dicts (list): A list of valid gradient dictionaries. Each gradient dictionary
            typically contains mappings from layer names to corresponding gradients.
        average (bool): If true, the gradients are averaged, rather summed.
    Returns:
        dict: A gradient dictionary with averaged gradients.
    """
    # Build average
    avg_valid_grads_dict = {}

    # Iterate over the keys in the model's state_dict (which should correspond to the parameter names)
    for key in self.model.state_dict().keys():
        # Find the gradients for the current parameter across all the returned dictionaries
        all_grads = [grad_dict[key].to(self.device) for grad_dict in valid_grad_dicts if key in grad_dict]

        # If there are no gradients for this parameter, skip it
        if len(all_grads) == 0:
            continue

        # Initialize an empty tensor on the correct device to hold the sum of gradients
        grad_sum = torch.zeros_like(self.model.state_dict()[key], device=self.device)

        # Add each grad tensor to the sum (this replaces stacking and taking the mean)
        for grad in all_grads:
            grad_sum += grad.to(self.device)

        # Divide the sum by the number of gradients to get the average (note: this is an in-place operation)
        if average:
            grad_sum.div_(len(all_grads))

        # Assign the averaged gradient to the averaged gradients dictionary
        avg_valid_grads_dict[key] = grad_sum.to(self.device)

        # Delete all_grads and grad_sum to free up memory
        del all_grads
        del grad_sum

    return avg_valid_grads_dict


def is_valid_grad_dict(self, grad_dict) -> bool:
    """
    This function checks whether a given grad_dict is valid for a PyTorch model.
    The grad_dict is valid if:
        - it is a dictionary and not empty
        - all keys in the grad_dict match the keys in the model's state_dict
        - all values in the grad_dict are float tensors
        - all elements of the tensors are finite
        - the shapes of tensors in the grad_dict match with corresponding tensors in the model's state_dict

    Parameters:
        grad_dict (dict): The gradient dictionary to validate. This dict typically contains
            mappings from layer names to corresponding gradients.

    Returns:
        bool: True if the grad_dict is valid, False otherwise.
    """
    # Check if the grad_dict is a non-empty dictionary
    if grad_dict is None or not isinstance(grad_dict, dict) or len(grad_dict.keys()) == 0:
        bt.logging.warning(f'Invalid grad_dict: Is None, empty or not a dict: {grad_dict}')
        return False

    # Iterate over all keys in the grad_dict
    for key in grad_dict.keys():

        # If the key is not in the model's state_dict, the input grad_dict is not valid
        if key not in self.model.state_dict().keys():
            bt.logging.warning(f'Invalid grad_dict: Grad dict keys do not match the model: {key}')
            return False

        # If the corresponding value is None, the input grad_dict is not valid
        if grad_dict[key] is None:
            bt.logging.warning(f'Invalid grad_dict: Grad is none: {grad_dict[key]}')
            return False

        # If the value is not a float tensor, the input grad_dict is not valid
        if not isinstance(grad_dict[key], (torch.FloatTensor, torch.cuda.FloatTensor)):
            bt.logging.warning(f'Invalid grad_dict: Grad is not float tensor: {grad_dict[key]}')
            return False

        # If any elements of the tensor are not finite, the input grad_dict is not valid
        if not torch.all(torch.isfinite(grad_dict[key])):
            bt.logging.warning(f'Invalid grad_dict: Grad is not finite: {grad_dict[key]}')
            return False

        # Ensure device is correct.
        grad_dict[key].to(self.device)

        # If the shape of the tensor does not match the corresponding tensor in the model,
        # the input grad_dict is not valid
        if grad_dict[key].shape != self.model.state_dict()[key].shape:
            bt.logging.warning(f"Invalid grad_dict: Grad dimensions do not match the model: {grad_dict[key].shape}")
            return False

    # If none of the above conditions are met, the grad_dict is valid
    return True


# Tests
import pytest
import unittest
import tracemalloc
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

class TestMergeGrads(unittest.TestCase):
    def setUp(self):
        # Set up a fake instance of your class that includes the merge_weights method
        self.instance = SimpleNamespace()

        # Mock the necessary attributes for the instance
        self.instance.metagraph = MagicMock()
        self.instance.dendrite = MagicMock()
        self.instance.model = MagicMock()
        self.instance.wandb = MagicMock()
        self.instance.device = 'cpu'
        self.instance.global_accumulated_ids = []

        # Set up some fake model weights
        self.model_weights = { 'fc1.weight': torch.randn(3, 3), 'fc2.weight': torch.randn(3, 3) }
        self.instance.model.state_dict.return_value = self.model_weights
        self.instance.model.named_parameters.return_value = [ ('fc1.weight', MagicMock()), ('fc2.weight', MagicMock()) ]
        for name, param in self.instance.model.named_parameters.return_value:
            param.grad = torch.randn(3, 3)
    
    def test_merge_valid_grad_dict(self):

        # Set up the mock to return a state_dict that does not include the weights for 'fc1'
        self.instance.dendrite.query.return_value = SimpleNamespace( samples =  [1], grads = {'fc1.weight': torch.randn(3, 3), 'fc2.weight': torch.randn(3, 3)} )

        # Valid grad dict.
        _merge_grads( self.instance, None )

        # Calculate the expected averaged grads
        expected_grads = {
        'fc1.weight': torch.stack(
            [
                self.model_weights['fc1.weight'], 
                self.instance.dendrite.query().grads['fc1.weight']
            ]).mean(dim=0),
        'fc2.weight': torch.stack(
            [
                self.model_weights['fc2.weight'], 
                self.instance.dendrite.query().grads['fc1.weight']
            ]).sum(dim=0)
        }

        # Verify the model's grads were updated correctly
        for name, param in self.instance.model.named_parameters:
            self.assertTrue(torch.allclose(expected_grads[name], param.grad), msg=f"updated weights for {key} are incorrect")

    def test_merge_valid_grad_dict_multiple(self):

        # Set up the mock to return a state_dict that does not include the weights for 'fc1'
        self.instance.dendrite.query.return_value = [
            SimpleNamespace( samples = [1], grads = {'fc1.weight': torch.randn(3, 3), 'fc2.weight': torch.randn(3, 3)}),
            SimpleNamespace( samples = [1], grads = {'fc1.weight': torch.randn(3, 3), 'fc2.weight': torch.randn(3, 3)})
        ]

        # Valid grad dict.
        _merge_grads( self.instance, None )

        # Calculate the expected averaged grads
        expected_grads = {
        'fc1.weight': torch.stack(
            [
                self.model_weights['fc1.weight'], 
                self.instance.dendrite.query()[0].grads['fc1.weight'],
                self.instance.dendrite.query()[1].grads['fc1.weight']

            ]).mean(dim=0),
        'fc2.weight': torch.stack(
            [
                self.model_weights['fc2.weight'], 
                self.instance.dendrite.query()[0].grads['fc2.weight'],
                self.instance.dendrite.query()[1].grads['fc2.weight']
            ]).sum(dim=0)
        }

        # Verify the model's grads were updated correctly
        for name, param in self.instance.model.named_parameters:
            self.assertTrue(torch.allclose(expected_grads[name], param.grad), msg=f"updated weights for {key} are incorrect")


    def test_merge_valid_grad_dict_multiple_some_wrong(self):

        # Set up the mock to return a state_dict that does not include the weights for 'fc1'
        self.instance.dendrite.query.return_value = [
            SimpleNamespace( samples = [1], grads = {'fc1.weight': torch.randn(3, 3), 'fc2.weight': torch.randn(3, 3)}),
            SimpleNamespace( samples = [1], grads = {'fc1.weight': torch.randn(3, 3), 'fc2.weight': torch.randn(3, 3)}),
            SimpleNamespace( samples = [1], grads = {'fc1.weight': torch.randn(3, 4), 'fc2.weight': torch.randn(3, 3)}),
            SimpleNamespace( samples = [1], grads = {'fc1.weight': None, 'fc2.weight': torch.randn(3, 3)}),
            SimpleNamespace( samples = [1], grads = None ),
        ]

        # Valid grad dict.
        _merge_grads( self.instance, None )

        # Calculate the expected averaged grads
        expected_grads = {
        'fc1.weight': torch.stack(
            [
                self.model_weights['fc1.weight'], 
                self.instance.dendrite.query()[0].grads['fc1.weight'],
                self.instance.dendrite.query()[1].grads['fc1.weight']

            ]).mean(dim=0),
        'fc2.weight': torch.stack(
            [
                self.model_weights['fc2.weight'], 
                self.instance.dendrite.query()[0].grads['fc2.weight'],
                self.instance.dendrite.query()[1].grads['fc2.weight']
            ]).sum(dim=0)
        }

        # Verify the model's grads were updated correctly
        for name, param in self.instance.model.named_parameters:
            self.assertTrue(torch.allclose(expected_grads[name], param.grad), msg=f"updated weights for {key} are incorrect")

    def test_return_is_none(self):
        # Set up the mock to return a state_dict that does not include the weights for 'fc1'
        self.instance.dendrite.query.return_value = None

        # No valid grad dicts.
        with pytest.raises(Exception) as excinfo:
            _merge_grads( self.instance, None )
        assert "There are no valid gradient dicts." in str(excinfo.value)

    def test_return_is_empty(self):
        # Set up the mock to return a state_dict that does not include the weights for 'fc1'
        self.instance.dendrite.query.return_value = {}

        # No valid grad dicts.
        with pytest.raises(Exception) as excinfo:
            _merge_grads( self.instance, None )
        assert "There are no valid gradient dicts." in str(excinfo.value)

    def test_return_is_invalid_name(self):
        # Set up the mock to return a state_dict that does not include the weights for 'fc1'
        self.instance.dendrite.query.return_value = SimpleNamespace( samples = [1], grads = {'fc1.weight': torch.randn(3, 3), 'fc2.weightkansdsa': torch.randn(3, 3)})

        # No valid grad dicts.
        with pytest.raises(Exception) as excinfo:
            _merge_grads( self.instance, None )
        assert "There are no valid gradient dicts." in str(excinfo.value)

    def test_return_is_invalid_none(self):
        # Set up the mock to return a state_dict that does not include the weights for 'fc1'
        self.instance.dendrite.query.return_value = SimpleNamespace( samples = [1], grads = {'fc1.weight': torch.randn(3, 3), 'fc2.weightkansdsa': None})

        # No valid grad dicts.
        with pytest.raises(Exception) as excinfo:
            _merge_grads( self.instance, None )
        assert "There are no valid gradient dicts." in str(excinfo.value)

    def test_return_is_invalid_dtype(self):
        # Set up the mock to return a state_dict that does not include the weights for 'fc1'
        self.instance.dendrite.query.return_value = SimpleNamespace( samples = [1], grads = {'fc1.weight': torch.randint(0, 10, (3, 3), dtype=torch.int64), 'fc2.weightkansdsa':  torch.randint(0, 10, (3, 3), dtype=torch.int64)})

        # No valid grad dicts.
        with pytest.raises(Exception) as excinfo:
            _merge_grads( self.instance, None )
        assert "There are no valid gradient dicts." in str(excinfo.value)

    def test_invalid_dimension(self):
        # Set up the mock to return a state_dict that does not include the weights for 'fc1'
        self.instance.dendrite.query.return_value = SimpleNamespace( samples = [1], grads = {'fc1.weight': torch.randn(3, 4), 'fc2.weight': torch.randn(3, 3)} )

        # No valid grad dicts.
        with pytest.raises(Exception) as excinfo:
            _merge_grads( self.instance, None )
        assert "There are no valid gradient dicts." in str(excinfo.value)

    def test_samples_accumulate(self):
        # Set up the mock to return a state_dict that does not include the weights for 'fc1'
        self.instance.dendrite.query.return_value = [
                SimpleNamespace( samples = [1], grads = {'fc1.weight': torch.randn(3, 3), 'fc2.weight': torch.randn(3, 3)} ),
                SimpleNamespace( samples = [1], grads = {'fc1.weight': torch.randn(3, 3), 'fc2.weight': torch.randn(3, 3)} ),
                SimpleNamespace( samples = [1], grads = {'fc1.weight': torch.randn(3, 3), 'fc2.weight': torch.randn(3, 3)} ),
                SimpleNamespace( samples = [1], grads = {'fc1.weight': torch.randn(3, 3), 'fc2.weight': torch.randn(3, 3)} ),
                SimpleNamespace( samples = [1], grads = {'fc1.weight': torch.randn(3, 3), 'fc2.weight': torch.randn(3, 3)} ),
                SimpleNamespace( samples = [1], grads = {'fc1.weight': torch.randn(3, 3), 'fc2.weight': torch.randn(3, 3)} ),
            ]

        # Merge grads.
        _merge_grads( self.instance, None )

        # Check that the samples have accumulated
        assert len( self.instance.global_accumulated_ids ) == 6

    
if __name__ == "__main__":
    unittest.main()