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

def merge_grads( self ):


    try:
        # Get axons to merge with.
        online_axons = [self.metagraph.axons[uid] for uid in get_online_uids( self )]
        if len(online_axons) == 0: raise Exception('There are no online uids to average gradients with.')

        # Merge gradients.
        _merge_grads( self, online_axons )

    # On error log invalid step.
    except Exception as e:
        bt.logging.error( f'Failed to merge grads with error {e}')
        self.wandb.log({'successfully_average_gradients': 0.0})

def _merge_grads( self, axons: typing.List[ bt.axon ]  ):
    """
    Function to average the gradients of a model across several online axons.
    """
    # Log that the gradient reduction process is starting
    bt.logging.info('Reducing gradients.')
    self.wandb.log({'reduce_gradients_event': 1.0})

    # Use the dendrite's query function to retrieve the gradients for the online axons
    # If the query function only returns one dictionary, wrap it in a list for the later iteration
    grad_dicts = self.dendrite.query( axons, GetGrads())
    if not isinstance(grad_dicts, list): grad_dicts = [grad_dicts]

    # Filter out invalid grads.
    # Check that there are valid gradient dicts to average.
    valid_grad_dicts = [ grad_dict for grad_dict in grad_dicts if is_valid_grad_dict( self, grad_dict ) ]
    if len(valid_grad_dicts) == 0: raise Exception('There are no valid gradient dicts.')
    self.wandb.log( {'n_valid_grad_dicts': len(valid_grad_dicts)} )

    # Average the grad dicts.
    avg_valid_grads_dict = average_grad_dicts( self, valid_grad_dicts )

    # Apply the averaged gradients to the model's parameters
    apply_averaged_gradients( self, avg_valid_grads_dict )

    # Log the successful reduction of gradients
    bt.logging.success(f'Successfully reduced {len(grad_dicts)} grads.')
    self.wandb.log({'successfully_average_gradients': 1.0})

    # Sanity check delete memory
    del grad_dicts
    del valid_grad_dicts
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

def average_grad_dicts(self, valid_grad_dicts: typing.List[typing.Dict[str, torch.Tensor]]) -> typing.Dict[str, torch.Tensor]:
    """
    This function averages the gradients from a list of valid gradient dictionaries and 
    returns a new gradient dictionary with the averaged gradients.

    Parameters:
        valid_grad_dicts (list): A list of valid gradient dictionaries. Each gradient dictionary
            typically contains mappings from layer names to corresponding gradients.

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

        # Set up some fake model weights
        self.model_weights = { 'fc1.weight': torch.randn(3, 3), 'fc2.weight': torch.randn(3, 3) }
        self.instance.model.state_dict.return_value = self.model_weights
        self.instance.model.named_parameters.return_value = [ ('fc1.weight', MagicMock()), ('fc2.weight', MagicMock()) ]
        for name, param in self.instance.model.named_parameters.return_value:
            param.grad = torch.randn(3, 3)
    
    def test_merge_valid_grad_dict(self):

        # Set up the mock to return a state_dict that does not include the weights for 'fc1'
        self.instance.dendrite.query.return_value = {'fc1.weight': torch.randn(3, 3), 'fc2.weight': torch.randn(3, 3)}

        # Valid grad dict.
        _merge_grads( self.instance, None )

        # Calculate the expected averaged grads
        expected_grads = {
        'fc1.weight': torch.stack(
            [
                self.model_weights['fc1.weight'], 
                self.instance.dendrite.query()['fc1.weight']
            ]).mean(dim=0),
        'fc2.weight': torch.stack(
            [
                self.model_weights['fc2.weight'], 
                self.instance.dendrite.query()['fc1.weight']
            ]).mean(dim=0)
        }

        # Verify the model's grads were updated correctly
        for name, param in self.instance.model.named_parameters:
            self.assertTrue(torch.allclose(expected_grads[name], param.grad), msg=f"updated weights for {key} are incorrect")

    def test_merge_valid_grad_dict_multiple(self):

        # Set up the mock to return a state_dict that does not include the weights for 'fc1'
        self.instance.dendrite.query.return_value = [
             {'fc1.weight': torch.randn(3, 3), 'fc2.weight': torch.randn(3, 3)},
             {'fc1.weight': torch.randn(3, 3), 'fc2.weight': torch.randn(3, 3)}
        ]

        # Valid grad dict.
        _merge_grads( self.instance, None )

        # Calculate the expected averaged grads
        expected_grads = {
        'fc1.weight': torch.stack(
            [
                self.model_weights['fc1.weight'], 
                self.instance.dendrite.query()[0]['fc1.weight'],
                self.instance.dendrite.query()[1]['fc1.weight']

            ]).mean(dim=0),
        'fc2.weight': torch.stack(
            [
                self.model_weights['fc2.weight'], 
                self.instance.dendrite.query()[0]['fc2.weight'],
                self.instance.dendrite.query()[1]['fc2.weight']
            ]).mean(dim=0)
        }

        # Verify the model's grads were updated correctly
        for name, param in self.instance.model.named_parameters:
            self.assertTrue(torch.allclose(expected_grads[name], param.grad), msg=f"updated weights for {key} are incorrect")


    def test_merge_valid_grad_dict_multiple_some_wrong(self):

        # Set up the mock to return a state_dict that does not include the weights for 'fc1'
        self.instance.dendrite.query.return_value = [
            {'fc1.weight': torch.randn(3, 3), 'fc2.weight': torch.randn(3, 3)},
            {'fc1.weight': torch.randn(3, 3), 'fc2.weight': torch.randn(3, 3)},
            {'fc1.weight': torch.randn(3, 4), 'fc2.weight': torch.randn(3, 3)},
            {'fc1.weight': None, 'fc2.weight': torch.randn(3, 3)},
            None,
        ]

        # Valid grad dict.
        _merge_grads( self.instance, None )

        # Calculate the expected averaged grads
        expected_grads = {
        'fc1.weight': torch.stack(
            [
                self.model_weights['fc1.weight'], 
                self.instance.dendrite.query()[0]['fc1.weight'],
                self.instance.dendrite.query()[1]['fc1.weight']

            ]).mean(dim=0),
        'fc2.weight': torch.stack(
            [
                self.model_weights['fc2.weight'], 
                self.instance.dendrite.query()[0]['fc2.weight'],
                self.instance.dendrite.query()[1]['fc2.weight']
            ]).mean(dim=0)
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
        self.instance.dendrite.query.return_value = {'fc1.weight': torch.randn(3, 3), 'fc2.weightkansdsa': torch.randn(3, 3)}

        # No valid grad dicts.
        with pytest.raises(Exception) as excinfo:
            _merge_grads( self.instance, None )
        assert "There are no valid gradient dicts." in str(excinfo.value)

    def test_return_is_invalid_none(self):
        # Set up the mock to return a state_dict that does not include the weights for 'fc1'
        self.instance.dendrite.query.return_value = {'fc1.weight': torch.randn(3, 3), 'fc2.weightkansdsa': None}

        # No valid grad dicts.
        with pytest.raises(Exception) as excinfo:
            _merge_grads( self.instance, None )
        assert "There are no valid gradient dicts." in str(excinfo.value)

    def test_return_is_invalid_dtype(self):
        # Set up the mock to return a state_dict that does not include the weights for 'fc1'
        self.instance.dendrite.query.return_value = {'fc1.weight': torch.randint(0, 10, (3, 3), dtype=torch.int64), 'fc2.weightkansdsa':  torch.randint(0, 10, (3, 3), dtype=torch.int64)}

        # No valid grad dicts.
        with pytest.raises(Exception) as excinfo:
            _merge_grads( self.instance, None )
        assert "There are no valid gradient dicts." in str(excinfo.value)

    def test_invalid_dimension(self):
        # Set up the mock to return a state_dict that does not include the weights for 'fc1'
        self.instance.dendrite.query.return_value = {'fc1.weight': torch.randn(3, 4), 'fc2.weight': torch.randn(3, 3)}

        # No valid grad dicts.
        with pytest.raises(Exception) as excinfo:
            _merge_grads( self.instance, None )
        assert "There are no valid gradient dicts." in str(excinfo.value)


if __name__ == "__main__":
    unittest.main()