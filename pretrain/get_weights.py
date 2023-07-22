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

# Protocol Definition to get Weights
class GetWeights( bt.Synapse ):
    """
    The GetWeights class is used to get the weights of the model.
    It subclasses the bittensor Synapse.
    """
    # Weights per variable in the model.
    weights: typing.Optional[ typing.Dict[ str, bt.Tensor ] ] = None

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

def merge_weights( self ):
    try:

        # Get axons to merge weights with.
        online_axons = [self.metagraph.axons[uid] for uid in get_online_uids( self )]
        if len(online_axons) == 0: raise Exception('There are no online uids to average weights with.')

        # Merge weights.
        _merge_weights( self, online_axons)

    except Exception as e:
        bt.logging.error( f'Failed to merge weights with error {e}')
        self.wandb.log({ 'average_weights_event': 0.0 })

def _merge_weights(self, axons: typing.List[ bt.axon ] ):
    """
    This method retrieves the model weights from all miners in the network and then averages 
    these weights. The averaged weights are then used to update the local model.
    """

    # Log the start of the weight averaging process.
    bt.logging.info('Starting the weight averaging process.')
    self.wandb.log({ 'average_weights_event': 1.0 })

    # Query all miners for their model weights.
    # If only one model's weights were returned, make sure it's in a list.
    state_dicts = self.dendrite.query( axons, GetWeights() )    
    if not isinstance(state_dicts, list): state_dicts = [state_dicts]

    # Filter out invalid weights.
    valid_state_dicts = [state_dict for state_dict in state_dicts if is_valid_state_dict( self, state_dict )] 
    if len(valid_state_dicts) == 0: raise Exception('There are no valid weights dicts.')
    self.wandb.log( {'n_valid_weight_dicts': len(valid_state_dicts)} )

    # Average and apply the valid state dicts to the model.
    avg_state_dict = average_state_dicts( self, valid_state_dicts )
    self.model.load_state_dict( avg_state_dict )
    
    # Log the weights average success.
    bt.logging.success(f'Successfully averaged {len(state_dicts)} weights.') 
    self.wandb.log({ 'successfully_average_weights': 1.0 })

    # Sanity check delete memory
    del state_dicts
    del valid_state_dicts
    del avg_state_dict
    torch.cuda.empty_cache() # Clear cache if existent.
    gc.collect()

def average_state_dicts( self, valid_state_dicts: typing.List[typing.Dict[str, torch.Tensor]]) -> typing.Dict[str, torch.Tensor]:
    """
    This function averages the weights from a list of valid state dictionaries and 
    returns a new state dictionary with the averaged weights.

    Parameters:
        valid_state_dicts (list): A list of valid state dictionaries. Each state dictionary
            typically contains mappings from layer names to corresponding parameters.

    Returns:
        dict: A state dictionary with averaged weights.
    """
    # Create a new state dictionary for the averaged weights.
    avg_state_dict = {}

    for key in self.model.state_dict().keys():
        # Initialize accumulation variables.
        total_weights = torch.zeros_like(self.model.state_dict()[key]).to(self.device)
        num_weights = 0

        # Accumulate weights and their count.
        for state_dict in valid_state_dicts:
            if key in state_dict:
                total_weights += state_dict[key].to(self.device)
                num_weights += 1

        # Skip this key if there are no weights available.
        if num_weights == 0: 
            continue

        # Compute the average of the weights.
        avg_weight = total_weights / num_weights

        # Assign the average weight to the new state dictionary.
        avg_state_dict[key] = avg_weight.to(self.device)

        del total_weights
        del num_weights
    
    return avg_state_dict


def is_valid_state_dict(self, state_dict) -> bool:
    """
    This function checks whether a given state_dict is valid for a PyTorch model.
    The state_dict is valid if:
        - it is a dictionary and not empty
        - all keys in the state_dict match the keys in the model's state_dict
        - all values in the state_dict are float tensors
        - all elements of the tensors are finite
        - the shapes of tensors in the state_dict match with corresponding tensors in the model's state_dict

    Parameters:
        state_dict (dict): The state dictionary to validate. This dict typically contains
            mappings from layer names to corresponding parameters.

    Returns:
        bool: True if the state_dict is valid, False otherwise.
    """
    # Check if the state_dict is a non-empty dictionary
    if state_dict is None or not isinstance(state_dict, dict) or len(state_dict.keys()) == 0:
        bt.logging.warning(f'Invalid state_dict: Is None, empty or not a dict: {state_dict}')
        return False

    # Iterate over all keys in the state_dict
    for key in state_dict.keys():

        # If the key is not in the model's state_dict, the input state_dict is not valid
        if key not in self.model.state_dict().keys():
            bt.logging.warning(f'Invalid state_dict: Keys do not match the model: {key}')
            return False

        # If the corresponding value is None, the input state_dict is not valid
        if state_dict[key] is None:
            bt.logging.warning(f'Invalid state_dict: Weight is none: {state_dict[key]}')
            return False

        # If the value is not a float tensor, the input state_dict is not valid
        if not isinstance(state_dict[key], (torch.FloatTensor, torch.cuda.FloatTensor)):
            bt.logging.warning(f'Invalid state_dict: Weight is not float tensor: {state_dict[key]}')
            return False

        # If any elements of the tensor are not finite, the input state_dict is not valid
        if not torch.all(torch.isfinite(state_dict[key])):
            bt.logging.warning(f'Invalid state_dict: Weight is not finite: {state_dict[key]}')
            return False

        # Ensure device is correct.
        state_dict[key].to(self.device)

        # If the shape of the tensor does not match the corresponding tensor in the model, 
        # the input state_dict is not valid
        if state_dict[key].shape != self.model.state_dict()[key].shape:
            bt.logging.warning(f"Invalid state_dict: Weight dimensions do not match the model: {state_dict[key].shape}")
            return False

    # If none of the above conditions are met, the state_dict is valid
    return True

# Testss
import pytest
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

class TestMergeWeights(unittest.TestCase):
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
        self.model_weights = {'fc1.weight': torch.randn(3, 3), 'fc1.bias': torch.randn(3)}

    def test_no_weights_found(self):
        # Set up the model to have some weights
        self.instance.model.state_dict.return_value = self.model_weights

        # Set up the mock to return a state_dict that does not include the weights for 'fc1'
        self.instance.dendrite.query.return_value = {'fc2.weight': torch.randn(3, 3), 'fc2.bias': torch.randn(3)}

        # Run the function
        with pytest.raises(Exception) as excinfo:
            _merge_weights( self.instance, None )
        assert "There are no valid weights dicts." in str(excinfo.value)

    def test_successful_average(self):
        # Set up the model to have some weights
        self.instance.model.state_dict.return_value = self.model_weights

        # Set up the mock to return a state_dict that does include the weights for 'fc1'
        self.instance.dendrite.query.return_value = self.model_weights

        # Run the function
        _merge_weights( self.instance, None )

    def test_wrong_model_weights(self):
        # Set up the model to have some weights
        self.instance.model.state_dict.return_value = self.model_weights

        # Set up the mock to return a state_dict with weights for a different model
        wrong_model_weights = {'conv1.weight': torch.randn(3, 3, 3), 'conv1.bias': torch.randn(3)}
        self.instance.dendrite.query.return_value = wrong_model_weights

        # Run the function
        with pytest.raises(Exception) as excinfo:
            _merge_weights( self.instance, None )
        assert "There are no valid weights dicts." in str(excinfo.value)

    def test_empty_model_weights(self):
        # Set up the model to have some weights
        self.instance.model.state_dict.return_value = self.model_weights

        # Set up the mock to return a state_dict with weights for a different model
        empty_model_weights = {}
        self.instance.dendrite.query.return_value = empty_model_weights

        # Run the function
        with pytest.raises(Exception) as excinfo:
            _merge_weights( self.instance, None )
        assert "There are no valid weights dicts." in str(excinfo.value)

    def test_multiple_weights(self):
        # Set up the model to have some weights
        self.instance.model.state_dict.return_value = self.model_weights

        # Set up the mock to return a state_dict with weights for a different model
        multiple_responses = [ {'fc1.weight': torch.randn(3, 3), 'fc1.bias': torch.randn(3)}, {'fc1.weight': torch.randn(3, 3), 'fc1.bias': torch.randn(3)} ]
        self.instance.dendrite.query.return_value = multiple_responses

        # Run the function
        _merge_weights( self.instance, None )

    def test_multiple_weights_some_wrong(self):
        # Set up the model to have some weights
        self.instance.model.state_dict.return_value = self.model_weights

        # Set up the mock to return a state_dict with weights for a different model
        multiple_responses_some_wrong = [ 
            {'fc1.weight': torch.randn(3, 3), 'fc1.bias': torch.randn(3)}, 
            {'conv1.weight': torch.randn(3, 3, 3), 'conv1.bias': torch.randn(3)}, 
            {} 
        ]
        self.instance.dendrite.query.return_value = multiple_responses_some_wrong

        # Run the function
        _merge_weights( self.instance, None )

    def test_merged_weights_are_average(self):
        # Set up the model to have some weights
        self.instance.model.state_dict.return_value = self.model_weights

        # Define some weights that will be returned by the dendrite
        model_weights1 = {'fc1.weight': torch.randn(3, 3), 'fc1.bias': torch.randn(3)}
        model_weights2 = {'fc1.weight': torch.randn(3, 3), 'fc1.bias': torch.randn(3)}
        dendrite_weights = [model_weights1, model_weights2]

        # Set up the mock to return these weights
        self.instance.dendrite.query.return_value = dendrite_weights

        # Run the function
        _merge_weights( self.instance, None )

        # Calculate the expected averaged weights
        expected_weights = {'fc1.weight': torch.stack([model_weights1['fc1.weight'], model_weights2['fc1.weight']]).mean(dim=0),
                            'fc1.bias': torch.stack([model_weights1['fc1.bias'], model_weights2['fc1.bias']]).mean(dim=0)}

        # Verify the model's weights were updated correctly
        updated_weights = self.instance.model.load_state_dict.call_args[0][0]  # Get the first argument passed to load_state_dict
        for key in expected_weights.keys():
            self.assertTrue(torch.allclose(updated_weights[key], expected_weights[key]), msg=f"updated weights for {key} are incorrect")


    def test_different_dimensions_weights(self):
        self.instance.model.state_dict.return_value = self.model_weights
        
        # Incorrect dimensional weights.
        different_dimension_weights = {'fc1.weight': torch.randn(3, 4), 'fc1.bias': torch.randn(4)}
        self.instance.dendrite.query.return_value = different_dimension_weights

        # Run the function
        with pytest.raises(Exception) as excinfo:
            _merge_weights( self.instance, None )
        assert "There are no valid weights dicts." in str(excinfo.value)

    def test_invalid_weights_returned(self):
        # Build invalid weights returned
        self.instance.model.state_dict.return_value = self.model_weights
        self.instance.dendrite.query.return_value = None  # or 'invalid', or any other invalid data type
       
        # Run the function
        with pytest.raises(Exception) as excinfo:
            _merge_weights( self.instance, None )
        assert "There are no valid weights dicts." in str(excinfo.value)

    def test_empty_model_state_dict(self):
        # The model's state_dict is empty
        self.instance.model.state_dict.return_value = {}  
        self.instance.dendrite.query.return_value = [self.model_weights]

        # Run the function
        with pytest.raises(Exception) as excinfo:
            _merge_weights( self.instance, None )
        assert "There are no valid weights dicts." in str(excinfo.value)

    def test_return_is_invalid_none(self):
        # Set up the mock to return a state_dict that does not include the weights for 'fc1'
        self.instance.dendrite.query.return_value = {'fc1.weight': torch.randn(3, 3), 'fc2.weight': None}

        # No valid grad dicts.
        with pytest.raises(Exception) as excinfo:
            _merge_weights( self.instance, None )
        assert "There are no valid weights dicts." in str(excinfo.value)

    def test_return_is_invalid_dtype(self):
        # Set up the mock to return a state_dict that does not include the weights for 'fc1'
        self.instance.dendrite.query.return_value = {'fc1.weight': torch.randint(0, 10, (3, 3), dtype=torch.int64), 'fc2.weightkansdsa':  torch.randint(0, 10, (3, 3), dtype=torch.int64)}

        # No valid grad dicts.
        with pytest.raises(Exception) as excinfo:
            _merge_weights( self.instance, None )
        assert "There are no valid weights dicts." in str(excinfo.value)


if __name__ == '__main__':
    unittest.main()
