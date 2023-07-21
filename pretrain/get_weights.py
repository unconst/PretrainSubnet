
import gc
import torch
import typing
import bittensor as bt

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
        _merge_weights( self )
    except Exception as e:
        bt.logging.error( f'Failed to merge weights with error {e}')

def _merge_weights(self):
    """
    This method retrieves the model weights from all miners in the network and then averages 
    these weights. The averaged weights are then used to update the local model.
    """

    # Log the start of the weight averaging process.
    bt.logging.info('Starting the weight averaging process.')
    self.wandb.log({ 'average_weights_event': 1.0 })

    # Retrieve the online axons (miners) in the metagraph (network).
    online_axons = [self.metagraph.axons[uid] for uid in self.get_online_uids()]
    
    # If there are no online axons, there's nothing to average, so we return immediately.
    if len(online_axons) == 0: 
        raise Exception('There are no online uids to average weights with.')
    bt.logging.info(f'Averaging weights with uids: {self.get_online_uids()}')

    # Query all miners for their model weights.
    state_dicts = self.dendrite.query(online_axons, GetWeights())
    
    # If only one model's weights were returned, make sure it's in a list.
    if not isinstance(state_dicts, list): 
        state_dicts = [state_dicts]

    # Filter out invalid state_dicts.
    valid_state_dicts = []
    for state_dict in state_dicts:
        # TODO: Add error handling for when the state_dict is None, not a dictionary, 
        # the keys don't match the model's keys, or the weights dimensions don't match.
        if state_dict is None or not isinstance(state_dict, dict):
            bt.logging.warning('Invalid state_dict: Not a dictionary or None.')
            continue

        if set(state_dict.keys()) != set(self.model.state_dict().keys()):
            bt.logging.warning('Invalid state_dict: Keys do not match the model.')
            continue

        if not all( state_dict[key] != None for key in state_dict.keys()):
            bt.logging.warning('Invalid state_dict: grad is none')
            continue

        if not all( isinstance(state_dict[key], (torch.FloatTensor, torch.cuda.FloatTensor)) for key in state_dict.keys() ):
            bt.logging.warning('Invalid state_dict: grads are not float tensor.')
            continue

        if not all( torch.all( torch.isfinite(state_dict[key]) ) for key in state_dict.keys()):
            bt.logging.warning('Invalid state_dict: Grads are not finite')

        if not all(torch.tensor(state_dict[key]).shape == torch.tensor(self.model.state_dict()[key]).shape for key in state_dict.keys()):
            bt.logging.warning('Invalid state_dict: Weights dimensions do not match the model.')
            continue

        valid_state_dicts.append(state_dict)
    if len(valid_state_dicts) == 0:
        raise Exception('There are no valid weights dicts.')
    self.wandb.log( {'valid_weight_dicts': len(valid_state_dicts)} )

    # Create a new state dictionary for the averaged weights.
    avg_state_dict = {}

    for key in self.model.state_dict().keys():
        # Initialize accumulation variables.
        total_weights = torch.zeros_like(self.model.state_dict()[key]).to(self.device)
        num_weights = 0

        # Accumulate weights and their count.
        for state_dict in valid_state_dicts:
            if key in state_dict:
                total_weights += state_dict[key]
                num_weights += 1

        # Skip this key if there are no weights available.
        if num_weights == 0: 
            continue

        # Compute the average of the weights.
        avg_weight = total_weights / num_weights

        # Assign the average weight to the new state dictionary.
        avg_state_dict[key] = avg_weight

        del total_weights
        del num_weights

    # If there are any averaged weights, load them into the local model.
    if avg_state_dict:
        self.model.load_state_dict(avg_state_dict)
    
    bt.logging.success(f'Successfully averaged {len(state_dicts)} weights.') 
    self.wandb.log({ 'successfully_average_weights': 1.0 })

    # Sanity check delete memory
    del state_dicts
    del valid_state_dicts
    del avg_state_dict
    gc.collect()

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
        self.instance.get_online_uids = MagicMock()
        self.instance.model = MagicMock()
        self.instance.wandb = MagicMock()
        self.instance.device = 'cpu'

        # Set up some fake model weights
        self.model_weights = {'fc1.weight': torch.randn(3, 3), 'fc1.bias': torch.randn(3)}

    def test_no_online_axons(self):
        # Set up the mock to return an empty list of online uids
        self.instance.get_online_uids.return_value = []

        # Run the function
        with pytest.raises(Exception) as excinfo:
            _merge_weights( self.instance )
        assert "There are no online uids to average weights with." in str(excinfo.value)

    def test_no_weights_found(self):
        # Set up the mock to return a list of online uids
        self.instance.get_online_uids.return_value = ['uid1']

        # Set up the model to have some weights
        self.instance.model.state_dict.return_value = self.model_weights

        # Set up the mock to return a state_dict that does not include the weights for 'fc1'
        self.instance.dendrite.query.return_value = {'fc2.weight': torch.randn(3, 3), 'fc2.bias': torch.randn(3)}

        # Run the function
        with pytest.raises(Exception) as excinfo:
            _merge_weights( self.instance )
        assert "There are no valid weights dicts." in str(excinfo.value)

    def test_successful_average(self):
        # Set up the mock to return a list of online uids
        self.instance.get_online_uids.return_value = ['uid1']

        # Set up the model to have some weights
        self.instance.model.state_dict.return_value = self.model_weights

        # Set up the mock to return a state_dict that does include the weights for 'fc1'
        self.instance.dendrite.query.return_value = self.model_weights

        # Run the function
        _merge_weights( self.instance )

    def test_wrong_model_weights(self):
        # Set up the mock to return a list of online uids
        self.instance.get_online_uids.return_value = ['uid1']

        # Set up the model to have some weights
        self.instance.model.state_dict.return_value = self.model_weights

        # Set up the mock to return a state_dict with weights for a different model
        wrong_model_weights = {'conv1.weight': torch.randn(3, 3, 3), 'conv1.bias': torch.randn(3)}
        self.instance.dendrite.query.return_value = wrong_model_weights

        # Run the function
        with pytest.raises(Exception) as excinfo:
            _merge_weights( self.instance )
        assert "There are no valid weights dicts." in str(excinfo.value)

    def test_empty_model_weights(self):
        # Set up the mock to return a list of online uids
        self.instance.get_online_uids.return_value = ['uid1']

        # Set up the model to have some weights
        self.instance.model.state_dict.return_value = self.model_weights

        # Set up the mock to return a state_dict with weights for a different model
        empty_model_weights = {}
        self.instance.dendrite.query.return_value = empty_model_weights

        # Run the function
        with pytest.raises(Exception) as excinfo:
            _merge_weights( self.instance )
        assert "There are no valid weights dicts." in str(excinfo.value)

    def test_multiple_weights(self):
        # Set up the mock to return a list of online uids
        self.instance.get_online_uids.return_value = ['uid1', 'uid2']

        # Set up the model to have some weights
        self.instance.model.state_dict.return_value = self.model_weights

        # Set up the mock to return a state_dict with weights for a different model
        multiple_responses = [ {'fc1.weight': torch.randn(3, 3), 'fc1.bias': torch.randn(3)}, {'fc1.weight': torch.randn(3, 3), 'fc1.bias': torch.randn(3)} ]
        self.instance.dendrite.query.return_value = multiple_responses

        # Run the function
        _merge_weights( self.instance )

    def test_multiple_weights_some_wrong(self):
        # Set up the mock to return a list of online uids
        self.instance.get_online_uids.return_value = ['uid1', 'uid2', 'uid3']

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
        _merge_weights( self.instance )

    def test_merged_weights_are_average(self):
        # Set up the mock to return a list of online uids
        self.instance.get_online_uids.return_value = ['uid1', 'uid2']

        # Set up the model to have some weights
        self.instance.model.state_dict.return_value = self.model_weights

        # Define some weights that will be returned by the dendrite
        model_weights1 = {'fc1.weight': torch.randn(3, 3), 'fc1.bias': torch.randn(3)}
        model_weights2 = {'fc1.weight': torch.randn(3, 3), 'fc1.bias': torch.randn(3)}
        dendrite_weights = [model_weights1, model_weights2]

        # Set up the mock to return these weights
        self.instance.dendrite.query.return_value = dendrite_weights

        # Run the function
        _merge_weights( self.instance )

        # Calculate the expected averaged weights
        expected_weights = {'fc1.weight': torch.stack([model_weights1['fc1.weight'], model_weights2['fc1.weight']]).mean(dim=0),
                            'fc1.bias': torch.stack([model_weights1['fc1.bias'], model_weights2['fc1.bias']]).mean(dim=0)}

        # Verify the model's weights were updated correctly
        updated_weights = self.instance.model.load_state_dict.call_args[0][0]  # Get the first argument passed to load_state_dict
        for key in expected_weights.keys():
            self.assertTrue(torch.allclose(updated_weights[key], expected_weights[key]), msg=f"updated weights for {key} are incorrect")


    def test_different_dimensions_weights(self):
        self.instance.get_online_uids.return_value = ['uid1', 'uid2']
        self.instance.model.state_dict.return_value = self.model_weights
        
        # Incorrect dimensional weights.
        different_dimension_weights = {'fc1.weight': torch.randn(3, 4), 'fc1.bias': torch.randn(4)}
        self.instance.dendrite.query.return_value = different_dimension_weights

        # Run the function
        with pytest.raises(Exception) as excinfo:
            _merge_weights( self.instance )
        assert "There are no valid weights dicts." in str(excinfo.value)

    def test_invalid_weights_returned(self):
        self.instance.get_online_uids.return_value = ['uid1']

        # Build invalid weights returned
        self.instance.model.state_dict.return_value = self.model_weights
        self.instance.dendrite.query.return_value = None  # or 'invalid', or any other invalid data type
       
        # Run the function
        with pytest.raises(Exception) as excinfo:
            _merge_weights( self.instance )
        assert "There are no valid weights dicts." in str(excinfo.value)

    def test_empty_model_state_dict(self):
        self.instance.get_online_uids.return_value = ['uid1']

        # The model's state_dict is empty
        self.instance.model.state_dict.return_value = {}  
        self.instance.dendrite.query.return_value = [self.model_weights]

        # Run the function
        with pytest.raises(Exception) as excinfo:
            _merge_weights( self.instance )
        assert "There are no valid weights dicts." in str(excinfo.value)

    def test_return_is_invalid_none(self):
        # Set up the mock to return a list of online uids
        self.instance.get_online_uids.return_value = ['uid1']

        # Set up the mock to return a state_dict that does not include the weights for 'fc1'
        self.instance.dendrite.query.return_value = {'fc1.weight': torch.randn(3, 3), 'fc2.weight': None}

        # No valid grad dicts.
        with pytest.raises(Exception) as excinfo:
            _merge_weights( self.instance )
        assert "There are no valid weights dicts." in str(excinfo.value)

    def test_return_is_invalid_dtype(self):
        # Set up the mock to return a list of online uids
        self.instance.get_online_uids.return_value = ['uid1']

        # Set up the mock to return a state_dict that does not include the weights for 'fc1'
        self.instance.dendrite.query.return_value = {'fc1.weight': torch.randint(0, 10, (3, 3), dtype=torch.int64), 'fc2.weightkansdsa':  torch.randint(0, 10, (3, 3), dtype=torch.int64)}

        # No valid grad dicts.
        with pytest.raises(Exception) as excinfo:
            _merge_weights( self.instance )
        assert "There are no valid weights dicts." in str(excinfo.value)


if __name__ == '__main__':
    unittest.main()
