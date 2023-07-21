import gc
import torch
import typing
import bittensor as bt
from .misc import get_online_uids

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
        merge_grads( self )
    except Exception as e:
        bt.logging.error( f'Failed to merge grads with error {e}')

def _merge_grads( self ):
    """
    Function to average the gradients of a model across several online axons.
    """

    # Log that the gradient reduction process is starting
    bt.logging.info('Reducing gradients.')
    self.wandb.log({'reduce_gradients_event': 1.0})

    online_axons = [self.metagraph.axons[uid] for uid in self.get_online_uids()]
    if len(online_axons) == 0:
        raise Exception('There are no online uids to average gradients with.')
    bt.logging.debug(f'Reducing grads with axons:\n {online_axons}')

    # Use the dendrite's query function to retrieve the gradients for the online axons
    grad_dicts = self.dendrite.query(online_axons, GetGrads())

    # If the query function only returns one dictionary, wrap it in a list for the later iteration
    if not isinstance(grad_dicts, list):
        grad_dicts = [grad_dicts]

    # Filter out invalid grads.
    valid_grad_dicts = []
    for grad_dict in grad_dicts:
        # TODO: Add error handling for when the grad_dict is None, not a dictionary, 
        # the keys don't match the model's keys, or the weights dimensions don't match.
        if grad_dict is None or not isinstance(grad_dict, dict):
            bt.logging.warning('Invalid grad_dict: Not a dictionary or None.')
            continue

        if set(grad_dict.keys()) != set(self.model.state_dict().keys()):
            bt.logging.warning('Invalid grad_dict: Keys do not match the model.')
            continue

        if not all( grad_dict[key] != None for key in grad_dict.keys()):
            bt.logging.warning('Invalid grad_dict: grad is none')
            continue

        if not all( isinstance(grad_dict[key], (torch.FloatTensor, torch.cuda.FloatTensor)) for key in grad_dict.keys() ):
            bt.logging.warning('Invalid grad_dict: grads are not float tensor.')
            continue

        if not all( torch.all( torch.isfinite(grad_dict[key]) ) for key in grad_dict.keys()):
            bt.logging.warning('Invalid grad_dict: Grads are not finite.')
            continue

        if not all(torch.tensor(grad_dict[key]).shape == torch.tensor(self.model.state_dict()[key]).shape for key in grad_dict.keys()):
            bt.logging.warning('Invalid grad_dict: Weights dimensions do not match the model.')
            continue

        valid_grad_dicts.append(grad_dict)

    # Check that there are valid gradient dicts to average.
    if len(valid_grad_dicts) == 0:
        raise Exception('There are no valid gradient dicts.')
    self.wandb.log( {'valid_grad_dicts': len(valid_grad_dicts)} )

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
            grad_sum += grad

        # Divide the sum by the number of gradients to get the average (note: this is an in-place operation)
        grad_sum.div_(len(all_grads))

        # Assign the averaged gradient to the averaged gradients dictionary
        avg_valid_grads_dict[key] = grad_sum

        # Delete all_grads and grad_sum to free up memory
        del all_grads
        del grad_sum

    # Apply the averaged gradients to the model's parameters
    for name, param in self.model.named_parameters():
        # Only apply the gradients if the parameter exists in the averaged gradients dictionary
        if name in avg_valid_grads_dict:
            # If the parameter already has a gradient, add the averaged gradient to it
            # Otherwise, assign the averaged gradient as the parameter's gradient
            if param.grad is not None:
                param.grad += avg_valid_grads_dict[name]
            else:
                param.grad = avg_valid_grads_dict[name].clone()

    # Log the successful reduction of gradients
    bt.logging.success(f'Successfully reduced {len(grad_dicts)} grads.')
    self.wandb.log({'successfully_average_gradients': 1.0})

    # Sanity check delete memory
    del grad_dicts
    del valid_grad_dicts
    del avg_valid_grads_dict
    gc.collect()


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
        self.instance.get_online_uids = MagicMock()
        self.instance.model = MagicMock()
        self.instance.wandb = MagicMock()
        self.instance.device = 'cpu'

        # Set up some fake model weights
        self.model_weights = { 'fc1.weight': torch.randn(3, 3), 'fc2.weight': torch.randn(3, 3) }
        self.instance.model.state_dict.return_value = self.model_weights
        self.instance.model.named_parameters.return_value = [ ('fc1.weight', MagicMock()), ('fc2.weight', MagicMock()) ]
        for name, param in self.instance.model.named_parameters.return_value:
            param.grad = torch.randn(3, 3)

    def test_no_online_axons(self):
        # Set up the mock to return an empty list of online uids
        self.instance.get_online_uids.return_value = []

        # None to average with
        with pytest.raises(Exception) as excinfo:
            _merge_grads( self.instance )
        assert "There are no online uids to average gradients with." in str(excinfo.value)
    
    def test_merge_valid_grad_dict(self):
        # Set up the mock to return a list of online uids
        self.instance.get_online_uids.return_value = ['uid1']

        # Set up the mock to return a state_dict that does not include the weights for 'fc1'
        self.instance.dendrite.query.return_value = {'fc1.weight': torch.randn(3, 3), 'fc2.weight': torch.randn(3, 3)}

        # Valid grad dict.
        _merge_grads( self.instance )

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
        # Set up the mock to return a list of online uids
        self.instance.get_online_uids.return_value = ['uid1']

        # Set up the mock to return a state_dict that does not include the weights for 'fc1'
        self.instance.dendrite.query.return_value = [
             {'fc1.weight': torch.randn(3, 3), 'fc2.weight': torch.randn(3, 3)},
             {'fc1.weight': torch.randn(3, 3), 'fc2.weight': torch.randn(3, 3)}
        ]

        # Valid grad dict.
        _merge_grads( self.instance )

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
        # Set up the mock to return a list of online uids
        self.instance.get_online_uids.return_value = ['uid1']

        # Set up the mock to return a state_dict that does not include the weights for 'fc1'
        self.instance.dendrite.query.return_value = [
            {'fc1.weight': torch.randn(3, 3), 'fc2.weight': torch.randn(3, 3)},
            {'fc1.weight': torch.randn(3, 3), 'fc2.weight': torch.randn(3, 3)},
            {'fc1.weight': torch.randn(3, 4), 'fc2.weight': torch.randn(3, 3)},
            {'fc1.weight': None, 'fc2.weight': torch.randn(3, 3)},
            None,
        ]

        # Valid grad dict.
        _merge_grads( self.instance )

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
        # Set up the mock to return a list of online uids
        self.instance.get_online_uids.return_value = ['uid1']

        # Set up the mock to return a state_dict that does not include the weights for 'fc1'
        self.instance.dendrite.query.return_value = None

        # No valid grad dicts.
        with pytest.raises(Exception) as excinfo:
            _merge_grads( self.instance )
        assert "There are no valid gradient dicts." in str(excinfo.value)

    def test_return_is_empty(self):
        # Set up the mock to return a list of online uids
        self.instance.get_online_uids.return_value = ['uid1']

        # Set up the mock to return a state_dict that does not include the weights for 'fc1'
        self.instance.dendrite.query.return_value = {}

        # No valid grad dicts.
        with pytest.raises(Exception) as excinfo:
            _merge_grads( self.instance )
        assert "There are no valid gradient dicts." in str(excinfo.value)

    def test_return_is_invalid_name(self):
        # Set up the mock to return a list of online uids
        self.instance.get_online_uids.return_value = ['uid1']

        # Set up the mock to return a state_dict that does not include the weights for 'fc1'
        self.instance.dendrite.query.return_value = {'fc1.weight': torch.randn(3, 3), 'fc2.weightkansdsa': torch.randn(3, 3)}

        # No valid grad dicts.
        with pytest.raises(Exception) as excinfo:
            _merge_grads( self.instance )
        assert "There are no valid gradient dicts." in str(excinfo.value)

    def test_return_is_invalid_none(self):
        # Set up the mock to return a list of online uids
        self.instance.get_online_uids.return_value = ['uid1']

        # Set up the mock to return a state_dict that does not include the weights for 'fc1'
        self.instance.dendrite.query.return_value = {'fc1.weight': torch.randn(3, 3), 'fc2.weightkansdsa': None}

        # No valid grad dicts.
        with pytest.raises(Exception) as excinfo:
            _merge_grads( self.instance )
        assert "There are no valid gradient dicts." in str(excinfo.value)

    def test_return_is_invalid_dtype(self):
        # Set up the mock to return a list of online uids
        self.instance.get_online_uids.return_value = ['uid1']

        # Set up the mock to return a state_dict that does not include the weights for 'fc1'
        self.instance.dendrite.query.return_value = {'fc1.weight': torch.randint(0, 10, (3, 3), dtype=torch.int64), 'fc2.weightkansdsa':  torch.randint(0, 10, (3, 3), dtype=torch.int64)}

        # No valid grad dicts.
        with pytest.raises(Exception) as excinfo:
            _merge_grads( self.instance )
        assert "There are no valid gradient dicts." in str(excinfo.value)

    def test_invalid_dimension(self):
         # Set up the mock to return a list of online uids
        self.instance.get_online_uids.return_value = ['uid1']

        # Set up the mock to return a state_dict that does not include the weights for 'fc1'
        self.instance.dendrite.query.return_value = {'fc1.weight': torch.randn(3, 4), 'fc2.weight': torch.randn(3, 3)}

        # No valid grad dicts.
        with pytest.raises(Exception) as excinfo:
            _merge_grads( self.instance )
        assert "There are no valid gradient dicts." in str(excinfo.value)


if __name__ == "__main__":
    unittest.main()