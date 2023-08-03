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

import torch
import typing
import random
import bittensor as bt

class GetParams(bt.Synapse):
    """
    This class inherits from bittensor.Synapse. It contains a single attribute `params`,
    which is expected to hold the state_dict of a PyTorch model.
    """
    params: typing.Optional[typing.Dict[str, bt.Tensor]] = None

    def serialize(self, model: torch.nn.Module):
        """
        This function serializes the state_dict of a PyTorch model into a dictionary of tensors.

        Parameters:
            model (torch.nn.Module): The PyTorch model whose state_dict is to be serialized.

        Returns:
            None
        """
        self.params = {k: bt.tensor(v) for k, v in model.state_dict().items()}

    def deserialize(self) -> typing.Dict[str, torch.Tensor]:
        """
        This function deserializes the `params` attribute into a state_dict for a PyTorch model.

        Returns:
            state_dict (typing.Dict[str, torch.Tensor]): The deserialized state dictionary.
        """
        if self.params is None:
            return {}
        else:
            try:
                # Convert all tensors to torch tensors.
                state_dict = {k: v.tensor() for k, v in self.params.items()}
                return state_dict
            except:
                bt.logging.warning("Failed to deserialize state_dict.")
                return {}

def reduce(model, dendrite, metagraph, replace:bool = False) -> bool:
    """
    This function averages model parameters with parameters of a randomly selected online Axon in the network.

    Parameters:
        model (nn.Module): The model to update.
        dendrite (Dendrite): Dendrite instance to query Axons.
        metagraph (Metagraph): Metagraph instance containing Axon network information.
        replace (bool): Whether to replace the model's parameters with the averaged parameters. If False, the averaged parameters are stored in the model's buffer.

    Returns:
        None
    """
    # Query all axons in the network.
    pings = dendrite.query( metagraph.axons, timeout = 2 )

    # Filter out online axons.
    online = [axon for ping, axon in zip(pings, metagraph.axons) if ping.is_success and ping.axon.hotkey != ping.dendrite.hotkey ]
    if not online:
        bt.logging.warning("No online uids to all reduce with.")
        return False
    else:
        bt.logging.debug(f"Found {len(online)} online uids to all reduce with.")

    # Randomly select an axon to query.
    to_query = random.choice(online)

    # Reduce with the selected axon.
    return reduce_with_axon(model, dendrite, to_query, replace=replace)

def reduce_with_axon(model, dendrite, axon, replace:bool = False) -> bool:
    """
    Function to fetch the parameters of a selected axon, validate them,
    and if valid, average the model's parameters with the fetched parameters.
    This function is useful in a distributed machine learning context where 
    multiple models are being trained in parallel, and their parameters are 
    periodically synchronized or "reduced" to ensure they all learn consistently.

    Parameters:
        model (torch.nn.Module): The PyTorch model whose parameters are to be updated.
        dendrite (bittensor.Dendrite): The dendrite through which the model connects to the network.
        axon (bittensor.Axon): The selected axon whose parameters are to be fetched.
        replace (bool): Whether to replace the model's parameters with the averaged parameters. If False, the averaged parameters are stored in the model's buffer.

    Returns:
        bool: Whether the parameter averaging was successful.
    """
    bt.logging.debug(f"Reducing with axon: {axon}.")

    # Fetch the parameters of the selected axon.
    state_dict = dendrite.query(axon, GetParams(), timeout = 24 )

    # Validate the received state_dict. If it's not valid, log a warning and return without updating parameters.
    if not is_valid_state_dict(model, state_dict):
        bt.logging.warning("Invalid state dict received from axon.")
        return
    else:
        bt.logging.debug("Valid state dict received from axon.")

    # If the state_dict is valid, average the model's parameters with the received parameters.
    for name, param in model.state_dict().items():
        if name in state_dict:
            if replace:
                param.data = state_dict[name].to(model.device).data
            else:
                element = state_dict[name].to(model.device)
                param.data = (param.data + element.data) / 2

    # Log that the parameter averaging is complete.
    bt.logging.info("All reduce successful.")
    return True
    

def is_valid_state_dict(model, state_dict) -> bool:
    """
    This function checks whether a given state_dict is valid for a PyTorch model.
    The state_dict is valid if:
        - it is a dictionary and not empty
        - all keys in the state_dict match the keys in the model's state_dict
        - all values in the state_dict are float tensors
        - all elements of the tensors are finite
        - the shapes of tensors in the state_dict match with corresponding tensors in the model's state_dict

    Parameters:
        model (nn.Module): The model to validate the state_dict for.
        state_dict (dict): The state dictionary to validate.

    Returns:
        bool: True if the state_dict is valid, False otherwise.
    """
    # TODO: Consider adding checks for non-finite values in the model's parameters as well.
    
    # Check if the state_dict is a non-empty dictionary.
    if not state_dict or not isinstance(state_dict, dict):
        bt.logging.warning('Invalid state_dict: It is either None, empty or not a dictionary.')
        return False

    for key in state_dict.keys():

        # Check if the key is present in the model's state_dict.
        if key not in model.state_dict().keys():
            bt.logging.warning(f'Invalid state_dict: Contains extra key not found in the model: {key}')
            return False

        # Check if the corresponding tensor is not None.
        if state_dict[key] is None:
            bt.logging.warning(f'Invalid state_dict: Contains None for key: {key}')
            return False

        # Check if the corresponding tensor is a float tensor.
        if not isinstance(state_dict[key], (torch.FloatTensor, torch.cuda.FloatTensor)):
            bt.logging.warning(f'Invalid state_dict: Tensor for key {key} is not a float tensor.')
            return False

        # Check if all elements in the tensor are finite.
        if not torch.all(torch.isfinite(state_dict[key])):
            bt.logging.warning(f'Invalid state_dict: Tensor for key {key} has non-finite values.')
            return False

        # Check if the tensor is on the correct device.
        state_dict[key] = state_dict[key]

        # Check if the shape of the tensor matches the corresponding tensor in the model.
        if state_dict[key].shape != model.state_dict()[key].shape:
            bt.logging.warning(f'Invalid state_dict: Shape mismatch for key {key}.')
            return False

    return True


import unittest
import torch.nn as nn
import torch

class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 1)
            
class TestBittensorFunctions(unittest.TestCase):

    def test_is_valid_state_dict_valid(self):
        """
        Test is_valid_state_dict function with a valid state dict.
        """
        model = TestModel()
        state_dict_valid = model.state_dict()
        self.assertTrue(is_valid_state_dict(model, state_dict_valid))

    def test_is_valid_state_dict_invalid(self):
        """
        Test is_valid_state_dict function with various invalid state dicts.
        """
        model = TestModel()

        # Test with empty state_dict
        state_dict_empty = {}
        self.assertFalse(is_valid_state_dict(model, state_dict_empty))

        # Test with extra keys
        state_dict_extra_keys = {"extra_key": torch.randn(10, 1)}
        self.assertFalse(is_valid_state_dict(model, state_dict_extra_keys))

        # Test with missing keys
        state_dict_missing_keys = {"weight": torch.randn(10, 1)}
        self.assertFalse(is_valid_state_dict(model, state_dict_missing_keys))

        # Test with wrong types
        state_dict_wrong_type = {"weight": "not a tensor", "bias": "not a tensor"}
        self.assertFalse(is_valid_state_dict(model, state_dict_wrong_type))

        # Test with non-finite values
        state_dict_non_finite = {"weight": torch.tensor([float('inf')]), "bias": torch.tensor([float('nan')])}
        self.assertFalse(is_valid_state_dict(model, state_dict_non_finite))

        # Test with wrong shapes
        state_dict_wrong_shapes = {"weight": torch.randn(10, 2), "bias": torch.randn(10)}
        self.assertFalse(is_valid_state_dict(model, state_dict_wrong_shapes))

if __name__ == "__main__":
    unittest.main()
