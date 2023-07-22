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

import os
import torch
import argparse
import bittensor as bt

def init_config( cls ) -> bt.config:
    """
    Configuration method for DMiner class.

    Returns:
    Bittensor configuration object with default values set.
    """
    parser = argparse.ArgumentParser()

    # Network connection parameters.
    parser.add_argument( 
        '--netuid', 
        type = int, 
        default = 97,
        help="The chain subnet uid.", 
    )
    parser.add_argument( 
        '--chain_endpoint', 
        type = str, 
        default = "wss://test.finney.opentensor.ai",
        help="The chain endpoint to connect with.", 
    )

    # Training parameters.
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Training learning rate.",
    )
    parser.add_argument(
        "--bs",
        type = int,
        default = 16,
        help = "Training batch size.",
    )
     parser.add_argument(
        "--n_accumulation_steps",
        type = int,
        default = 1,
        help = "Number of steps before we apply an accumulation step.",
    )

    # Neuron identification.
    parser.add_argument(
        "--name",
        type = str,
        default = "pretrain",
        help = "Trials for this neuron go in logging_dir / wallet_cold / wallet_hot / netuid / name. ",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the validator on.",
    )

    # Wandb params.
    parser.add_argument(
        "--wandb.off", 
        action="store_true", 
        default=False,
        help="Turn off wandb.", 
    )
    parser.add_argument(
        "--wandb.project_name",
        type=str,
        default="openpretrain",
        help="The name of the project where you are sending the new run.",
    )
    parser.add_argument(
        "--wandb.entity",
        type=str,
        default="opentensor-dev",
        help="An entity is a username or team name where youre sending runs.",
    )
    parser.add_argument(
        "--wandb.offline",
        action="store_true",
        default=False,
        help="Runs wandb in offline mode.",
    )
    parser.add_argument(
        "--wandb.notes",
        type=str,
        default="",
        help="Notes to add to the wandb run.",
    )
    bt.wallet.add_args( parser )
    bt.axon.add_args( parser )
    bt.logging.add_args( parser )
    config = bt.config( parser )

    # Create full path.
    full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,
            config.wallet.name,
            config.wallet.hotkey,
            config.netuid,
            config.name,
        )
    )
    config.full_path = os.path.expanduser( full_path )
    if not os.path.exists( config.full_path ):
        os.makedirs( config.full_path, exist_ok=True )

    return config
