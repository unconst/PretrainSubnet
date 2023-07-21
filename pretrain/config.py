import argparse
import bittensor as bt

def init_config( cls ) -> bt.config:
    """
    Configuration method for DMiner class.

    Returns:
    Bittensor configuration object with default values set.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument( '--netuid', type = int, default = 97)
    parser.add_argument( '--chain_endpoint', type = str, default = "wss://test.finney.opentensor.ai")
    bt.wallet.add_args( parser )
    bt.axon.add_args( parser )
    bt.logging.add_args( parser )
    config = bt.config( parser )
    return config
