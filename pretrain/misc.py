import wandb
import typing
import bittensor as bt

def get_online_uids( self ) -> typing.List[int]:
    current_block = self.subtensor.block
    return [ uid for uid, update in enumerate( self.metagraph.last_update ) if current_block - update < 100 ]

def init_wandb( self ):
    # Init wandb
    tags = [ self.wallet.hotkey.ss58_address ]
    self.wandb = wandb.init(
        anonymous="allow",
        reinit = True,
        project = "open-pretrain",
        mode = "offline",
        entity = "opentensor-dev",
        config = self.config,
        tags = [self.wallet.hotkey.ss58_address],
    )
    bt.logging.success(
        prefix="Started a new wandb run",
        sufix=f"<blue> {self.wandb.name} </blue>",
    )
