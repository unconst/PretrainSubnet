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
import bittensor as bt
from .misc import get_online_uids
from .get_grads import get_grads, merge_grads
from .get_weights import get_weights, merge_weights

def init_run_state( self ):
    bt.logging.info(f'Init run state: Start')

    self.subtensor.register(wallet = self.wallet, netuid = self.config.netuid )

    # Build, attach, serve and start Axon for communication with other miners.
    self.axon.serve( 
        netuid = self.config.netuid, 
        subtensor = self.subtensor 
    ).start()
    bt.logging.info( 'Started Axon.')

    # Fetch the current network state (metagraph) from Subtensor.
    self.metagraph = self.subtensor.metagraph(self.config.netuid)
    bt.logging.info( 'Synced Metagraph.')

    # Fetch my uid.
    self.my_uid = self.metagraph.hotkeys.index( self.wallet.hotkey.ss58_address )
    bt.logging.info( f'Got uid {self.my_uid} ')

    # Set ping weights.
    if self.subtensor.block - self.metagraph.last_update[ self.my_uid ] > 50:
        self.subtensor.set_weights( 
            netuid = self.config.netuid, 
            wallet = self.wallet, 
            uids = [self.my_uid], 
            weights = [1.0],
            wait_for_inclusion = False,
            wait_for_finalization = False,
        )
        bt.logging.info( 'Set weights ')
        self.wandb.log({ 'set_weights': 1.0 })

    bt.logging.info(f'Init run state: Done')


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
    init_run_state( self )

    # Initialize step counter
    total_epoch_steps = 0
    total_training_steps = 0
    total_accumulation_steps = 0

    # Training loop
    while True:
        bt.logging.info(f'Starting new epoch: {global_step}')

        # Fetch the current network state (metagraph) from Subtensor.
        self.metagraph = self.subtensor.metagraph( self.config.netuid )
        bt.logging.info(f'Synced Metagraph: {self.metagraph}')

        online_uids = get_online_uids( self )
        bt.logging.info( f'Online UIDS {online_uids}')
        self.wandb.log({ 'n_online': len( online_uids ) })

        # Set the model to training mode.
        self.model.train()

        # Clear out all existing gradients in the model.
        self.optimizer.zero_grad()

        # Merge weights
        merge_weights( self )

        # Train on epoch.
        for acc_step, batch in enumerate( self.dataloader ):
            bt.logging.success(f'Step: {acc_step}/{self.config.n_accumulation_steps}, Accumulations: {total_accumulation_steps}, Training: {total_training_steps}, Epoch: {total_epoch_steps}')

            # Zero out gradients calculated in the previous iteration.
            # and save them for others to query.
            self.saved_grads = { name: bt.tensor(parameter.grad.clone()) for name, parameter in self.model.named_parameters() if parameter.grad is not None }
            self.optimizer.zero_grad()

            # Move the batch tensors to the same device as the model
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Forward pass
            outputs = self.model(**batch)
            loss = outputs.loss

            # Backward pass
            loss.backward()

            # Inc total_accumulation steps.
            total_accumulation_steps += 1
            if ( acc_step + 1 ) % self.config.n_accumulation_steps == 0:
                bt.logging.success(f'Finished accumulating: Running training step.')

                # Average local gradients with remote.
                merge_grads( self )

                # Update the weights
                self.optimizer.step()

                # Log the loss value for this batch.
                total_training_steps += 1

            # Set weights if required.
            if self.subtensor.block - self.metagraph.last_update[ self.my_uid ] > 75:
                self.subtensor.set_weights( 
                    netuid = self.config.netuid, 
                    wallet = self.wallet, 
                    uids = [self.my_uid], 
                    weights = [1.0],
                    wait_for_inclusion = False,
                    wait_for_finalization = False,
                )
                bt.logging.info( 'Set weights ')
                self.wandb.log({ 'set_weights': 1.0 })
            
            # Log all params. 
            torch.cuda.empty_cache() # Clear cache if existent.
            self.wandb.log({ 'block': self.subtensor.block })
            self.wandb.log({ 'total_training_steps': total_training_steps })
            self.wandb.log({ 'total_accumulation_steps': total_accumulation_steps })
            self.wandb.log({ 'train_loss': loss })
            bt.logging.info(f"Loss: {loss.item()}") 

        # Log finished epoch
        total_epoch_steps += 1
        self.wandb.log({ 'total_epoch_steps': total_epoch_steps })
        bt.logging.success(f'Finished epoch: { total_epoch_steps }')


