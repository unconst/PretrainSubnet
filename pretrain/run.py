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
import sys
import pprint
import traceback
import bittensor as bt
from .misc import get_online_uids
from .get_grads import get_grads, merge_grads
from .get_weights import get_weights, merge_weights

def init_run_state( self ):
    bt.logging.info(f'Init run state: Start')

    self.subtensor.register(wallet = self.wallet, netuid = self.config.netuid )

    # Fetch the current network state (metagraph) from Subtensor.
    self.metagraph = self.subtensor.metagraph(self.config.netuid)
    bt.logging.info( 'Synced Metagraph.')

    # Fetch my uid.
    self.my_uid = self.metagraph.hotkeys.index( self.wallet.hotkey.ss58_address )
    bt.logging.info( f'Got uid {self.my_uid} ')

    # Build, attach, serve and start Axon for communication with other miners.
    self.axon.serve( 
        netuid = self.config.netuid, 
        subtensor = self.subtensor 
    ).start()
    bt.logging.info( 'Started Axon.')

    # Set weights if we are required.
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

    # Re-fetch the current network state (metagraph) from Subtensor.
    self.metagraph = self.subtensor.metagraph(self.config.netuid)
    bt.logging.info( 'Re-synced Metagraph.')
    bt.logging.info(f'Currently: {get_online_uids(self)} online uids')

    # Set the model to training mode.
    self.model.train()

    # Merge weights to get the current weight distribution from others
    merge_weights( self )

    bt.logging.info(f'Init run state: Done') 

def run( self ):
    """
    Main training loop.
    """

    # Set up axon, set weights, sync graph.
    init_run_state( self )

    # Current block counter.
    self.current_block = self.subtensor.block 
    
    # List of data indices accumulated so far (via gradients merging and local training)
    self.global_accumulated_ids = []

    # List of axons hotkeys we have merged with this around already
    self.hotkeys_seen_this_round = set()

    # Counter for samples accumulated from remote host.
    self.remote_samples_accumulated = 0

    # Counter for samples accumulated during this step.
    self.local_samples_accumulated = 0

    # Counter for total samples applied across all steps.
    self.total_samples_applied = 0

    # Counter for the number of times we have merged gradients.
    self.total_gradient_merges = 0 

    # Counter for the number of times we have merged weights.
    self.total_weight_merges = 0 

    # Counter for the number of times we have applied gradients.
    self.total_applied_grads = 0 

    # Counter for the number of times we have synced the metagraph.
    self.total_graph_synced = 0 

    # Counter for the number of times we have set weights on chain.
    self.total_weights_set = 0 

    # Counter number of times get_grads was called on us.
    self.total_grads_shared = 0

    # Counter number samples shared with other miners.
    self.total_samples_shared = 0

    # Loop through epoch.
    for global_step, batch in enumerate( self.dataset.dataloader ):
        try:

            # Build batch.
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Forward pass
            outputs = self.model( input_ids = batch['input_ids'], attention_mask = batch['attention_mask'], labels = batch['labels'])
            loss = outputs.loss

            # Backward pass
            loss.backward()

            # Increment counters.
            self.current_block = self.subtensor.block 
            self.local_samples_accumulated += self.config.bs

            # Extend the list of accumualted samples
            self.global_accumulated_ids.extend( batch['id'].tolist() )

            # Log counters.
            log_event = {
                'global_step': global_step,
                'loss': loss.item(),
                'block': self.current_block,
                'total_samples_applied': self.total_samples_applied,
                'local_samples_accumulated': self.local_samples_accumulated,
                'remote_samples_accumulated': self.remote_samples_accumulated,
                'global_accumulated_ids': len( self.global_accumulated_ids ),
                'total_gradient_merges': self.total_gradient_merges,
                'total_weight_merges': self.total_weight_merges,
                'total_applied_grads': self.total_applied_grads,
                'total_graph_synced': self.total_graph_synced,
                'total_weights_set': self.total_weights_set,
                'total_grads_shared': self.total_grads_shared,
                'total_samples_shared': self.total_samples_shared,
            }
            self.wandb.log( log_event )
            bt.logging.info( "\n" + pprint.pformat(log_event) ) 

            # Merge gradients every steps_till_gradient_merge steps.
            if (global_step + 1) % self.config.steps_till_gradient_merge == 0:
                # Picks up to K miners and merges gradients with them.
                bt.logging.debug(f'Merging gradients.')
                merge_grads( self )
                self.total_gradient_merges += 1

            # Merge weights every steps_till_weights_merge steps.
            if (global_step + 1) % self.config.steps_till_weights_merge == 0:
                # Picks up to K miners and merges weights with them.
                bt.logging.debug(f'Merging weights.')
                merge_weights( self )
                self.total_weight_merges += 1

            # If we reached our accumulation level, apply the gradients.
            if (global_step + 1) % self.config.steps_till_gradient_apply == 0:
                # Apply accumulated gradients to the model state.
                bt.logging.debug(f'Applying gradients.')
                for param in self.model.parameters():
                    param.grad /= len( self.global_accumulated_ids )
                self.optimizer.step()
                assert len( self.global_accumulated_ids ) == self.remote_samples_accumulated + self.local_samples_accumulated
                self.total_samples_applied += len( self.global_accumulated_ids ) # increment all applied samples.
                self.total_applied_grads += 1 # Increment total applied grads.
                self.remote_samples_accumulated = 0 # Zero out remote accumulated samples
                self.local_samples_accumulated = 0 # Zero out local accumulated samples
                self.global_accumulated_ids = [] # Zero out accumualted samples.
                self.hotkeys_seen_this_round = set() # Zero out peers we've seen this round.

            # Sync the graph every blocks_till_resync blocks.
            if self.current_block % self.config.blocks_till_resync == 0: 
                # Fetch the current network state (metagraph) from Subtensor.
                bt.logging.debug(f'Syncing metagraph.')
                self.metagraph = self.subtensor.metagraph( self.config.netuid )
                self.total_graph_synced += 1
                bt.logging.info(f'Currently: {get_online_uids(self)} online uids')

            # Set weights every blocks_till_set_weights blocks
            if self.current_block % self.config.blocks_till_set_weights == 0:
                # Set weights on chain for ping.
                bt.logging.debug(f'Setting weights.')
                self.subtensor.set_weights( 
                    netuid = self.config.netuid, 
                    wallet = self.wallet, 
                    uids = [self.my_uid], 
                    weights = [1.0],
                    wait_for_inclusion = False,
                    wait_for_finalization = False,
                )
                self.total_weights_set += 1

        except KeyboardInterrupt:
            self.wandb.finish()

        except Exception as e:
            bt.logging.error( traceback.format_exc() )





