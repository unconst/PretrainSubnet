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
import sys
import time
import wandb
import random
import subprocess
import bittensor as bt
from signal import SIGTERM

class Process:
    """Process class to manage external processes"""
    def __init__(self, *args, **kwargs):
        """Initialize the process with arguments"""
        self.args = args
        self.kwargs = kwargs
        self.process = None

    def start(self):
        """Start a new process"""
        self.process = subprocess.Popen(*self.args, **self.kwargs)

    def stop(self):
        """Stop an existing process"""
        if self.process:
            self.process.send_signal(SIGTERM)
            self.process.wait()
            self.process = None

    def restart(self):
        """Restart an existing process"""
        self.stop()
        self.start()

    def is_alive(self):
        """Check if the process is still running"""
        return self.process is not None and self.process.poll() is None

def main():
    """
    Main function to start the process and continuously check for git changes.
    If changes are detected, pull them and restart the process.
    """
    
    # Get run state.
    wandb_run_id = wandb.util.generate_id()
    branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).strip().decode('utf-8')
    running_git_hash = subprocess.check_output(['git', 'rev-parse', f'origin/{branch}']).strip()

    # Start process.
    bt.logging.success( f'Starting: {sys.executable} src/train.py --wandb_run_id {wandb_run_id} { sys.argv[1:] }' )
    p = Process([sys.executable, 'src/train.py', '--wandb_run_id', wandb_run_id ] + sys.argv[1:] , stdout=sys.stdout, stderr=sys.stderr)
    p.start()
    
    try:
        while True:

            # Check if the process is still running
            if not p.is_alive():
                bt.logging.success('Process terminated. Restarting...')
                p.restart()

            # Check if there are git changes on this local branch.
            lastest_git_hash = subprocess.check_output(['git', 'rev-parse', f'origin/{branch}']).strip()

            # Check if the branch hash has changed, if it has, pull install and restart.
            if lastest_git_hash != running_git_hash:
                bt.logging.success('Changes detected. Pulling updates and restarting...')
                subprocess.run(['git', 'fetch', f'origin/{branch}'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                subprocess.run(['git', 'pull', f'origin/{branch}'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                subprocess.run([ sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], check=True)
                p.restart()
                running_git_hash = lastest_git_hash
            
            # All good, continue.
            else:
                bt.logging.success('No changes detected. Continuing.')

            # Wait (3 minutes -≥ 9 minutes)
            time.sleep(random.randint(180, 181))

    except KeyboardInterrupt:
        # Log and stop both processes.
        bt.logging.success('Interrupted by user. Exiting.')
        p.stop()

if __name__ == '__main__':
    main()