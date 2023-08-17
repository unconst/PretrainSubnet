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
import signal
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
            self.process.send_signal(signal.SIGTERM)
            self.process.wait()
            self.process = None

    def restart(self):
        """Restart an existing process"""
        self.stop()
        self.start()

    def is_alive(self):
        """Check if the process is still running"""
        return self.process is not None and self.process.poll() is None

def get_current_spec_version() -> int:
    import importlib
    import src
    importlib.reload(src)
    return src.__spec_version__

def main():
    """
    Main function to start the process and continuously check for git changes.
    If changes are detected, pull them and restart the process.
    """
    
    # Get run state.
    wandb_run_id = wandb.util.generate_id()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).strip().decode('utf-8')

    # Get spec version
    running_spec_version = get_current_spec_version()
    bt.logging.success( f'Current spec version: {running_spec_version}')

    # Start process.
    bt.logging.success( f'Starting: {sys.executable} src/train.py --wandb_run_id {wandb_run_id} { sys.argv[1:] }' )
    p = Process([sys.executable, 'src/train2.py', '--wandb_run_id', wandb_run_id ] + sys.argv[1:] , stdout=sys.stdout, stderr=sys.stderr)
    p.start()        
    
    # Endless loop until killed.
    while True:

        # Catch errors in the below script.
        try:

            # Check if the process is still running
            if not p.is_alive():
                bt.logging.success('Process terminated. Restarting...')
                p.restart()


            # Check if the branch hash has changed, if it has, pull install and restart.
            bt.logging.success('Changes detected. Pulling updates and restarting...')

            # Stash the local changes.
            stash_result = subprocess.run(['git', 'stash'], cwd = script_dir, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            if stash_result.returncode != 0:
                bt.logging.error(stash_result.stderr.decode())
            else:
                bt.logging.success( f'Called stash with output: { stash_result.stdout } ')

            # Fetch the latest changes.
            fetch_result = subprocess.run(['git', 'fetch'], cwd = script_dir, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            if fetch_result.returncode != 0:
                bt.logging.error(fetch_result.stderr.decode())
            else:
                bt.logging.success( f'Called fetch with output: { fetch_result.stdout}' )

            # Pull the latest changes.
            pull_result = subprocess.run(['git', 'pull'], cwd = script_dir, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            if pull_result.returncode != 0:
                bt.logging.error(pull_result.stderr.decode())
            else:
                bt.logging.success( f'Called pull with output: {pull_result.stdout}' )

            # Check if there are git changes on this local branch.
            latest_spec_version = get_current_spec_version()
            bt.logging.success( f'Current spec version: {latest_spec_version}')

            if latest_spec_version != running_spec_version:
                # Change current.
                bt.logging.success( f'Changes detected: {latest_spec_version} != {running_spec_version}')
                running_spec_version = latest_spec_version

                # Insall the local changes.
                install_result = subprocess.run([ sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                if install_result.returncode != 0:
                    bt.logging.error(install_result.stderr.decode())
                else:
                    bt.logging.success( f'Called install with output: {install_result.stdout }' )

                    # Restart the script.
                    bt.logging.success( f'Restarted the script.' )
                    p.restart()
            
            # All good, continue.
            else:
                bt.logging.success('No changes detected. Continuing.')

            # Wait.
            time.sleep(random.randint(60, 180))  

        # Catch user stop.      
        except KeyboardInterrupt:
            # Log and stop both processes.
            bt.logging.success('Interrupted by user. Exiting.')
            p.stop()
            break

        # Catch other errors.
        except Exception as e:
            bt.logging.error(f'Saw uncaught error {e}')
            continue

if __name__ == '__main__':
    main()