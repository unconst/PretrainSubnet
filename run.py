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
    

def get_version() -> str:
    """Function to get the version number from pretrain/__init__.py file."""
    with open('pretrain/__init__.py') as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip("'")

def git_has_changes() -> bool:
    """
    Function to check if there are any changes on the current git branch.
    If changes are detected, it also re-installs the package.
    """
    current_branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).strip().decode('utf-8')
    bt.logging.success(f'Checking git changes on {current_branch}' )

    result = subprocess.run(['git', 'fetch'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError('Error fetching git updates')

    # Get local version
    local_version = get_version()

    # Get remote version by temporarily checking out remote branch
    subprocess.run(['git', 'checkout', f'origin/{current_branch}'])
    remote_version = get_version()

    # Check back to local branch
    subprocess.run(['git', 'checkout', current_branch])

    # If version has increased, re-install the package
    if remote_version != local_version:
        bt.logging.success(f'Reinstalling pretrain package with updates' )
        subprocess.run(['pip', 'install', '.'], check=True)

    return remote_version != local_version

def git_pull():
    """
    Function to pull the latest changes from the current git branch.
    """
    bt.logging.success('Pulling git changes')
    result = subprocess.run(['git', 'pull'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError('Error pulling git updates')

def main():
    """
    Main function to start the process and continuously check for git changes.
    If changes are detected, pull them and restart the process.
    """
    bt.logging.success( f'Starting: python pretrain/neuron.py { sys.argv[1:] }' )
    p = Process(['python', 'pretrain/neuron.py'] + sys.argv[1:] , stdout=sys.stdout, stderr=sys.stderr)
    p.start()
    
    try:
        while True:

            # Check if the process is still running
            if not p.is_alive():
                bt.logging.success('Process terminated. Restarting...')
                p.restart()

            # Check if there are git changes on this local branch.
            if git_has_changes():
                bt.logging.success('Changes detected. Pulling updates and restarting...')
                git_pull()
                p.restart()
            
            # All good, continue.
            else:
                bt.logging.success('No changes detected. Continuing.')

            # Wait 3 minutes.
            time.sleep(180)

    except KeyboardInterrupt:
        # Log and stop both processes.
        bt.logging.success('Interrupted by user. Exiting.')
        p.stop()

if __name__ == '__main__':
    main()

