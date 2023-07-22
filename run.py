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
from threading import Thread

class Process:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.process = None

    def start(self):
        self.process = subprocess.Popen(*self.args, **self.kwargs)

    def stop(self):
        if self.process:
            self.process.send_signal(SIGTERM)
            self.process.wait()
            self.process = None

    def restart(self):
        self.stop()
        self.start()

def git_has_changes():
    current_branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).strip().decode('utf-8')
    bt.logging.success(f'Checking git changes on {current_branch}' )
    result = subprocess.run(['git', 'fetch'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError('Error fetching git updates')
    result = subprocess.run(['git', 'diff', current_branch, 'origin/'+current_branch], stdout=subprocess.PIPE)
    return result.stdout != b''

def git_pull():
    bt.logging.success('Pulling git changes')
    result = subprocess.run(['git', 'pull'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError('Error pulling git updates')

def main():
    bt.logging.success( f'Starting: python pretrain/neuron.py { sys.argv[1:] }' )
    p = Process(['python', 'pretrain/neuron.py'] + sys.argv[1:] , stdout=sys.stdout, stderr=sys.stderr)
    p.start()
    try:
        while True:

            # Check if there are git changes on this local branch.
            if git_has_changes():
                bt.logging.success('Changes detected. Pulling updates and restarting...')
                git_pull()
                p.restart()
            else:
                bt.logging.success('No changes detected. Continuing.')

            # Wait 30 seconds.
            time.sleep(30)

    except KeyboardInterrupt:
        # Log and stop both processes.
        bt.logging.success('Interrupted by user. Exiting.')
        p.stop()

if __name__ == '__main__':
    main()
