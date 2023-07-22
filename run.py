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
    bt.logging.success('Checking git changes on origin/main' )
    result = subprocess.run(['git', 'fetch'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError('Error fetching git updates')
    result = subprocess.run(['git', 'diff', 'HEAD', 'origin/main'], stdout=subprocess.PIPE)
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
            if git_has_changes():
                bt.logging.success('Changes detected. Pulling updates and restarting...')
                git_pull()
                p.restart()
            time.sleep(10)
    except KeyboardInterrupt:
        bt.logging.success('Interrupted by user. Exiting.')
        p.stop()

if __name__ == '__main__':
    main()
