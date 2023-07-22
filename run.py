import os
import sys
import time
import subprocess

def git_has_changes():
    result = subprocess.run(['git', 'fetch'])
    if result.returncode != 0:
        raise RuntimeError('Error fetching git updates')
    result = subprocess.run(['git', 'diff', 'HEAD', 'origin/main'], stdout=subprocess.PIPE)
    return result.stdout != b''

def git_pull():
    result = subprocess.run(['git', 'pull'])
    if result.returncode != 0:
        raise RuntimeError('Error pulling git updates')

def main():
    try:
        print('Starting.')
        os.execv(sys.executable, ['python'] + sys.argv)
        while True:
            if git_has_changes():
                print('Changes detected. Pulling updates and restarting...')
                git_pull()
                os.execv(sys.executable, ['python'] + sys.argv)
            time.sleep(10)
    except KeyboardInterrupt:
        print('Interrupted by user. Exiting.')
        sys.exit(0)

if __name__ == '__main__':
    main()
