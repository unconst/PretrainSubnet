# PretrainSubnet

# Install

1. You require python3.11 to run/install this package.
[Linux Install](https://iohk.zendesk.com/hc/en-us/articles/16724475448473-Install-Python-3-11-on-ubuntu)
[Mac Install](https://pythontest.com/python/installing-python-3-11/)

2. Install package.
```bash
python3.11 -m pip install -e .
```

3. Install [Weights and Biases](https://docs.wandb.ai/quickstart) and login via:
```bash
wandb login
```

# Run
To run a miner or multiple:
```bash
# To run from the auto update script.
python3.11 run.py --wallet.hotkey miner1 --axon.port 8091
...
# To run the main script directly.
python3.11 pretrain/neuron.py --wallet.hotkey miner1 --axon.port 8092
```
