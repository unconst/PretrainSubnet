# PretrainSubnet

# Install

1. You require python3.11 to run:
[Linux Install](https://iohk.zendesk.com/hc/en-us/articles/16724475448473-Install-Python-3-11-on-ubuntu)
[Mac Install](https://pythontest.com/python/installing-python-3-11/)

2. Install requirements.
```bash
python3.11 -m pip install -r requirements.txt
```

3. Install [Weights and Biases](https://docs.wandb.ai/quickstart) and login via:
```bash
wandb login
```

# Run
To run a miner or multiple:
```bash
python3.11 pretrain/neuron.py --wallet.hotkey miner1 --axon.port 8091
...
python3.11 pretrain/neuron.py --wallet.hotkey miner1 --axon.port 8092
```