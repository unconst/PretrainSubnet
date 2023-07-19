# PretrainSubnet

# Install

1. Install requirements.
```bash
python -m pip install -r requirements.txt
```

2. Install [Weights and Biases](https://docs.wandb.ai/quickstart) and login via:
```bash
wandb login
```

# Run
To run a miner or multiple:
```bash
python miner.py --wallet.hotkey miner1 --axon.port 8091
...
python miner.py --wallet.hotkey miner1 --axon.port 8092
```