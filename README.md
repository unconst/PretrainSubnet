# PretrainSubnet
Uses cocktailSGD-like compression to transfer gradients across the network

## Install

Install cuda tools in a conda env
```bash
conda create -n pretrain python=3.10
conda activate pretrain
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c conda-forge cupy nccl #cudatoolkit=11.8
```

Install the rest of the requirements
```bash
python -m pip install -r requirements.txt
```