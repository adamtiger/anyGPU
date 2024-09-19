# How to set up g4dn instance on AWS

## Install miniconda

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

Execute the install and follow the commands.

## Create environment

```
conda create -n anygpu python=3.10
```

## Clone the repository

E.g.:

```
git clone https://github.com/Zyphra/Zamba2.git
```

Connect with vscode remotely and install libraries.

Also needs to be installed (the flash-attn python package install needs more than 32gb ram!): 

```
pip install -U mamba-ssm causal-conv1d
pip install transformer_engine[pytorch]
```

For the memory problem this can help:
```
MAX_JOBS=4 pip install flash-attn --no-build-isolation
```


## Cuda install

Follow instructions on [this link](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#network-repo-installation-for-ubuntu).

Reboot and the following line to the bashrc:

```
export PATH=/usr/local/cuda-12.6/bin${PATH:+:${PATH}}
```

The driver needs to be installed separately:
```
sudo apt-get install nvidia-open
```

Install cudnn too:
```
sudo apt-get -y install cudnn9-cuda-12
```

## Test it with running the test file


