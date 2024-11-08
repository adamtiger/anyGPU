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

## Cuda install

Follow instructions on [this link](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#network-repo-installation-for-ubuntu).

Reboot and the following line to the bashrc:

```
export PATH=/usr/local/cuda-12.6/bin${PATH:+:${PATH}}
```

```
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64\
                         ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

Pay attention to the version too:
```
sudo apt install nvidia-utils-550
```

The driver needs to be installed separately:
(https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/index.html#ubuntu)

But for selecting the right one (pay attention for the versions!)
```
sudo apt install ubuntu-drivers-common
sudo ubuntu-drivers devices
```

```
sudo apt install nvidia-driver-xxx
```

Install cudnn too:
```
sudo apt-get -y install cudnn9-cuda-12
```

Other commands for information:
```
nvidia-smi --query-gpu=compute_cap --format=csv
```

```
lspci | grep -i nvidia
```

Loaded driver version check:
```
cat /proc/driver/nvidia/version
```

Checking cuda with torch:

```
torch.cuda.is_available()
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
MAX_JOBS=4 pip install transformer_engine[pytorch] --no-build-isolation
```
The last one takes a lot of time to complete. Several hours (around 3h).

For the memory problem this can help:
```
MAX_JOBS=1 pip install flash-attn --no-build-isolation
```

## Vulkan installation

Vulkan [linux install guide](https://vulkan.lunarg.com/doc/sdk/1.3.296.0/linux/getting_started_ubuntu.html)

latest cmake on ubuntu (sudo snap install cmake --classic)

```
sudo snap install cmake --classic
```

## Test it with running the test file

Problems (with zamba2 execution):
    - sm_75 is not enough (T4)
	- unable to execute on cpu!
	- bigger gpu machine is required!
	- A10G is sm_86

With A10G, the models work fine.
But zambra1.2b is quite inaccurate and hallucinates a lot.
