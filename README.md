# Pneuma-Sigmod

This repository is created for Pneuma's submission to SIGMOD.

## Setup

Make sure Conda is installed.

Type "conda" in terminal, if "command not found", install Conda from https://docs.conda.io/en/latest/

**Create Conda environment. Important: Installing nvidia cuda-toolkit is required for flash-attn**
```shell
conda create --name pneuma-sigmod python=3.12 -y
conda activate pneuma-sigmod
conda install -c nvidia cuda-toolkit -y
export CUDA_HOME=$CONDA_PREFIX
```

**Install requirements**
```shell
bash setup.sh
```

