# Pneuma-Sigmod

This repository is created for Pneuma's submission to SIGMOD.

## Setup

**Create Conda environment**
```shell
conda create --name pneuma-sigmod python=3.12 -y
conda activate pneuma-sigmod
conda install -c nvidia cuda-toolkit -y
export CUDA_HOME=$CONDA_PREFIX
```

**Install requirements**
```shell
pip install -r requirements.txt
cd chroma
git init
cd ..
wget https://storage.googleapis.com/pneuma_open/chroma.tar
tar -xf chroma.tar
rm chroma.tar
pip install ./chroma/
```

