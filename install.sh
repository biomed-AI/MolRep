#!/bin/bash

# set CUDA variable (defaults to cpu if no argument is provided to the script)
# available options for for pytorch 1.7.1 are cpu, cu92, cu100, cu101
CUDA_VERSION=cu110

# default pytorch version is 1.7.1
PYTORCH_VERSION=1.7.1

# set Your Conda Path
your_conda_path=/data/user/raojh/anaconda3/bin

# create virtual environment and activate it
conda env create -f environment.yaml
source ${your_conda_path}/activate
source activate MolRep

# install torch-geometric dependencies
pip install torch_scatter==2.0.6 torch_sparse==0.6.9 torch_cluster==1.5.9 torch_spline_conv==1.2.1 -f https://data.pyg.org/whl/torch-${PYTORCH_VERSION}+${CUDA_VERSION}.html
pip install torch-geometric==1.6.3

# setup for graphormer
python graphormer_setup.py build_ext --inplace