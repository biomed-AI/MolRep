
# set CUDA variable (defaults to cpu if no argument is provided to the script)
# available options for for pytorch 1.7.1 are cpu, cu92, cu100, cu101
CUDA_VERSION=cu110

# set Your Conda Path
your_conda_path=/home/anaconda3/bin/

# create virtual environment and activate it
conda env create -f environment.yaml
source ${your_conda_path}activate MolRep


# default pytorch version is 1.7.1
PYTORCH_VERSION=1.7.1

# install pytorch
if [[ "$CUDA_VERSION" == "cpu" ]]; then 
  pip install torch==1.7.1+cpu torchvision==0.8.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
elif [[ "$CUDA_VERSION" == 'cu92' ]]; then
  pip install torch==1.7.1+cu92 torchvision==0.8.2+cu92 -f https://download.pytorch.org/whl/torch_stable.html
elif [[ "$CUDA_VERSION" == 'cu101' ]]; then
  pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 -f https://download.pytorch.org/whl/torch_stable.html
elif [[ "$CUDA_VERSION" == 'cu102' ]]; then
  pip install torch==1.7.1 torchvision==0.8.2
elif [[ "$CUDA_VERSION" == 'cu110' ]]; then
  pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
fi

# install torch-geometric dependencies
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${PYTORCH_VERSION}+${CUDA_VERSION}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${PYTORCH_VERSION}+${CUDA_VERSION}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${PYTORCH_VERSION}+${CUDA_VERSION}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${PYTORCH_VERSION}+${CUDA_VERSION}.html
pip install torch-geometric==1.6.3
