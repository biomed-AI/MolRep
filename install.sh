# default pytorch version is 1.5.0
PYTORCH_VERSION=1.5.0

# set CUDA variable (defaults to cpu if no argument is provided to the script)
# available options for for pytorch 1.4.0 are cpu, cu92, cu100, cu101
CUDA_VERSION=${1:-cpu}

# create virtual environment and activate it
conda env create -f environment.yaml

# install requirements
pip install -r requirements.txt

# install pytorch
if [[ "$CUDA_VERSION" == "cpu" ]]; then 
  conda install pytorch==${PYTORCH_VERSION} cpuonly -y
elif [[ "$CUDA_VERSION" == 'cu92' ]]; then
  conda install pytorch==${PYTORCH_VERSION} cudatoolkit=9.2 -y
elif [[ "$CUDA_VERSION" == 'cu100' ]]; then
  conda install pytorch==${PYTORCH_VERSION} cudatoolkit=10.0 -y
elif [[ "$CUDA_VERSION" == 'cu101' ]]; then
    conda install pytorch==${PYTORCH_VERSION} cudatoolkit=10.1 -y
fi

# install torch-geometric dependencies
pip install torch-scatter==latest+${CUDA_VERSION} -f https://pytorch-geometric.com/whl/torch-${PYTORCH_VERSION}.html
pip install torch-sparse==latest+${CUDA_VERSION} -f https://pytorch-geometric.com/whl/torch-${PYTORCH_VERSION}.html
pip install torch-cluster==latest+${CUDA_VERSION} -f https://pytorch-geometric.com/whl/torch-${PYTORCH_VERSION}.html
pip install torch-spline-conv==latest+${CUDA_VERSION} -f https://pytorch-geometric.com/whl/torch-${PYTORCH_VERSION}.html
pip install torch-geometric==1.6.1
