# ADMET_comparison
 
## Summary

If you found this package useful, please cite [arxiv]() for now:
```

```

## Install & Usage
We provide a script to install the environment. You will need the conda package manager, which can be installed from [here](https://www.anaconda.com/products/individual).

To install the required packages, follow there instructions (tested on a linux terminal):

1) clone the repository

    git clone https://github.com/Jh-SYSU/ADMET_Comparison

2) `cd` into the cloned directory

    cd ADMET_Comparison

3) run the install script

    source install.sh [<your_cuda_version>]

Where `<your_cuda_version>` is an optional argument that can be either `cpu`, `cu92`, `cu100`, `cu101`. If you do not provide a cuda version, the script will default to `cpu`. The script will create a virtual environment named `ADMET_comparison`, with all the required packages needed to run our code. **Important:** do NOT run this command using `bash` instead of `source`!


### Instructions

To reproduce the experiments, running the codes as follows:
`python main.py --model_name <Model_name> --dataset-name <name> --gpu 0 --k_fold 5`