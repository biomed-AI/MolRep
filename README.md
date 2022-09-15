# MolRep: A Deep Representation Learning Library for Molecular Property Prediction


 
## Summary
 MolRep is a Python package for fairly measuring algorithmic progress on chemical property datasets. It currently provides a complete re-evaluation of 16 state-of-the-art deep representation models over 16 benchmark property datsaets.

<p align='center'>
<img src="https://github.com/biomed-AI/MolRep/blob/main/ADMET-TOC.jpg" alt="architecture"/>
</p>

If you found this package useful, please cite our papers: [MolRep](https://doi.org/10.1101/2021.01.13.426489) and [Mol-XAI](https://arxiv.org/abs/2107.04119) for now:
```
@article{rao2021molrep,
  title={MolRep: A Deep Representation Learning Library for Molecular Property Prediction},
  author={Rao, Jiahua and Zheng, Shuangjia and Song, Ying and Chen, Jianwen and Li, Chengtao and Xie, Jiancong and Yang, Hui and Chen, Hongming and Yang, Yuedong},
  journal={bioRxiv},
  year={2021},
  publisher={Cold Spring Harbor Laboratory}
}

@article{rao2021quantitative,
  title={Quantitative Evaluation of Explainable Graph Neural Networks for Molecular Property Prediction},
  author={Rao, Jiahua and Zheng, Shuangjia and Yang, Yuedong},
  journal={arXiv preprint arXiv:2107.04119},
  year={2021}
}

```

## Install & Usage
We provide a script to install the environment. You will need the conda package manager, which can be installed from [here](https://www.anaconda.com/products/individual).

To install the required packages, follow there instructions (tested on a linux terminal):

1) clone the repository

    git clone https://github.com/biomed-AI/MolRep

2) `cd` into the cloned directory

    cd MolRep

3) run the install script

    source install.sh

Where `<your_conda_path>` is your conda path, and `<CUDA_VERSION>` is an optional argument that can be either `cpu`, `cu92`, `cu100`, `cu101`, `cu110`. If you do not provide a cuda version, the script will default to `cu110`. The script will create a virtual environment named `MolRep`, with all the required packages needed to run our code. **Important:** do NOT run this command using `bash` instead of `source`!

## Data

Data (including Explainable Dataset) could be download from [Google_Driver](https://drive.google.com/drive/folders/1vGlhE3TJ4AhvUCa3ODdFw3O-zOMH1s7J?usp=sharing)

[!NEWS] The human experiments fro explainability task (molecules and results) are available at [Here](https://drive.google.com/drive/folders/1s1_GnX8fUp-N6GkGq5rZevpvmCV_KrMt?usp=sharing)

### Current Dataset
|Dataset|Task|Task type|#Molecule|Splits|Metric|Reference|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|QM7|1|Regression|7160|Stratified|MAE|[Wu et al.](https://arxiv.org/abs/1703.00564)|
|QM8|12|Regression|21786|Random|MAE|[Wu et al.](https://arxiv.org/abs/1703.00564)|
|QM9|12|Regression|133885|Random|MAE|[Wu et al.](https://arxiv.org/abs/1703.00564)|
|ESOL|1|Regression|1128|Random|RMSE|[Wu et al.](https://arxiv.org/abs/1703.00564)|
|FreeSolv|1|Regression|642|Random|RMSE|[Wu et al.](https://arxiv.org/abs/1703.00564)|
|Lipophilicity|1|Regression|4200|Random|RMSE|[Wu et al.](https://arxiv.org/abs/1703.00564)|
|BBBP|1|Classification|2039|Scaffold|ROC-AUC|[Wu et al.](https://arxiv.org/abs/1703.00564)|
|Tox21|12|Classification|7831|Random|ROC-AUC|[Wu et al.](https://arxiv.org/abs/1703.00564)|
|SIDER|27|Classification|1427|Random|ROC-AUC|[Wu et al.](https://arxiv.org/abs/1703.00564)|
|ClinTox|2|Classification|1478|Random|ROC-AUC|[Wu et al.](https://arxiv.org/abs/1703.00564)|
|Liver injury|1|Classification|2788|Random|ROC-AUC|[Xu et al.](https://pubs.acs.org/doi/10.1021/acs.jcim.5b00238)|
|Mutagenesis|1|Classification|6511|Random|ROC-AUC|[Hansen et al.](https://pubs.acs.org/doi/10.1021/ci900161g)|
|hERG|1|Classification|4813|Random|ROC-AUC|[Li et al.](https://doi.org/10.1002/minf.201700074)|
|MUV|17|Classification|93087|Random|PRC-AUC|[Wu et al.](https://arxiv.org/abs/1703.00564)|
|HIV|1|Classification|41127|Random|ROC-AUC|[Wu et al.](https://arxiv.org/abs/1703.00564)|
|BACE|1|Classification|1513|Random|ROC-AUC|[Wu et al.](https://arxiv.org/abs/1703.00564)|


## Methods

### Current Methods

#### Self-/unsupervised Models
|Methods|Descriptions|Reference|
|:-:|-|:-:|
|Mol2Vec|Mol2Vec is an unsupervised approach to learns vector representations of molecular substructures that point in similar directions for chemically related substructures.| [Jaeger et al.](https://pubs.acs.org/doi/full/10.1021/acs.jcim.7b00616)|
|N-Gram graph|N-gram graph is a simple unsupervised representation for molecules that first embeds the vertices in the molecule graph and then constructs a compact representation for the graph by assembling the ver-tex embeddings in short walks in the graph.|[Liu et al.](http://papers.neurips.cc/paper/9054-n-gram-graph-simple-unsupervised-representation-for-graphs-with-applications-to-molecules.pdf)|
|FP2Vec|FP2Vec is a molecular featurizer that represents a chemical compound as a set of trainable embedding vectors and combine with CNN model.|[Jeon et al.](https://academic.oup.com/bioinformatics/article/35/23/4979/5487389)|
|VAE|VAE is a framework for training two neural networks (encoder and decoder) to learn a mapping from high-dimensional molecular representation into a lower-dimensional space.|[Kingma et al.](https://arxiv.org/abs/1312.6114)|

#### Sequence Models
|Methods|Descriptions|Reference|
|:-:|-|:-:|
|BiLSTM|BiLSTM is an artificial recurrent neural network (RNN) architecture to encoding sequences from compound SMILES strings.|[Hochreiter et al.](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.676.4320&rep=rep1&type=pdf)|
|SALSTM|SALSTM is a self-attention mechanism with improved BiLSTM for molecule representation.|[Zheng et al](https://pubs.acs.org/doi/10.1021/acs.jcim.8b00803)|
|Transformer|Transformer is a network based solely on attention mechanisms and dispensing with recurrence and convolutions entirely to encodes compound SMILES strings.|[Vaswani et al.](https://arxiv.org/abs/1706.03762)|
|MAT|MAT is a molecule attention transformer utilized inter-atomic distances and the molecular graph structure to augment the attention mechanism.|[Maziarka et al.](https://arxiv.org/abs/2002.08264)|

#### Graph Models
|Methods|Descriptions|Reference|
|:-:|-|:-:|
|DGCNN|DGCNN is a deep graph convolutional neural network that proposes a graph convolution model with SortPooling layer which sorts graph vertices in a consistent order to learning the embedding of molec-ular graph.| [Zhang et al.](https://muhanzhang.github.io/papers/AAAI_2018_DGCNN.pdf)|
|GraphSAGE|GraphSAGE is a framework for inductive representation learning on molecular graphs that used to generate low-dimensional representations for atoms and performs sum, mean or max-pooling neigh-borhood aggregation to updates the atom representation and molecular representation.| [Hamilton et al.](https://arxiv.org/abs/1706.02216)|
|GIN|GIN is the Graph Isomorphism Network that builds upon the limitations of GraphSAGE to capture different graph structures with the Weisfeiler-Lehman graph isomorphism test.|[Xu et al.](https://arxiv.org/abs/1810.00826)|
|ECC|ECC is an Edge-Conditioned Convolution Network that learns a different parameter for each edge label (bond type) on the molecular graph, and neighbor aggregation is weighted according to specific edge parameters.|[Simonovsky et al.](https://arxiv.org/abs/1704.02901)|
|DiffPool|DiffPool combines a differentiable graph encoder with its an adaptive pooling mechanism that col-lapses nodes on the basis of a supervised criterion to learning the representation of molecular graphs.|[Ying et al.](https://arxiv.org/abs/1806.08804)|
|MPNN|MPNN is a message-passing graph neural network that learns the representation of compound molecular graph. It mainly focused on obtaining effective vertices (atoms) embedding| [Gilmer et al.](https://arxiv.org/abs/1704.01212)|
|D-MPNN|DMPNN is another message-passing graph neural network that messages associated with directed edges (bonds) rather than those with vertices. It can make use of the bond attributes.|[Yang et al.](https://pubs.acs.org/doi/10.1021/acs.jcim.9b00237)|
|CMPNN|CMPNN is the graph neural network that improve the molecular graph embedding by strengthening the message interactions between edges (bonds) and nodes (atoms).|[Song et al.](https://www.ijcai.org/Proceedings/2020/0392.pdf)|


### Training

To train a model by K-fold, run [5-fold-training_example.ipynb](https://github.com/Jh-SYSU/MolRep/blob/main/Examples/5-fold-training_example.ipynb).

### Testing 

To test a pretrained model, run [testing-example.ipynb](https://github.com/Jh-SYSU/MolRep/blob/main/Examples/testing-example.ipynb).


### Explainable
To explain the GNN model, run [Explainer_Experiments.py](https://github.com/Jh-SYSU/MolRep/blob/main/Examples/Explainer_Experiments.py)


More results will be updated soon.
