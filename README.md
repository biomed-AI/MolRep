# MolRep: Benchmarking Representation Learning Models for Molecular Property Prediction


 
## Summary

If you found this package useful, please cite [arxiv]() for now:
```

```

## Install & Usage
We provide a script to install the environment. You will need the conda package manager, which can be installed from [here](https://www.anaconda.com/products/individual).

To install the required packages, follow there instructions (tested on a linux terminal):

1) clone the repository

    git clone https://github.com/Jh-SYSU/MolRep

2) `cd` into the cloned directory

    cd MolRep

3) run the install script

    source install.sh [<your_cuda_version>]

Where `<your_cuda_version>` is an optional argument that can be either `cpu`, `cu92`, `cu100`, `cu101`. If you do not provide a cuda version, the script will default to `cpu`. The script will create a virtual environment named `ADMET_comparison`, with all the required packages needed to run our code. **Important:** do NOT run this command using `bash` instead of `source`!

## Data

### Current Dataset
|Dataset|Task|Task type|#Molecule|Splits|Metric|Reference|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|QM7|1|Regression|7160|Stratified|MAE|[Wu et al.]()|
|QM8|12|Regression|21786|Random|MAE|[Wu et al.]()|
|QM9|12|Regression|133885|Random|MAE|[Wu et al.]()|
|ESOL|1|Regression|1128|Random|RMSE|[Wu et al.]()|
|FreeSolv|1|Regression|642|Random|RMSE|[Wu et al.]()|
|Lipophilicity|1|Regression|4200|Random|RMSE|[Wu et al.]()|
|BBBP|1|Classification|2039|Scaffold|ROC-AUC|[Wu et al.]()|
|Tox21|12|Classification|7831|Random|ROC-AUC|[Wu et al.]()|
|SIDER|27|Classification|1427|Random|ROC-AUC|[Wu et al.]()|
|ClinTox|2|Classification|1478|Random|ROC-AUC|[Wu et al.]()|
|Liver injury|1|Classification|2788|Random|ROC-AUC|[Xu et al.]()|
|Mutagenesis|1|Classification|6511|Random|ROC-AUC|[Hansen et al.]()|
|hERG|1|Classification|4813|Random|ROC-AUC|[Li et al.]()|
|MUV|17|Classification|93087|Random|PRC-AUC|[Wu et al.]()|
|HIV|1|Classification|41127|Random|ROC-AUC|[Wu et al.]()|
|BACE|1|Classification|1513|Random|ROC-AUC|[Wu et al.]()|


## Methods

### Current Methods

#### Self-/unsupervised Models
|Methods|Descriptions|Reference|
|:-:|-|:-:|
|Mol2Vec|Mol2Vec is an unsupervised approach to learns vector representations of molecular substructures that point in similar directions for chemically related substructures.| [Jaeger et al.]()|
|N-Gram graph|N-gram graph is a simple unsupervised representation for molecules that first embeds the vertices in the molecule graph and then constructs a compact representation for the graph by assembling the ver-tex embeddings in short walks in the graph.|[Liu et al.]()|
|FP2Vec|FP2Vec is a molecular featurizer that represents a chemical compound as a set of trainable embedding vectors and combine with CNN model.|[Jeon et al.]()|
|VAE|VAE is a framework for training two neural networks (encoder and decoder) to learn a mapping from high-dimensional molecular representation into a lower-dimensional space.|[Kingma et al.]()|

#### Sequence Models
|Methods|Descriptions|Reference|
|:-:|-|:-:|
|BiLSTM|BiLSTM is an artificial recurrent neural network (RNN) architecture to encoding sequences from compound SMILES strings.|[Hochreiter et al.]()|
|SALSTM|SALSTM is a self-attention mechanism with improved BiLSTM for molecule representation.|[Zheng et al]()|
|Transformer|Transformer is a network based solely on attention mechanisms and dispensing with recurrence and convolutions entirely to encodes compound SMILES strings.|[Vaswani et al.]()|
|MAT|MAT is a molecule attention transformer utilized inter-atomic distances and the molecular graph structure to augment the attention mechanism.|[Maziarka et al.]()|

#### Graph Models
|Methods|Descriptions|Reference|
|:-:|-|:-:|
|DGCNN|DGCNN is a deep graph convolutional neural network that proposes a graph convolution model with SortPooling layer which sorts graph vertices in a consistent order to learning the embedding of molec-ular graph.| [Zhang et al.]()|
|GraphSAGE|GraphSAGE is a framework for inductive representation learning on molecular graphs that used to generate low-dimensional representations for atoms and performs sum, mean or max-pooling neigh-borhood aggregation to updates the atom representation and molecular representation.| [Hamilton et al.]()|
|GIN|GIN is the Graph Isomorphism Network that builds upon the limitations of GraphSAGE to capture different graph structures with the Weisfeiler-Lehman graph isomorphism test.|[Xu et al.]()|
|ECC|ECC is an Edge-Conditioned Convolution Network that learns a different parameter for each edge label (bond type) on the molecular graph, and neighbor aggregation is weighted according to specific edge parameters.|[Simonovsky et al.]()|
|DiffPool|DiffPool combines a differentiable graph encoder with its an adaptive pooling mechanism that col-lapses nodes on the basis of a supervised criterion to learning the representation of molecular graphs.|[Ying et al.]()|
|MPNN|MPNN is a message-passing graph neural network that learns the representation of compound molecular graph. It mainly focused on obtaining effective vertices (atoms) embedding| [Gilmer et al.]()|
|D-MPNN|DMPNN is another message-passing graph neural network that messages associated with directed edges (bonds) rather than those with vertices. It can make use of the bond attributes.|[Yang et al.]()|
|CMPNN|CMPNN is the graph neural network that improve the molecular graph embedding by strengthening the message interactions between edges (bonds) and nodes (atoms).|[Song et al.]()|


### Training

To train a model by K-fold, run [5-fold-training_example.ipynb](https://github.com/Jh-SYSU/MolRep/blob/main/Examples/5-fold-training_example.ipynb).

### Testing 

To test a pretrained model, run [testing-example.ipynb](https://github.com/Jh-SYSU/MolRep/blob/main/Examples/testing-example.ipynb).


### Results

#### Results on Classification Tasks.

|Datasets|BBBP|Tox21|SIDER|ClinTox|MUV|HIV|BACE|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|Mol2Vec|0.9213±0.0052|0.8139±0.0081|0.6043±0.0061|0.8572±0.0054|0.1178±0.0032|0.8413±0.0047|0.8284±0.0023|
|N-Gram graph|0.9012±0.0385|0.8371±0.0421|0.6482±0.0437|0.8753±0.0077|0.1011±0.0000|0.8378±0.0034|0.8472±0.0057|
|FP2Vec|0.8076±0.0032|0.8578±0.0076|0.6678±0.0068|0.8834±0.0432|0.0856±0.0031|0.7894±0.0052|0.8129±0.0492|
|VAE|0.8378±0.0031|0.8315±0.0382|0.6493±0.0762|0.8674±0.0124|0.0794±0.0001|0.8109±0.0381|0.8368±0.0762|
|BiLSTM|0.8391±0.0032|0.8279±0.0098|0.6092±0.0303|0.8319±0.0120|0.0382±0.0000|0.7962±0.0098|0.8263±0.0031|
|SALSTM|0.8482±0.0329|0.8253±0.0031|0.6308±0.0036|0.8317±0.0003|0.0409±0.0000|0.8034±0.0128|0.8348±0.0019|
|Transformer|0.9610±0.0119|0.8129±0.0013|0.6017±0.0012|0.8572±0.0032|0.0716±0.0017|0.8372±0.0314|0.8407±0.0738|
|MAT|0.9620±0.0392|0.8393±0.0039|0.6276±0.0029|0.8777±0.0149|0.0913±0.0001|0.8653±0.0054|0.8519±0.0504|
|DGCNN|0.9311±0.0434|0.7992±0.0057|0.6007±0.0053|0.8302±0.0126|0.0438±0.0000|0.8297±0.0038|0.8361±0.0034|
|GraphSAGE|0.9630±0.0474|0.8166±0.0041|0.6403±0.0045|0.9116±0.0146|0.1145±0.0000|0.8705±0.0724|0.9316±0.0360|
|GIN|0.8746±0.0359|0.8178±0.0031|0.5904±0.0000|0.8842±0.0004|0.0832±0.0000|0.8015±0.0328|0.8275±0.0034|
|ECC|0.9620±0.0003|0.8677±0.0090|0.6750±0.0092|0.8862±0.0831|0.1308±0.0013|0.8733±0.0025|0.8419±0.0092|
|DiffPool|0.8732±0.0391|0.8012±0.0130|0.6087±0.0130|0.8345±0.0233|0.0934±0.0001|0.8452±0.0042|0.8592±0.0391|
|MPNN|0.9321±0.0312|0.8440±0.014|0.6313±0.0121|0.8414±0.0294|0.0572±0.0001|0.8032±0.0092|0.8493±0.0013|
|DMPNN|0.9562±0.0070|0.8429±0.0391|0.6378±0.0329|0.8692±0.0051|0.0867±0.0032|0.8137±0.0072|0.8678±0.0372|
|CMPNN|0.9854±0.0215|0.8593±0.0088|0.6581±0.0020|0.9169±0.0065|0.1435±0.0002|0.8687±0.0003|0.8932±0.0019|