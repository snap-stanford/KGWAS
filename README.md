<p align="center"><img src="./fig/kgwas_logo.png" alt="logo" width="600px" /></p>

# Genetics discovery powered by functional genomics knowledge graph

Genome-wide association studies (GWASs) have identified tens of thousands of disease-associated variants and provided critical insights into developing effective treatments. However, limited sample sizes have hindered the discovery of variants for less common and rare diseases.
Here, we introduce KGWAS, a novel geometric deep learning method that leverages a massive functional knowledge graph across variants and genes to improve detection power in small-cohort GWASs significantly.

## Installation

Install Pytorch Geometric follow [this instruction](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) and then do:

```bash
pip install KGWAS
```

## Data download
To ensure fast user experience, we provide a default fast mode of KGWAS, which uses Enformer embedding for variant feature and ESM embedding for gene features (instead of the baselineLD for variant and PoPS for gene since they are large files). For the fast mode, you do not need to download any data, the KGWAS API will automatically download the relevant files. This mode can be used to apply KGWAS to your own GWAS sumstats. 

If you want to (1) use the full mode of KGWAS or (2) access the null/causal simulations or (3) access the 21 subsampled GWAS sumstats across various sample sizes or (4) analyze the KGWAS sumstats for subsampled data or (5) analyze the KGWAS sumstats for all UKBB ICD10 diseases, please download everything from [here](). Note that this file is large (around 40GB) and may take a while to download.

## Core KGWAS API Usage

```python

from kgwas import KGWAS, KGWAS_Data
data = KGWAS_Data(data_path = './data') ## initialize KGWAS data class with data path

data.load_kg() ## load the knowledge graph
data.load_external_gwas(PATH) ## load the GWAS file
data.process_gwas_file() ## process the GWAS file
data.prepare_split() ## prepare the train/val/test split

run = KGWAS(data, device = 'cuda:0', seed = 1) ## initialize KGWAS model
run.initialize_model()

run.train(epoch = 10) ## train the model
```


## Tutorial [[Coming soon!]]

| Notebook | Try on Colab | Description                                             |
----------|--------------|---------------------------------------------------------|
| [Introduction](demo/introduction.ipynb) | [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />]()   | Tutorial on key KGWAS API and functionalities. |
| [Apply KGWAS to your own sumstats](demo/kgwas_use_your_own_gwas.ipynb) | [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />]()   | Tutorial on applying KGWAS to your own GWAS summary statistics. |
| [Use alternative variant/gene/program embedding](demo/kgwas_subsampling.ipynb) | [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />]()   | Tutorial on using alternative variant/gene/program embedding (e.g. foundation model embedding). |
| [Simulation analysis](demo/kgwas_subsampling.ipynb) | [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />]()   | Tutorial on the subsampling analysis. |
| [Subsampling analysis](demo/kgwas_subsampling.ipynb) | [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />]()   | Tutorial on the subsampling analysis. |


## Extended API Usage

`data.load_kg(snp_init_emb = 'enformer', go_init_emb = 'random', gene_init_emb = 'esm', sample_edges = False, sample_ratio = 1)`


### Cite Us

```bibtex
@misc{kgwas,
      title={Small-cohort GWAS discovery with AI over massive functional genomics knowledge graph},
      author={Kexin Huang and Tony Zeng and Soner Koc and Alexandra Pettet and Jingtian Zhou and Mika Jain and Dongbo Sun and Camilo Ruiz and Hongyu Ren and Laurence Howe and Tom Richardson and Adrian Cortes and Katie Aiello and Kim Branson and Andreas Pfenning and Jesse Engreitz and Martin Jinye Zhang and Jure Leskovec},
      year={2024}
}
```