<p align="center"><img src="./fig/kgwas_logo.png" alt="logo" width="600px" /></p>

# Genetics discovery powered by functional genomics knowledge graph

## Overview

Genome-wide association studies (GWASs) have identified tens of thousands of disease-associated variants and provided critical insights into developing effective treatments. However, limited sample sizes have hindered the discovery of variants for less common and rare diseases.
Here, we introduce KGWAS, a novel geometric deep learning method that leverages a massive functional knowledge graph across variants and genes to improve detection power in small-cohort GWASs significantly.

## Installation

Install Pytorch Geometric follow [this instruction](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) and then do:

```bash
pip install KGWAS
```

## Core KGWAS API Usage


```python

from kgwas import KGWAS, KGWAS_Data
data = KGWAS_Data()
data.load_kg()

data.load_external_gwas(PATH)
data.process_gwas_file()
data.prepare_split()

run = KGWAS(data,
            weight_bias_track = True,
            device = device,
            proj_name = 'KGWAS',
            exp_name = exp_name,
            seed = seed)

run.initialize_model()
run.train(epoch = 10)
```

## Tutorial

| Notebook | Try on Colab | Description                                             |
----------|--------------|---------------------------------------------------------|
| [kgwas_tutorial.ipynb](demo/kgwas_tutorial.ipynb) | [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />]()   | Tutorial on key KGWAS API and functionalities. |
| [kgwas_use_your_own_gwas.ipynb](demo/kgwas_use_your_own_gwas.ipynb) | [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />]()   | Tutorial on applying KGWAS to your own GWAS summary statistics. |
| [kgwas_subsampling.ipynb](demo/kgwas_subsampling.ipynb) | [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />]()   | Tutorial on the subsampling analysis. |


### Cite Us

```bibtex
@misc{kgwas,
      title={Small-cohort GWAS discovery with AI over massive functional genomics knowledge graph},
      author={Kexin Huang and Tony Zeng and Soner Koc and Alexandra Pettet and Jingtian Zhou and Mika Jain and Dongbo Sun and Camilo Ruiz and Hongyu Ren and Laurence Howe and Tom Richardson and Adrian Cortes and Katie Aiello and Kim Branson and Andreas Pfenning and Jesse Engreitz and Martin Jinye Zhang and Jure Leskovec},
      year={2024}
}
```