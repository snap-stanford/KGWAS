# KGWAS: genetics discovery powered by functional genomics knowledge graph


### Installation

```bash
pip install kgwas
```

### How to use KGWAS


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


### Reproducing experiments


### Cite Us