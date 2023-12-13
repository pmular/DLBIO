## Few-Shot Learning via Optimal Transport



### Code Map

We addded the following files 

- backbone.fcnet.CustomFCNet 
- conf.dataset.method.custom_sot.yaml 
- conf.dataset.method.protonet_attention.yaml 
- conf.dataset.method.protonet_sot.yaml 
- methods.custom_sot.py 
- methods.protonet_attention.py
- methods.protonet_sot.py
- methods.transform.py
- finetune.py


## Conda

We have not added any additional package so to reproduce the environment the instructions remain the same

Create a conda env and install requirements with:

```bash
conda env create -f environment.yml 
```

Before each run, activate the environment with:

```bash
conda activate few-shot-benchmark 
```


## How to reproduce our results

To run any of the models in any of the two datasets one needs to simply use the following command 
```bash
python3 run.py exp.name={exp_name} method={method_name} dataset={dataset}
```
where the options are 

- **Datasets**: *swissprot*, *tabula_muris*
- **Methods**: *protonet*, *protonet_sot*, *protonet_attention*, *custom_sot*

The hyperparameters need to be specified in the yaml files. Our best model in *swissprot* is *protonet_sot* using the following hyperparameters

```bash
lr: 0.001  ot_reg: 0.01  width: 128  depth: 1  dropout: 0
```

while for *tabula_muris* is *custom_sot* with 

```bash
lr: 0.0005  ot_reg: 0.05  width: 256  depth: 1  dropout: 0.1
```


## Hyperparameter Tunning

We tunned our hyperparameters via Weights and Biases sweeps. It can be done by simply running

```bash
python3 finetune.py
```

previously specifying in the file the model and dataset to be used.

