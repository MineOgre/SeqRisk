# Demo: Running SeqRisk Model Variants

This repository includes a small demo setup to run the SeqRisk methods.

## 1) Create the environment

Create the conda environment from the provided YAML file:

conda env create -f SeqRisk.yml
conda activate SeqRisk

## Step 2 — Run the model variants

Four variants of the model can be run as follows (each uses a config file in `./config_files/`):

### SeqRisk: LVAE + Transformer

python LVAE.py --f=./config_files/SeqRisk-LVAE-Transformer.txt

### SeqRisk: VAE + Transformer

python VAE.py --f=./config_files/SeqRisk-VAE-Transformer.txt


### SeqRisk: VAE + MLP

python VAE.py --f=./config_files/SeqRisk-VAE-MLP.txt


### SeqRisk: Transformer only

python survival_transformer.py --f=./config_files/SeqRisk-Transformer-only.txt
