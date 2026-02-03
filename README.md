# Steps to Run A demo with the specified Methods


1- Environment can be created with SeqRisk.yml file
2- Four variants of the model can be run as follows:
For SeqRisk: LVAE+Tranformer ;

python LVAE.py --f=./config_files/SeqRisk-LVAE-Tranformer.txt 

For SeqRisk: VAE+Tranformer ;

python VAE.py --f=./config_files/SeqRisk-VAE-Tranformer.txt


For SeqRisk: VAE+MLP ;

python VAE.py --f=./config_files/SeqRisk-VAE-MLP.txt

For SeqRisk: Transformer only ; 

python survival_transformer.py --f=./config_files/SeqRisk-Tranformer-only.txt