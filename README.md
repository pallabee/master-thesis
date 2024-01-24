# master-thesis
This repo contains all code for the implementation done in Master's Thesis.

Source of datasets 

1. Graph Generation:
- DBLP and IMDB : https://github.com/lingchen0331/HGEN
- PubMed : https://github.com/yangji9181/HNE/tree/master/Data

2. Node Feature Generation:
- DBLP : https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.DBLP.html
- IMDB : https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.IMDB.html
- PubMed : https://github.com/yangji9181/HNE/tree/master/Data
  
Instructions to run code
1. To run all jupyter notebooks create a conda environment using
   <br />
   conda create --name myenv --file requirements.txt
   <br />
   conda activate myenv
   <br />
<html>

2. To run code for diffusion_graph_gen and diffusion_node_feature_gen in approach_3, follow the installation steps of DiGress: https://github.com/cvignac/DiGress
3. Use the .dat files from the datasets for Graph Generation wherever .dat files are present in code, e.g. dataset_link.dat, dataset_node.dat 
