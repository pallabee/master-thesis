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
1. For approaches 2 and 3, use the .dat files from the datsets for graph generation wherever .dat files are present, like dataset_link.dat,dataset_node.dat 
1. For approach_3/diffusion_graph_gen and approach_3/diffusion_node_feature_gen, follow the installation steps of DiGress: https://github.com/cvignac/DiGress
