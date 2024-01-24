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
3. For approaches 2 and 3, use the .dat files from the datsets for graph generation wherever .dat files are present, like dataset_link.dat,dataset_node.dat 
4. To run code for diffusion_graph_gen and diffusion_node_feature_gen in approach_3, follow the installation steps of DiGress: https://github.com/cvignac/DiGress
