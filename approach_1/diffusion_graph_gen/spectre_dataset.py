import os
import pathlib
import torch
import pandas as pd
import numpy as np
import networkx as nx
import warnings
warnings.filterwarnings('ignore')
import csv
from littleballoffur import ForestFireSampler
import torch.nn.functional as F
import torch_geometric.utils
from torch_geometric.data import InMemoryDataset
from abstract_dataset import AbstractDataModule, AbstractDatasetInfos

def remap_indices(G):
    val_list = [*range(0, G.number_of_nodes(), 1)]
    return dict(zip(G,val_list))
class SpectreGraphDataset(InMemoryDataset):
    def __init__(self, dataset_name, split, root, node_size,transform=None, pre_transform=None, pre_filter=None):
        self.dataset_name = dataset_name
        self.split = split
        self.node_size = node_size
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    @property
    def processed_file_names(self):
            return [self.split + '.pt']

    def process(self):
        data_list = []
        #------------------------------------PubMed------------------------------------------------------

        # df_nodes = pd.read_table(('node_pubmed.dat'), names=['node_id', 'node_name', 'node_type', 'node_attributes'],
        #                          quoting=csv.QUOTE_NONE)
        # df_edges = pd.read_table(('link_pubmed.dat'), names=['source', 'target', 'link_type', 'link_weight'])
        # df_labels_train = pd.read_table(('label_pubmed.dat'), names=['node_id', 'node_name', 'node_type', 'node_label'])
        #
        # G = nx.from_pandas_edgelist(
        #     df_edges,
        #     #edge_attr="link_type",
        #     create_using=nx.Graph(),
        # )
        # pubmed_node_features = df_nodes[['node_id', 'node_type', 'node_attributes']]
        # pubmed_node_features = pubmed_node_features.rename(columns={"node_attributes": "feature"})
        #
        # # Add node attributes
        # nodes_attr = pubmed_node_features.set_index('node_id').to_dict(orient='index')
        # nx.set_node_attributes(G, nodes_attr)
        # for n in G.nodes:
        #     if n in df_labels_train['node_id'].unique():
        #         # print(n)
        #         G.nodes[n]["label"] = df_labels_train.loc[df_labels_train['node_id'] == n, 'node_label'].values[0]

        # ------------------------------------PubMed------------------------------------------------------
        # ------------------------------------DBLP------------------------------------------------------
        #df_nodes = pd.read_table(('dblp_node.dat'), sep=' ', names=['node_id', 'node_name', 'node_type'],
         #                        encoding='latin-1')
        #df_edges = pd.read_table(('dblp_link.dat'), sep=' ', names=['source', 'target'])

        # G = nx.from_pandas_edgelist(
        #     df_edges,
        #     create_using=nx.Graph()
        # )
        # dblp_node_features = df_nodes[['node_id', 'node_type']]
        # nodes_attr = dblp_node_features.set_index('node_id').to_dict(orient='index')
        # nx.set_node_attributes(G, nodes_attr)

        # ------------------------------------DBLP------------------------------------------------------
        # ------------------------------------IMDB------------------------------------------------------
        df_nodes = pd.read_table(
            ('imdb_node.dat'), sep=' ',
            names=['node_id', 'node_name', 'node_type'], encoding='latin-1')
        df_edges = pd.read_table(
            ('imdb_link.dat'), sep=' ',
            names=['source', 'target'])
        #Removing Genre nodes
        node3_list = list(df_nodes[df_nodes['node_type'] == 3]['node_id'])
        df_edges = df_edges[~df_edges['source'].isin(node3_list)]
        df_edges = df_edges[~df_edges['target'].isin(node3_list)]

        G = nx.from_pandas_edgelist(
            df_edges,
            create_using=nx.Graph()
        )
        imdb_node_features = df_nodes[['node_id', 'node_type']]
        nodes_attr = imdb_node_features.set_index('node_id').to_dict(orient='index')
        nx.set_node_attributes(G, nodes_attr)
        # ------------------------------------IMDB------------------------------------------------------

        model = ForestFireSampler(self.node_size)
        subgraph = model.sample(G)
        mapping = remap_indices(subgraph)
        subgraph = nx.relabel_nodes(subgraph, mapping)
        node_data = subgraph.nodes(data=True)

        node_type = []
        node_id = []
        for key, value in node_data:
            node_id.append(key)

            node_type.append(value['node_type'])

        subgraph_df = pd.DataFrame()
        subgraph_df["node_id"] = node_id
        subgraph_df["node_type"] = node_type



        A = nx.adjacency_matrix(subgraph)
        adj = torch.tensor(np.asarray(A.todense()))

        n = adj.shape[-1]
        #node_classes = [0, 1, 2,3]
        node_classes = [0, 1, 2]
        node_types = node_type

        X = F.one_hot(torch.tensor(node_types), num_classes=len(node_classes)).float()
        y = torch.zeros([1, 0]).float()

        edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)
        edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
        edge_attr[:, 1] = 1
        num_nodes = n * torch.ones(1, dtype=torch.long)

        data = torch_geometric.data.Data(x=X, edge_index=edge_index, edge_attr=edge_attr,
                                         y=y, n_nodes=num_nodes)
        data_list.append(data)

        if self.pre_filter is not None and not self.pre_filter(data):
            pass
        if self.pre_transform is not None:
            data = self.pre_transform(data)

        data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])



class SpectreGraphDataModule(AbstractDataModule):
    def __init__(self, cfg,size):
        self.cfg = cfg
        self.datadir = cfg.dataset.datadir
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[0]
        root_path = os.path.join(base_path, self.datadir)


        datasets = {'train': SpectreGraphDataset(dataset_name=self.cfg.dataset.name,
                                                 split='train', root=root_path, node_size = size),
                    'val': SpectreGraphDataset(dataset_name=self.cfg.dataset.name,
                                        split='val', root=root_path,node_size = size),
                    'test': SpectreGraphDataset(dataset_name=self.cfg.dataset.name,
                                        split='test', root=root_path,node_size = size)}
        # print(f'Dataset sizes: train {train_len}, val {val_len}, test {test_len}')

        super().__init__(cfg, datasets)
        self.inner = self.train_dataset

    def __getitem__(self, item):
        return self.inner[item]


class SpectreDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule):
        self.datamodule = datamodule
        self.name = 'nx_graphs'
        self.n_nodes = self.datamodule.node_counts()
        #self.node_types = torch.tensor([0,1,2,3])
        self.node_types = torch.tensor([0, 1, 2])
        self.edge_types = self.datamodule.edge_counts()


        super().complete_infos(self.n_nodes, self.node_types)

