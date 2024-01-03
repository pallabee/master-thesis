import os
import csv

import pathlib
import torch
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_selection import VarianceThreshold
import torch_geometric.transforms as T
from torch_geometric.datasets.dblp import DBLP
from torch_geometric.datasets import IMDB
import torch_geometric.utils
from torch_geometric.data import InMemoryDataset
from abstract_dataset import AbstractDataModule, AbstractDatasetInfos


def apply_threshold(df):
    return df.applymap(lambda x: 0.0 if x < 0.5 else 1.0)
def convert_string_to_float(df):
    return df['node_attributes'].apply(lambda x: np.fromstring(x, dtype=float, sep=',' ))
def preprocess_class(df_class):
    df_class = df_class.reset_index()
    df_class = convert_string_to_float(df_class)

    x = torch.tensor(df_class).float()
    disease_class = pd.DataFrame(x.numpy())
    return disease_class

class FeatureDataset(InMemoryDataset):
    def __init__(self, dataset_name, split, root,transform=None, pre_transform=None, pre_filter=None):
        self.dataset_name = dataset_name
        self.split = split

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    @property
    def processed_file_names(self):
            return [self.split + '.pt']

    def download(self):
        # """
        # Download dataset
        # """

        train_data = []
        val_data = []
        test_data = []
        #class0 = pd.DataFrame()
        imp_feat = pd.DataFrame()

        if self.dataset_name == 'dblp':
            print('Dataset name',self.dataset_name)
            dataset = DBLP(root='./dblp_data', transform=T.Constant(node_types='conference'))
            data = dataset[0]
            #Author--------------------------------------------------
            author = data['author'].x.tolist()
            df = pd.DataFrame(author)
            df['class'] = data['author'].y.tolist()
            # Feature selection for Author class 0
            class0 = df[df['class'] == 3].drop(['class'], axis=1)
            X = class0
            # --------------------------------Variance Threshold--------------------------------------------------------------
            # selects important features using variance threshold
            # https://scikit-learn.org/stable/modules/feature_selection.html#removing-features-with-low-variance
            # sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
            # fitted_X = sel.fit_transform(X)
            # imp_feat = pd.DataFrame(fitted_X)
            # imp_feat = imp_feat.drop(imp_feat[imp_feat.sum(axis=1) == 0.0].index)
            #------------------------------------------------------------------------------------------------------------------
            #counting 1s----------------------------------------------------------------------------------------------------

            col_sum = X.sum(axis=0)
            sorted_colsum = sorted(col_sum, reverse=True)
            index_list = []
            for i in sorted_colsum[:8]:
                index_list.append(list(col_sum).index(i))
            imp_feat = X[index_list]
            # # ------------------------------------------------------------------------------------------------------------------
            # #Paper----------------------------------------------------
            # paper = data['paper'].x.tolist()
            # df_paper = pd.DataFrame(paper)
            #
            # X = df_paper
            # col_sum = X.sum(axis=0)
            # sorted_colsum = sorted(col_sum, reverse=True)
            # index_list = []
            # for i in sorted_colsum[:5]:
            #     index_list.append(list(col_sum).index(i))
            # imp_feat = X[index_list]
            # #Paper----------------------------------------------------
            #Term-------------------------------------------------------
            # term_df = pd.DataFrame(data['term'].x.numpy())
            # corr = term_df.corr()
            # m = ~(corr.mask(np.eye(len(corr), dtype=bool)).abs() > 0.19).any()
            # real = corr.loc[m]
            # list_index = np.ravel(real.index).tolist()
            # imp_feat_cont = term_df[list_index]
            # imp_feat = apply_threshold(imp_feat_cont)
            # Term-------------------------------------------------------
        elif self.dataset_name == 'imdb':
            dataset = IMDB(root='./imdb_data')
            data = dataset[0]
            movie = data['movie'].x.tolist()
            df = pd.DataFrame(movie)
            df['class'] = data['movie'].y.tolist()
            class0 = df[df['class'] == 2].drop(['class'], axis=1)
            # -----------------------------------count 1s logic--------------------------------------------------------------
            #selects top k features with most number of 1s
            col_sum = class0.sum(axis=0)
            # sorted_colsum = sorted(col_sum, reverse=True)
            # index_list = []
            # for i in sorted_colsum[:10]:
            #     index_list.append(list(col_sum).index(i))
            colsum_df = pd.DataFrame(col_sum)

            sorted_colsum = sorted(col_sum, reverse=True)[:5]
            index_list = list(np.ravel(colsum_df[colsum_df[0].isin(sorted_colsum)].index))

            imp_feat = class0[index_list]
            # ------------------------------------------------------------------------------------------------------------------
            # X = class0
            # sel = VarianceThreshold(threshold=(.2 * (1 - .2)))
            # fitted_X = sel.fit_transform(X)
            # imp_feat = pd.DataFrame(fitted_X)
            # imp_feat = imp_feat.drop(imp_feat[imp_feat.sum(axis=1) == 0.0].index)
            #-----------------------------------------------------------------------------------------------------------------
            #Director-------------------------------------------------------
            # director_df = pd.DataFrame(data['director'].x.numpy())
            # corr = director_df.corr()
            # m = ~(corr.mask(np.eye(len(corr), dtype=bool)).abs() > 0.3).any()
            # real = corr.loc[m]
            # list_index = np.ravel(real.index).tolist()
            # imp_feat_cont= director_df[list_index]
            # imp_feat = apply_threshold(imp_feat_cont)
            # Director-------------------------------------------------------
            #Actor-------------------------------------------------------
            # actor_df = pd.DataFrame(data['actor'].x.numpy())
            # corr = actor_df.corr()
            # m = ~(corr.mask(np.eye(len(corr), dtype=bool)).abs() > 0.2).any()
            # real = corr.loc[m]
            # list_index = np.ravel(real.index).tolist()
            # imp_feat_cont= actor_df[list_index]
            # imp_feat = apply_threshold(imp_feat_cont)
            # Actor-------------------------------------------------------
        elif self.dataset_name == 'pubmed':
            df_nodes = pd.read_table(('/media/pallabee/New Volume/CS/Thesis2023/designing_solution/graph_gen1/node_pubmed.dat'),names=['node_id', 'node_name', 'node_type', 'node_attributes'],quoting=csv.QUOTE_NONE)
            df_labels_train = pd.read_table(('/media/pallabee/New Volume/CS/Thesis2023/designing_solution/graph_gen1/label_pubmed.dat'),names=['node_id', 'node_name', 'node_type', 'node_label'])
            df_labels_test = pd.read_table(('/media/pallabee/New Volume/CS/Thesis2023/designing_solution/graph_gen1/label_pubmed.dat.test'),
                                           names=['node_id', 'node_name', 'node_type', 'node_label'])
            df_labels = pd.concat([df_labels_train, df_labels_test], ignore_index=True)
            # ##Select the nodes of type 1 which are labeled
            df_disease = pd.merge(df_nodes, df_labels, on="node_id")[['node_id', 'node_attributes', 'node_label']]
            class0 = df_disease[df_disease['node_label'] == 7].drop(['node_label', 'node_id'], axis=1)
            disease_class0 = preprocess_class(class0)
            # # -----------------------------------feature selection using correlation -realistic features-------------------------------------------------------------
            # corr = disease_class0.corr()
            # m = ~(corr.mask(np.eye(len(corr), dtype=bool)).abs() > 0.6).any()
            # real0 = corr.loc[m]
            # list0_index = np.ravel(real0.index).tolist()
            # imp_feat_cont = disease_class0[list0_index]
            # imp_feat = apply_threshold(imp_feat_cont)
            # # ------------------------------------------------------------------------------------------------------------------
            # -----------------------------------feature selection using correlation - Disease - 10 features for GNN training-------------------------------------------------------------
            corr = disease_class0.corr()
            m = ~(corr.mask(np.eye(len(corr), dtype=bool)).abs() > 0.54).any()
            real0 = corr.loc[m]
            real_ind0 = real0.index[0]
            imp_feat0 = corr[real_ind0].sort_values().head(9)
            list0_index = np.ravel(imp_feat0.index).tolist()
            list0_index.append(real_ind0)
            #print(disease_class0[list0_index])
            imp_feat_cont = disease_class0[list0_index]
            imp_feat = apply_threshold(imp_feat_cont)
            # Gene------------------------------------------------------------------------------------------------------------------
            # df_gene = df_nodes[df_nodes['node_type'] == 0]
            # df_gene = preprocess_class(df_gene)
            # corr = df_gene.corr()
            # m = ~(corr.mask(np.eye(len(corr), dtype=bool)).abs() > 0.3).any() #0.25
            # real0 = corr.loc[m]
            # list0_index = np.ravel(real0.index).tolist()
            # imp_feat_cont = df_gene[list0_index]
            # imp_feat = apply_threshold(imp_feat_cont)
            #-------------------------------------------------------------------------------------------------------------------
            # Chemical------------------------------------------------------------------------------------------------------------------
            # df_chemical = df_nodes[df_nodes['node_type'] == 2]
            # df_chemical = preprocess_class(df_chemical)
            # corr = df_chemical.corr()
            # m = ~(corr.mask(np.eye(len(corr), dtype=bool)).abs() > 0.25).any() #0.2
            # real0 = corr.loc[m]
            # list0_index = np.ravel(real0.index).tolist()
            # imp_feat_cont = df_chemical[list0_index]
            # imp_feat = apply_threshold(imp_feat_cont)
            #-------------------------------------------------------------------------------------------------------------------
            # Species------------------------------------------------------------------------------------------------------------------
            # df_species = df_nodes[df_nodes['node_type'] == 3]
            # df_species = preprocess_class(df_species)
            # corr = df_species.corr()
            # m = ~(corr.mask(np.eye(len(corr), dtype=bool)).abs() > 0.38).any() #0.3
            # real0 = corr.loc[m]
            # list0_index = np.ravel(real0.index).tolist()
            # imp_feat_cont = df_species[list0_index]
            # imp_feat = apply_threshold(imp_feat_cont)
            #-------------------------------------------------------------------------------------------------------------------
            # X = class0
            # sel = VarianceThreshold(threshold=(.2 * (1 - .2)))
            # fitted_X = sel.fit_transform(X)
            # imp_feat = pd.DataFrame(fitted_X)
            # imp_feat = imp_feat.drop(imp_feat[imp_feat.sum(axis=1) == 0.0].index)


        imp_feat = imp_feat.drop(imp_feat[imp_feat.sum(axis=1)==0.0].index)
            # Using train/test/val -80/10/10
            # https://stackoverflow.com/questions/38250710/how-to-split-data-into-3-sets-train-validation-and-test
        train, val, test = np.split(imp_feat.sample(frac=1, random_state=42),
                                        [int(.8 * len(imp_feat)), int(.9 * len(imp_feat))])

        #print('train',val.to_string())

        train_data.append(torch.tensor(train.values))
        val_data.append(torch.tensor(val.values))
        test_data.append(torch.tensor(test.values))

        torch.save(train_data, self.raw_paths[0])
        torch.save(val_data, self.raw_paths[1])
        torch.save(test_data, self.raw_paths[2])



    def process(self):
        file_idx = {'train': 0, 'val': 1, 'test': 2}
        raw_dataset = torch.load(self.raw_paths[file_idx[self.split]])

        data_list = []
        for d in raw_dataset:

            node_feature = torch.tensor(d)
            data = torch_geometric.data.Data(feature=node_feature)

            #data_list.append(data)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)
        torch.save(self.collate(data_list), self.processed_paths[0])


class FeatureDataModule(AbstractDataModule):
    def __init__(self, cfg):
        self.cfg = cfg
        self.datadir = cfg.dataset.datadir
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[0]
        root_path = os.path.join(base_path, self.datadir)


        datasets = {'train': FeatureDataset(dataset_name=self.cfg.dataset.name,
                                                 split='train', root=root_path),
                    'val': FeatureDataset(dataset_name=self.cfg.dataset.name,
                                        split='val', root=root_path),
                    'test': FeatureDataset(dataset_name=self.cfg.dataset.name,
                                        split='test', root=root_path)}
        # print(f'Dataset sizes: train {train_len}, val {val_len}, test {test_len}')

        super().__init__(cfg, datasets)
        self.inner = self.train_dataset

    def __getitem__(self, item):
        return self.inner[item]


class FeatureDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule):
        self.datamodule = datamodule
        self.feature_types = self.datamodule.feature_counts()


