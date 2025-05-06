import os
import numpy as np
from math import sqrt
from scipy import stats
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric import data as DATA
import torch

class CPDataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='EGFR',
                 xd=None, xt=None,  transform=None,
                 pre_transform=None,smile_graph=None,target_graph=None):

        #root is required for save preprocessed data, default is '/tmp'
        super(CPDataset, self).__init__(root, transform, pre_transform)
        # benchmark dataset, default = 'davis'
        self.dataset = dataset
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xd, xt, smile_graph,target_graph)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass
        #return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    # Customize the process method to fit the task of drug-target affinity prediction
    # Inputs:
    # XD - list of SMILES, XT: list of encoded target (categorical or one-hot),
    # Y: list of labels (i.e. affinity)
    # Return: PyTorch-Geometric format processed data
    def process(self, xd, xt,smile_graph,target_graph):
        #assert (len(xd) == len(xt) and len(xt) == len(y)), "The three lists must be the same length!"
        data_list = []
        data_len = len(xd)
        for i in range(data_len):
            print('Converting SMILES and Target to graph: {}/{}'.format(i+1, data_len))
            smiles = xd[i]
            target = xt[i]

            # convert SMILES to molecular representation using rdkit
            atom_size, x_features, x_edge_index, x_edge_attr,x_prop= smile_graph[smiles]
            #seq_size, t_features, t_edge_index= target_graph[target]
            seq_size, t_features = target_graph[target]
            # make the graph ready for PyTorch Geometrics GCN algorithms:

            GCNData = DATA.Data(num_nodes =torch.LongTensor([atom_size]),
                                x_smile=x_features,
                                smile_edge_index=x_edge_index,
                                smile_edge_attr = x_edge_attr,
                                #x_smile=torch.LongTensor(x_features,dtype=torch.long),
                                #smile_edge_index=torch.LongTensor(x_edge_index).transpose(1, 0),
                                #smile_edge_attr = torch.tensor(x_edge_attr, dtype=torch.long),
                                #y=torch.FloatTensor([labels])
                                )
            GCNData.target =torch.LongTensor(t_features)
            #GCNData.target_edge_index=torch.LongTensor(t_edge_index).transpose(1,0)
            GCNData.props = torch.tensor(x_prop, dtype=torch.float)
            #GCNData.__setitem__('atom_size', torch.LongTensor([atom_size]))
            GCNData.__setitem__('seq_size', torch.LongTensor([seq_size]))
            # append graph, label and target sequence to data list

            data_list.append(GCNData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        # save preprocessed data:
        torch.save((data, slices), self.processed_paths[0])
