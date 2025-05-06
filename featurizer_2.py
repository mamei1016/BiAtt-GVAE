import argparse
import pandas as pd
import numpy as np
import os
import torch
from rdkit import Chem
import networkx as nx
from Dataset_2 import CPDataset
from collections import defaultdict
from calprop import cal_prop

num_atom_feat = 2
word_dict = defaultdict(lambda: len(word_dict))

allowable_features = {
    'possible_atomic_num_list' : list(range(1, 119)),
    'possible_chirality_list' : [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_bonds' : [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC ]
}
# 图转换成SMILES
def graph_to_smiles(data_x,data_edge_index, data_edge_attr):   #放到真个模型里看怎么改变这个输入

    mol = Chem.MolFromSmiles('')
    mol = Chem.RWMol()
    #rdmolops = Chem.rdmolops.RDKitOps(gen_mol)
    atom_features = data_x
    num_atoms = atom_features.shape[0]
    for i in range(num_atoms):
        atomic_num_idx, chirality_tag_idx = atom_features[i]
        print('atomic_num_idx',atomic_num_idx )
        atomic_num = allowable_features['possible_atomic_num_list'][int(atomic_num_idx)]
        print('atomic_num',atomic_num)
        chirality_tag = allowable_features['possible_chirality_list'][int(chirality_tag_idx)]
        atom = Chem.Atom(atomic_num)
        atom.SetChiralTag(chirality_tag)
        mol.AddAtom(atom)

    edge_index = np.array(data_edge_index)
    num_bonds = edge_index.shape[1]
    edge_attr = data_edge_attr

    for j in range(0, num_bonds, 2):
        begin_idx = int(edge_index[0, j])
        end_idx = int(edge_index[1, j])
        bond_type_idx= edge_attr[j]
        print('bond_type_idx', bond_type_idx)
        bond_type = allowable_features['possible_bonds'][bond_type_idx]
        print('bond_type', bond_type)

        mol.AddBond(int(begin_idx), int(end_idx),bond_type )

    return mol, Chem.MolToSmiles(mol)

# 编码
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]


def smile_to_graph(smile):
    try:
        mol = Chem.MolFromSmiles(smile)
    except:
        raise RuntimeError("SMILES cannot been parsed!")


    atom_size = mol.GetNumAtoms()

    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = [allowable_features['possible_atomic_num_list'].index(
            atom.GetAtomicNum())] #+ [allowable_features['possible_chirality_list'].index(atom.GetChiralTag())]
        atom_features_list.append(atom_feature)
        # atom_feat[atom.GetIdx(), :] = atom_feature
    edges, edge_feature = [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_feature = allowable_features['possible_bonds'].index(
            bond.GetBondType())
        edges.append((i, j))
        edge_feature.append(bond_feature)
        edges.append((j, i))
        edge_feature.append(bond_feature)

    edge_index = torch.LongTensor(edges).transpose(-1, 0)
    edge_attr = torch.tensor(np.array(edge_feature), dtype=torch.long)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.float)

    prop=[]
    result = cal_prop(smile)
    prop.append(result)


    return atom_size, x, edge_index, edge_attr, prop

def aa_features(aa):
    results = one_of_k_encoding(aa,
                                ['A','N', 'C', 'Q', 'H', 'L', 'M', 'P', 'T', 'Y', 'R', 'D', 'E', 'G', 'I', 'K', 'F',
                                 'S', 'W', 'V', 'U'])
    return np.asarray(results, dtype=float)
'''
def target_to_graph(seq):
    seq_size = len(seq)
    eds_seq = []

    for i in range(seq_size - 1):
        eds_seq.append([i, i + 1])
    eds_seq = np.array(eds_seq)
    #print('eds_seq',eds_seq)
    # add an reserved extra node for drug node
    eds_d = []
    for i in range(seq_size):
        eds_d.append([i, seq_size])
    eds_d = np.array(eds_d)
    #print('eds_d',eds_d)
    eds = np.concatenate((eds_seq, eds_d))
    #print('eds', eds)
    edges = [tuple(i) for i in eds]
    g = nx.Graph(edges).to_directed()   # 图的边
    #print('g',g)
    features = []
    for i in range(seq_size):
        aa_feat = aa_features(seq[i])
        features.append(aa_feat)
    place_holder = np.zeros(features[0].shape, dtype=float)
    features.append(place_holder)  #点的特征

    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])




    return seq_size, features, edge_index
'''


def target_to_graph(seq):
    seq_size = len(seq)
    features = []
    for i in range(seq_size):
        aa_feat = aa_features(seq[i])
        features.append(aa_feat)
    place_holder = np.zeros(features[0].shape, dtype=float)
    features.append(place_holder)  #点的特征

    return seq_size, features

'''
def seq_cat(prot):

    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]):
        x[i] = seq_dict[ch]

    return x


seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v:(i+1) for i,v in enumerate(seq_voc)}
seq_dict_len = len(seq_dict)
max_seq_len = 1000
'''

def main(args):
    dataset = args.dataset

    compound_iso_smiles,target_sequence = [],[]
    opts = ['train', 'test']
    for opt in opts:
        df = pd.read_csv('data/' + dataset + '_' + opt + '.csv')
        compound_iso_smiles += list(df['compound_iso_smiles'])
        target_sequence += list(df['target_sequence'])
    compound_iso_smiles = set(compound_iso_smiles)
    target_sequence = set(target_sequence)
    smile_graph = {}
    target_graph = {}

    for smile in compound_iso_smiles:
        g = smile_to_graph(smile)
        smile_graph[smile] = g     #smile_graph包含的就是atom_size, atom_feat, edge_index, prop

    for seq in target_sequence :
        g = target_to_graph(seq)
        target_graph[seq] = g     #smile_graph包含的就是atom_size, atom_feat, edge_index, prop


    # convert to torch geometric data
    processed_data_file_train = 'data/processed/' + dataset + '_train.pt'
    processed_data_file_test = 'data/processed/' + dataset + '_test.pt'

    if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
        df = pd.read_csv('data/' + dataset + '_train.csv')
        train_drugs, train_prots, train_Y = list(df['compound_iso_smiles']), list(df['target_sequence']), list(
            df['affinity'])
        #XT = [seq_cat(t) for t in train_prots]
        train_drugs, train_prots, train_Y = np.asarray(train_drugs), np.asarray(train_prots), np.asarray(train_Y)
        df = pd.read_csv('data/' + dataset + '_test.csv')
        test_drugs, test_prots, test_Y = list(df['compound_iso_smiles']), list(df['target_sequence']), list(
            df['affinity'])
        #XT = [seq_cat(t) for t in test_prots]
        #XT = [split_sequence(t, ngram) for t in test_prots]

        test_drugs, test_prots, test_Y = np.asarray(test_drugs), np.asarray(test_prots), np.asarray(test_Y)

        # make data PyTorch Geometric ready
        print('preparing ', dataset + '_train.pt in pytorch format!')
        train_data = CPDataset(root='data', dataset=dataset + '_train', xd=train_drugs, xt=train_prots, y=train_Y,
                                    smile_graph=smile_graph,target_graph=target_graph)
        print('train_data', train_data,train_data[0])
        print('preparing ', dataset + '_test.pt in pytorch format!')
        test_data = CPDataset(root='data', dataset=dataset + '_test', xd=test_drugs, xt=test_prots, y=test_Y,
                                   smile_graph=smile_graph,target_graph=target_graph)
        print(processed_data_file_train, ' and ', processed_data_file_test, 'have been created')
        print('test_data',test_data,test_data[0].props)
        print('test_data', test_data, test_data[0])

    else:
        print(processed_data_file_train, ' and ', processed_data_file_test, ' are already created')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Creation of dataset")
    parser.add_argument("--dataset", type=str, default='davis',
                        help="Dataset Name (davis,EGFR,CDK2,MPro,BindingDB)")
    args = parser.parse_args()
    print('**********',args)
    main(args)


