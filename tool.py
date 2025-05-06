import os
import pandas as pd
import numpy as np
from rdkit import Chem
import matplotlib.pyplot as plt
from calprop import cal_prop

allowable_features = {
    'possible_atomic_num_list': list(range(1, 119)),
    'possible_chirality_list': [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_bonds' : [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs' : [ # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]
}



def prepareFolder():
    # exp_folder = 'testTransformer/'
    exp_folder = 'Ablution/'
    data_folder ='pre_train/'
    logs_folder = exp_folder + data_folder +'logs/'
    model_folder = exp_folder  + data_folder + 'model/'
    vis_folder = exp_folder  + data_folder + 'vis/'
    result_folder = exp_folder  + data_folder + 'metric_result/'

    if not os.path.isdir(exp_folder):
        os.mkdir(exp_folder)
    if not os.path.isdir(data_folder ):
        os.mkdir(data_folder )
    if not os.path.isdir(logs_folder):
        os.mkdir(logs_folder)
    if not os.path.isdir(model_folder):
        os.mkdir(model_folder)
    if not os.path.isdir(vis_folder):
        os.mkdir(vis_folder)
    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)
    # shutil.copyfile('../exp/model/model0.json',model_folder + 'model0.json')
    # shutil.copyfile('../exp/model/model0.h5',model_folder + 'model0.h5')
    #logger.add(logs_folder + "{time}.log")

    return logs_folder, model_folder, vis_folder,result_folder


def trainingVis(df1,df2, epoch, path):
    # make a figure
    fig = plt.figure(figsize=(16, 9))

    # subplot loss
    ax3 = fig.add_subplot(121)
    ax3.plot(df1['train total_loss'].tolist(), label='train_loss')
    ax3.plot(df2['test total_loss'].tolist(), label='test_loss')
    # ax3.plot(df['testing loss'].tolist(), label='test_loss')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Loss')
    ax3.set_title('Total Loss')
    ax3.legend()

    # subplot acc
    ax3.plot(df1['train recon_loss'].tolist(), label='train_recon_loss')
    ax3.plot(df2['test recon_loss'].tolist(), label='test_recon_loss')
    # ax3.plot(df['testing loss'].tolist(), label='test_loss')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Loss')
    ax3.set_title('Recon Loss')
    ax3.legend()

    plt.tight_layout()
    plt.title('Loss vs. Epoches', fontsize=20)
    plt.savefig(path + str(epoch)  + '.png')
    plt.close()



def graph_to_smiles(data_x, data_edge_index, data_edge_attr):

    mol = Chem.MolFromSmiles('')
    mol = Chem.RWMol()
        # rdmolops = Chem.rdmolops.RDKitOps(gen_mol)
    atom_features = data_x
    num_atoms = atom_features.shape[0]
    for i in range(num_atoms):
        atomic_num_idx = atom_features[i]
            # print('atomic_num_idx',atomic_num_idx )
        atomic_num = allowable_features['possible_atomic_num_list'][int(atomic_num_idx)]
            # print('atomic_num',atomic_num)
        atom = Chem.Atom(atomic_num)
        mol.AddAtom(atom)

    edge_index = np.array(data_edge_index)
    num_bonds = edge_index.shape[1]
    edge_attr = data_edge_attr

    for j in range(0, num_bonds, 2):
        begin_idx = int(edge_index[0, j])
        end_idx = int(edge_index[1, j])
        bond_type_idx = edge_attr[j]
            # print('bond_type_idx', bond_type_idx)
        bond_type = allowable_features['possible_bonds'][bond_type_idx]
            # print('bond_type', bond_type)

        mol.AddBond(int(begin_idx), int(end_idx), bond_type)
            # set bond direction

    return mol, Chem.MolToSmiles(mol)

# 序列唯一性
def canonic_smiles(smiles):
    mol = get_mol(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)

def get_mol(smiles):
    '''
    Loads SMILES/molecule into RDKit's object
    '''
    if isinstance(smiles, str):
        if len(smiles) == 0:
            return None
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
        return mol
    return smiles


def save_prop(file_path,result_path):
    w, l, q, s, n,lip = [], [], [], [], [],[]
    df = pd.read_csv(file_path)
    smiles = df['Smiles']
    c=0
    for smi in smiles:
        c+=1
        #print(c,smi)
        MolWt, mollop, molQED, molSA, molNP,lip5 = cal_prop(smi)
        # print(MolWt, MolLogP, molQED, SA,NP)
        w.append(MolWt)
        l.append(mollop)
        q.append(molQED)
        s.append(molSA)
        n.append(molNP)
        lip.append(lip5)
    df['mw'] = w
    df['lop'] = l
    df['qed'] = q
    df['sa'] = s
    df['np'] = n
    df['lipinski'] = lip
    df.to_csv(result_path,index=False)
    return df


