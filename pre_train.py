import argparse
import os
import os.path
import time
from time import gmtime
from time import strftime

import pandas as pd
import torch
import torch.optim as optim
from loguru import logger
from rdkit import Chem
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from rdkit.Chem.Crippen import MolLogP, MolMR
from rdkit.Chem import QED
from metrics.SA_Score import sascorer
from rdkit.Chem.rdMolDescriptors import CalcExactMolWt

from Dataset import CPDataset
#from model_2 import VGAE_ATT
from model_abl import VGAE_ATT
from tool import prepareFolder, graph_to_smiles, canonic_smiles,trainingVis
from train_test import train, test

parser = argparse.ArgumentParser(description="Run model")
parser.add_argument("--dataset", type=str, default='davis'
                                                   '', help="Dataset Name")
parser.add_argument('--compound_dim', type=int, default=1, help=' input dimention of compound')
parser.add_argument('--prot_dim', type=int, default=512, help='input dimention of protein')
parser.add_argument('--latent_dim', type=int, default=512, help='dimention of model')
parser.add_argument('--hidden_dim', type=int, default=256, help=' hidden dimention of decoder')
parser.add_argument('-lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--dropout_1', type=int, default=0.1, help='dropout for DTPGCN')
parser.add_argument('--dropout_2', type=int, default=0.2, help='dropout for CPropATT')
parser.add_argument('--num_epoch', type=int, default=50, help='epochs')
parser.add_argument('--batch_size', type=int, default=30, help='bs')
#parser.add_argument('--output', type=str, default='results/generated', help="output files path (default='../results/generated')")
args = parser.parse_args()


def pretrain(args):

    # 加载数据
    print('Begining running on _ ', args.dataset)
    processed_data_file_train = 'data/processed/' + args.dataset + '_train.pt'
    processed_data_file_test = 'data/processed/' + args.dataset + '_test.pt'
    if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
        print('please run create_data.py to prepare data in pytorch format!')
    #else:
    train_dataset = CPDataset(root='data', dataset=args.dataset + '_train')
    test_dataset = CPDataset(root='data', dataset=args.dataset + '_test')

    print('train_data', train_dataset[0])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #  define model
    vgae = VGAE_ATT(args.prot_dim,args.latent_dim,args.hidden_dim, args.dropout_1,args.dropout_2,device)
    vgae = vgae.to(device)

    optimizer_vgae = optim.Adam(vgae.parameters(), lr=args.lr)

    start = time.time()

    logs_folder, model_folder, vis_folder, result_folder = prepareFolder()
    logger.success(args)

    propertys = ['total_loss', 'kl_loss','recon_loss' ]
    prefixs_1 = ['train']
    prefixs_2 = ['test']
    columns_1 = [' '.join([pre, pro]) for pre in prefixs_1 for pro in propertys]
    columns_2 = [' '.join([pre, pro]) for pre in prefixs_2 for pro in propertys]
    logdf_1 = pd.DataFrame({}, columns=columns_1)
    logdf_2 = pd.DataFrame({}, columns=columns_2)
    # 开始训练
    for epoch in range(args.num_epoch):
        tqdm.write("Epoch= {:3d}".format(epoch))
        d1 = train(vgae, train_loader, optimizer_vgae, epoch, device)

        tqdm.write(
            f'Epoch [{epoch}/{args.num_epoch}]: Train loss: {d1[0]:.4f}, Train-kl loss: {d1[1]:.4f}, Test-recon loss: {d1[2]:.4f}')
            #f' Train-recon loss: {d1[2]:.4f},, Train-dis loss: {d1[3]:.4f},Train-cd loss: {d1[4]:.4f}')

        d2 = test(vgae, test_loader, epoch, device)

        tqdm.write(
            f'Epoch [{epoch}/{args.num_epoch}]: Test loss: {d2[0]:.4f},Test-kl loss: {d2[1]:.4f},Test-recon loss: {d2[2]:.4f}')
            #f' Test-recon loss: {d2[2]:.4f},, Test-dis loss: {d2[3]:.4f},Test-cd loss: {d2[4]:.4f}')

        logdf_1 = pd.concat([logdf_1, pd.DataFrame([d1], columns=columns_1)], ignore_index=True)
        logdf_2 = pd.concat([logdf_2, pd.DataFrame([d2], columns=columns_2)], ignore_index=True)
        logdf_1.to_csv(logs_folder + args.dataset+ '_no_tar_train_loss')
        logdf_2.to_csv(logs_folder + args.dataset+ '_no_tar_test_loss')

    # 保存模型
    torch.save(vgae.state_dict(), model_folder + args.dataset + '_no_tar_vgae.pt')   #最后一次的模型保存
    tqdm.write('Saving vgae model with loss {:.3f}...'.format(d1[0]))

    trainingVis(logdf_1, logdf_2, epoch, vis_folder+'no_tar')


    end = time.time()
    time_spent = strftime("%H:%M:%S", gmtime(end - start))
    print("train time spent {time}".format(time=time_spent))



if __name__ == "__main__":
    pretrain(args)

