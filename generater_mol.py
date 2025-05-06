import argparse
import os
from loguru import logger
import torch
import time
from time import strftime
from time import gmtime
import os.path
import pandas as pd
import math
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.loader import  DataLoader
from rdkit import Chem
from rdkit.Chem.Crippen import MolLogP, MolMR
from rdkit.Chem import QED
from metrics.SA_Score import sascorer
from rdkit.Chem.rdMolDescriptors import CalcExactMolWt
from model_abl import VGAE_ATT
from Dataset import CPDataset
from tool import prepareFolder,graph_to_smiles,canonic_smiles





parser = argparse.ArgumentParser(description="Generater molecules")
parser.add_argument("--dataset", type=str, default='EGFR', help="Dataset Name")
parser.add_argument('--conditions', type=str, default='data/conditions_1.csv', help="conditions path (default='data/conditions.csv')")
parser.add_argument('--compound_dim', type=int, default=1, help=' input dimention of compound')
parser.add_argument('--prot_dim', type=int, default=512, help='input dimention of protein')
parser.add_argument('--latent_dim', type=int, default=512, help='dimention of model')
parser.add_argument('--hidden_dim', type=int, default=256, help=' hidden dimention of decoder')
parser.add_argument('-lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--dropout_1', type=int, default=0.1, help='dropout for DTPGCN')
parser.add_argument('--dropout_2', type=int, default=0.2, help='dropout for CPropATT')
parser.add_argument('--num_epoch', type=int, default=100, help='epochs')
parser.add_argument('--batch_size', type=int, default=30, help='bs')
parser.add_argument('--output', type=str, default='Ablution/pre_train/results/generated', help="output files path (default='../results/generated')")
args = parser.parse_args()


exp_folder = 'Ablution/'
data_folder ='pre_train/'
logs_folder = exp_folder + data_folder +'logs/'
model_folder = exp_folder  + data_folder + 'model/'
vis_folder = exp_folder  + data_folder + 'vis/'
result_folder = exp_folder  + data_folder + 'ge_result/'

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


def main(args):

    #生成分子.对于不同的靶点来做。首先提前构建数据集
    print('Generating molecules...')

    print('Load data.')

    processed_data_file_train = 'data/processed/'+ args.dataset +'_val.pt'

    val_dataset = CPDataset(root='data', dataset=args.dataset + '_val')
    print('val_data', val_dataset[0])

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #logs_folder, model_folder, vis_folder, result_folder = prepareFolder()
    logger.success(args)


    propertys = ['total_loss', 'vgae_loss', 'recon_loss']
    prefixs = ['val']
    columns = [' '.join([pre, pro]) for pre in prefixs for pro in propertys]
    logdf = pd.DataFrame({}, columns=columns)
    

    vgae = VGAE_ATT(args.prot_dim,args.latent_dim,args.hidden_dim, args.dropout_1,args.dropout_2,device)

    # 加载模型
    #vgae.load_state_dict(torch.load(model_folder + 'vgae.pt'))

    vgae.load_state_dict(torch.load('Ablution/pre_train/model/davis_no_prop_vgae.pt'))

    optimizer = torch.optim.Adam(vgae.parameters(), lr=args.lr)



    loss_vgae = []
    loss_reconstruction = []
    loss_kl = []

    for epoch in range(1, args.num_epoch + 1):
        loss_record = []
        for step, data in tqdm(enumerate(val_loader)):

            # Sample from DataLoader
            data = data.to(device)

            x_recon, mean, logstd = vgae(data)

            # Calculate loss

            reconstruction_loss = F.mse_loss(x_recon, data.x_smile, reduction='mean')

            kl_divergence = -0.5 * torch.sum(1 + 2 * logstd - mean.pow(2) - logstd.exp().pow(2))

            vgae_loss = reconstruction_loss + kl_divergence  # 重构损失


            # 更新VGAE和判别器的参数
            optimizer.zero_grad()
            vgae_loss.backward(retain_graph=True)
            optimizer.step()

            loss_vgae.append(vgae_loss.detach().item())
            loss_reconstruction.append(reconstruction_loss.detach().item())
            loss_kl.append(kl_divergence.detach().item())
            
           
            if step % 100 == 0 and step != 0:
                tqdm.write("*" * 50)
                tqdm.write(
                    "#####epoch={}, step {:3d}  total_loss: {:5.2f}, reconstruction_loss: {:5.2f}, kl Loss: {:5.2f}, \n".format(
                        epoch,step, vgae_loss.item(), reconstruction_loss.item(),kl_divergence.item()))
                #每100个数输出
           

        avg_vage_loss = sum(loss_vgae) / len(loss_vgae)
        avg_recon_loss = sum(loss_reconstruction) / len(loss_reconstruction)
        avg_kl_loss = sum(loss_kl) / len(loss_kl)

        d=[avg_vage_loss,avg_recon_loss,avg_kl_loss]

        logdf = pd.concat([logdf, pd.DataFrame([d], columns=columns)], ignore_index=True)
        logdf.to_csv(logs_folder + args.dataset + '_no_prop_val_loss')


        tqdm.write(f'Epoch [{epoch}/{args.num_epoch}]: val loss: {avg_vage_loss:.4f}, val-recon loss: {avg_recon_loss:.4f},val-kl loss: {avg_kl_loss:.4f}')

    # Save the model

    torch.save(vgae,model_folder +args.dataset+'_no_prop_vgae.pt')
    tqdm.write('Saving model with loss {:.3f}...'.format( avg_vage_loss))


    # 通过循环生成，随机生成分子
    print('Generate the molecules')

    #vgae = VGAE_ATT(args.prot_dim, args.latent_dim, args.hidden_dim, args.dropout_1, args.dropout_2, device)

    #vgae = torch.load('Ablution/pre_train/model/EGFR_no_tar_vgae.pt')

    #vgae = vgae.to(device)

    logp = []
    mw = []
    qed = []
    sa = []
    id = []
    smi = []
    canonic_smi = []


    for times in range(1, 20):
        for i, data in enumerate(tqdm(val_loader)):
           #
            #for time in range(1, 11):

            latent = torch.randn(data.x_smile.shape[0], args.latent_dim)
            latent = latent.to(device)

            sample = vgae.decode(latent, data)

            mol, smile = graph_to_smiles(sample, data.smile_edge_index, data.smile_edge_attr)

            if mol is None or smile is None:
                continue

            can_smi = canonic_smiles(smile)
            smi.append(Chem.MolToSmiles(mol))
            canonic_smi.append(can_smi)


     #print(f'{i}_can_smiles{len(canonic_smi)}')
        #c = len(canonic_smi)

    smile_df = pd.DataFrame({'SMILES': smi, 'Can_SMILES': canonic_smi})
    smile_df.to_csv(result_folder + args.dataset + f'_no_prop_gen_smiles_{times}.csv', index=False)
    print(f'Saving gen_smiles.csv ({smile_df.describe()})...')
    print(f'Saving gen_smiles.csv ({smile_df.shape[0]})...')





if __name__ == "__main__":
    main(args)




