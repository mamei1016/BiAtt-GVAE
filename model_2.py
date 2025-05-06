import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from encoder_decoder_2 import CPropATT,BiATT,DTPGCN
from CPP_Discri import C_Discriminator, Discriminator

# 定义VGAE_ATT模型: 编码过程中融入靶点特征和性质

class VGAE_ATT(nn.Module):
    def __init__(self, prot_dim,latent_dim,hidden_dim, dropout_1,dropout_2,device):
        super(VGAE_ATT, self).__init__()

        # 定义GCN层
        self.x_input_dim = 2
        self.protein_dim = prot_dim
        self.latent_dim = latent_dim
        self.hidden_dim=hidden_dim
        self.dropout_1 = dropout_1
        self.dropout_2 = dropout_2
        self.layer_cnn = 3
        self.window = 5
        # encoder part
        self.DTPGCN= DTPGCN(self.x_input_dim, self.protein_dim, self.latent_dim, self.dropout_1,device)
        self.BiATT= BiATT(self.latent_dim)
        self.CPropATT =CPropATT(self.latent_dim,self.dropout_2,device)
        self.fc_3 = torch.nn.Linear(self.latent_dim,self.latent_dim)  # 线性化到同一维度

        # 定义重参数化层
        self.fc_mean = nn.Linear(latent_dim, latent_dim)
        self.fc_logstd = nn.Linear(latent_dim, latent_dim)

        # decoder part
        self.conv1 = GCNConv(self.latent_dim,self.hidden_dim)
        self.conv2 = GCNConv(self.hidden_dim, self.x_input_dim)
        #self.mlp = nn.Linear(1, self.latent_dim)


        #self.D = Discriminator(self.latent_dim)
        #self.c_D = C_Discriminator(self.latent_dim)

    def encode(self, data):
        x_drug,  x_target,x_prop = self.DTPGCN(data)
        x_drug_t, x_target = self.BiATT(x_drug, x_target)
        x_drug_p = self.CPropATT(x_drug, x_prop)
        x_final = x_drug_t+x_drug_p
        x_final = self.fc_3(x_final)
        mean = self.fc_mean(x_final)
        logstd = self.fc_logstd(x_final)

        return mean,logstd, x_final, x_target,x_prop

    def reparameterize(self, mean, logstd):
        std = torch.exp(0.5 * logstd)
        epsilon = torch.randn_like(std)
        return mean + epsilon * std

    def decode(self, z,prop, data):
        #prop= F.relu(self.mlp(prop.transpose(0, 1)))
        #z_new = torch.cat([z,prop],1)
        #z_new = self.CPropATT(z, prop)
        x = F.relu(self.conv1(z, data.smile_edge_index))
        #x = F.sigmoid(self.conv2(x, data.smile_edge_index))
        x = self.conv2(x, data.smile_edge_index)

        return x

    def forward(self, data):

        prop = data.props
        #prop = prop.expand(data.x_smile.shape[0], 5)

        mean, logstd, x_drug, x_target,x_prop= self.encode(data)
        z = self.reparameterize(mean, logstd)
        x_recon = self.decode(z,prop, data)

        #real_pred = self.D(z)
        #fake_latent = torch.randn(z.shape[0],z.shape[1])
        #fake_latent = torch.ones_like(z)
        #fake_pred = self.D(fake_latent)

        #y_pred = self.c_D(z, data.target)

        #print('mean ', mean.size(), )   Nv*latent
        #print('logstd ', logstd.size())  Nv*latent
        #print('x_drug ', x_drug.size(), )  Nv*latent
        #print('x_target ', x_target.size(), )  Nr*latent
        #print('x_prop ', x_prop.size())    5*latent
        #print('z ', z.size() )     Nv*latent
        # print('x_recon ', x_recon.size())    Nv*2

        #return x_recon, mean, logstd, real_pred,fake_pred, y_pred

        return x_recon, mean, logstd




