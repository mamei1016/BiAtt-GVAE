import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# 编码器中的组件

class ResidualBlock(torch.nn.Module):
    def __init__(self, outfeature):
        super(ResidualBlock, self).__init__()
        self.outfeature = outfeature
        self.gcn = GCNConv(outfeature,outfeature)
        self.ln = torch.nn.Linear(outfeature, outfeature, bias=False)
        self.relu = nn.ReLU()


    def forward(self, x, edge_index):
        identity = x
        out = self.gcn(x, edge_index)
        out = self.relu(out)
        out = self.ln(out)
        out += identity
        out = self.relu(out)
        return out

# GCN based model--用于compound和protein 特征学习
class DTPGCN(torch.nn.Module):
    def __init__(self, num_features_xd,prot_dim,latent_dim,  dropout,  device):
        super(DTPGCN, self).__init__()
        self.prot_dim = prot_dim
        self.latent_dim = latent_dim
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.device = device
        self.window = 5
        #self.num_rblock = 2
        self.layer_cnn = 2
        self.alpha = 0.1
        self.num_features_xt= 21

        # SMILES graph branch
        self.conv1_xd = GCNConv(num_features_xd, self.latent_dim//8)
        self.conv2_xd = GCNConv(self.latent_dim//8, self.latent_dim//4)
        #self.rblock_xd = ResidualBlock(num_features_xd*2)
        #self.fc_g1_d = torch.nn.Linear(num_features_xd*2,  self.latent_dim)
        self.fc_g2_d = torch.nn.Linear(self.latent_dim//4, self.latent_dim)    #512

        # protein graph branch
        #self.fc_g1_t = torch.nn.Linear(self.num_features_xt, self.latent_dim)
        #self.fc_g2_t = torch.nn.Linear(self.latent_dim, self.latent_dim)

        #self.embedding_layer_amino = nn.Embedding(1, self.prot_dim)
        self.conv_layers = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2 * self.window + 1,
                                                    stride=1, padding=self.window) for _ in range(self.layer_cnn)])
        self.W_prot = nn.Linear(self.num_features_xt, latent_dim)

        # propety embedding，5个性质 ExactMolWt(m), MolLogP(m),QED.qed(m),SAscore,NP
        self.mlp = nn.Linear(1, self.latent_dim)


    def forward(self, data,):
        x, edge_index, batch = data.x_smile, data.smile_edge_index, data.batch
        #x =x .to(torch.long)
        #edge_index = edge_index.to(torch.long)
        #print('x',x.size())
        x_target = data.target
        prop = data.props
        #print('seq_size', seq_size)
        #print('x_target', x_target.size())
        # drug branch
        x = self.conv1_xd(x, edge_index)
        x = self.relu(x)
        x = self.conv2_xd(x, edge_index)
        x = self.relu(x)
        #for i in range(self.num_rblock):
            #x = self.rblock_xd(x, edge_index)
        # flatten
        #x = self.relu(self.fc_g1_d(x))
        #x = self.dropout(x)
        x_drug = self.fc_g2_d(x)
        #print('encode x_drug', x_drug.size())

        # protein feed forward

        #xt = self.conv1_xt(x_target, t_edge_index)
        #xt = self.relu(xt)
        #xt = self.conv2_xt(xt, t_edge_index,)
        #xt = self.relu(xt)
        #for i in range(self.num_rblock):
            #xt = self.rblock_xt(xt, t_edge_index, )
        # flatten
        #xt = self.relu(self.fc_g1_t(x_target))
        #xt = self.dropout(xt)
        #amino_vector = self.fc_g2_t(xt)  # residue

        #amino_vector = self.embedding_layer_amino(x_target.long())
        #print(' a_m', amino_vector.size())

        amino_vector = torch.unsqueeze(x_target.long(), 0)
        #print(' a_m', amino_vector.size())
        amino_vector = torch.unsqueeze(amino_vector.long(), 0)
        #print(' a_1', amino_vector.size())
        for i in range(self.layer_cnn):
            amino_vector = F.leaky_relu(self.conv_layers[i](amino_vector.float()), self.alpha)
            #print('a_2', amino_vector.size())
        amino_vector = torch.squeeze(amino_vector, 0)
        amino_vector = torch.squeeze(amino_vector, 0)
        #print('a_3', amino_vector.size())

        amino_vector = F.leaky_relu(self.W_prot(amino_vector), self.alpha)

        #print('encode amino_vector', amino_vector.size())

        # 构建性质向量

        #print('prop[0]', prop[0][0])  # 取出一个性质
        prop_1 = self.mlp(torch.tensor([prop[0][0]])).to(self.device)
        #print('prop_1', prop_1.size(), prop_1)
        prop_2 = self.mlp(torch.tensor([prop[0][1]])).to(self.device)
        prop_3 = self.mlp(torch.tensor([prop[0][2]])).to(self.device)
        prop_4 = self.mlp(torch.tensor([prop[0][3]])).to(self.device)
        #prop_5 = self.mlp(torch.tensor([prop[0][4]]))
        #prop_6 = self.mlp(torch.tensor([prop[0][5]]))  # 1*latent_dim的张量
        #prop_vector = torch.cat([prop_1, prop_2, prop_3, prop_4, prop_5, prop_6],0)  # 6*latent_dim的张量
        prop_vector = torch.stack([prop_1, prop_2, prop_3, prop_4], 0)  # 4*latent_dim的张量   ,用于交叉注意力的运算，用了四个性质，也可以是5个
        #prop_vector = self.relu(self.mlp(prop_vector.transpose(0, 1)))  # 5*latent_dim的张量

        #print('prop_vector', prop_vector.size())

        return x_drug,  amino_vector, prop_vector
# 原子-残基图中的交叉注意力学习
class BiATT(nn.Module):
    def __init__(self, latent_dim):
        super(BiATT, self).__init__()

        self.latent_dim =latent_dim
        self.bidat_num = 4   # 多头注意力个数

        self.U = nn.ParameterList([nn.Parameter(torch.empty(size=(self.latent_dim, self.latent_dim))) for _ in range(self.bidat_num)])
        for i in range(self.bidat_num):
            nn.init.xavier_uniform_(self.U[i], gain=1.414)

        self.transform_c2p = nn.ModuleList([nn.Linear(self.latent_dim, self.latent_dim) for _ in range(self.bidat_num)])
        self.transform_p2c = nn.ModuleList([nn.Linear(self.latent_dim, self.latent_dim) for _ in range(self.bidat_num)])

        self.bihidden_c = nn.ModuleList([nn.Linear(self.latent_dim, self.latent_dim) for _ in range(self.bidat_num)])
        self.bihidden_p = nn.ModuleList([nn.Linear(self.latent_dim, self.latent_dim) for _ in range(self.bidat_num)])
        self.biatt_c = nn.ModuleList([nn.Linear(self.latent_dim * 2, 1) for _ in range(self.bidat_num)])
        self.biatt_p = nn.ModuleList([nn.Linear(self.latent_dim * 2, 1) for _ in range(self.bidat_num)])
        self.comb_c = nn.Linear(self.latent_dim * self.bidat_num, self.latent_dim)
        self.comb_p = nn.Linear(self.latent_dim * self.bidat_num, self.latent_dim)
        self.saved_atom_att = None
        self.saved_amino_att = None

    def forward(self, atoms_vector, amino_vector):
        atom_atts = []
        amino_atts = []
        for i in range(self.bidat_num):
            A = torch.tanh(torch.matmul(torch.matmul(atoms_vector, self.U[i]), amino_vector.transpose(0, 1)))#data的批量数据时应该为（1，2）
            #print("A",A.size())
            atoms_trans = torch.matmul(A, torch.tanh(self.transform_p2c[i](amino_vector)))
            amino_trans = torch.matmul(A.transpose(0, 1), torch.tanh(self.transform_c2p[i](atoms_vector)))

            atoms_tmp = (torch.cat([torch.tanh(self.bihidden_c[i](atoms_vector)), atoms_trans], dim=1))
            amino_tmp = (torch.cat([torch.tanh(self.bihidden_p[i](amino_vector)), amino_trans], dim=1))

            atoms_att = F.softmax(self.biatt_c[i](atoms_tmp),dim=1)  # Nv*1
            amino_att = F.softmax(self.biatt_p[i](amino_tmp),dim=1)  # Nr*1
            #print('atoms_att',atoms_att.size(),'\n','amino_att',amino_att.size())

            # 保存每个头的注意力权重
            atom_atts.append(atoms_att.detach().cpu())
            amino_atts.append(amino_att.detach().cpu())

            cf = atoms_vector * atoms_att
            pf = amino_vector * amino_att
            #print('cf ', cf .size(),'\n','pf ', pf .size() )


            if i == 0:
                cat_cf = cf
                cat_pf = pf
            else:
                cat_cf = torch.cat([cat_cf, cf], dim=1)  # Nv-(latent*bidat_num)
                cat_pf = torch.cat([cat_pf, pf], dim=1)   # Nr-(latent*bidat_num)
            #print('cat_cf ', cat_cf.size(), '\n', 'cat_pf ', cat_pf.size())

        # 转换为Tensor [heads, atoms, 1]
        self.saved_atom_att = torch.stack(atom_atts)
        self.saved_amino_att = torch.stack(amino_atts)

        cf_final = self.comb_c(cat_cf)  #Nv*latent,学习后原子特征
        pf_final = self.comb_p(cat_pf)  #Nr*latent
        #print('cf_final ', cf_final.size(), '\n', 'pf_final  ', pf_final.size())


        return cf_final, pf_final

class self_attention(torch.nn.Module):
    def __init__(self, input_dim,input_2,n_head,latent_dim ):

        super(self_attention, self).__init__()
        self.input_dim_1 = input_dim
        self.input_dim_2 = input_2
        self.n_heads = n_head
        self.latent_dim = latent_dim
        self.d_k = self.input_dim_1 // self.n_heads
        self.d_v = self.input_dim_2 // self.n_heads
        self.W_Q = torch.nn.Linear(self.input_dim_1, self.d_k * self.n_heads, bias=False)
        self.W_K = torch.nn.Linear(self.input_dim_2, self.d_k * self.n_heads, bias=False)
        self.W_V = torch.nn.Linear(self.input_dim_2, self.d_v * self.n_heads, bias=False)
        self.fc = torch.nn.Linear(self.n_heads * self.d_v, self.input_dim_2, bias=False)
        self.AN1 = torch.nn.LayerNorm(self.input_dim_2 )
        self.l1 = torch.nn.Linear(self.input_dim_2 , self.latent_dim)
        self.saved_attn = None  # 新增保存变量

    def forward(self, X1,X2,x2):

        Q = self.W_Q(X1).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        #print('Q',Q.size())
        K = self.W_K(X2).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        #print('K', K.size())
        V = self.W_V(X2).view(-1, self.n_heads, self.d_v).transpose(0, 1)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        # context: [n_heads, len_q, d_v], attn: [n_heads, len_q, len_k]
        attn = torch.nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        # context: [len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(-1, self.n_heads * self.d_v)
        output = self.fc(context)
        output = self.AN1(output)
        output =self.l1(output )
        self.saved_attn = F.softmax(scores, dim=-1).detach().cpu()  # 保存注意力权重
        return output

# 化合物原子和性质矩阵的子注意力学习
class CPropATT(nn.Module):
    """ compound feature extraction."""
    def __init__(self, latent_dim, dropout, device):
        super(CPropATT,self).__init__()
        self.ln = nn.LayerNorm(latent_dim)
        self.latent_dim = latent_dim
        self.n_layers = 3
        self.n_heads = 12
        self.dropout = dropout
        self.device = device
        self.do = nn.Dropout(dropout)
        self.fc_1 = nn.Linear(self.latent_dim, self.latent_dim)
        self.sa = self_attention(self.latent_dim, self.latent_dim,self.n_heads, self.latent_dim)
        self.saved_attentions = []  # 新增保存列表

    def forward(self, atom_vector, prop_vector ):

        self.saved_attentions = []  # 每次前向传播前清空

        # trg = [batch size, compound len, hid dim]

        for _ in range(self.n_layers):
            # 执行注意力计算
            attn_output = self.sa(atom_vector, prop_vector, prop_vector)

            # 保存当前层的注意力权重
            layer_attn = self.sa.saved_attn  # 从self_attention实例获取
            self.saved_attentions.append(layer_attn)

            atom_vector = self.ln(atom_vector + self.do(self.sa(atom_vector, prop_vector, prop_vector)))
            #atom_vector = self.fc_1(atom_vector + (self_attention(atom_vector, prop_vector, prop_vector)))
        # trg = [batch size,hid_dim]
        atom_vector = self.fc_1(atom_vector)
        return atom_vector


