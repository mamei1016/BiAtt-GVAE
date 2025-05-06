
import torch
from tqdm import tqdm
import torch.nn.functional as F



def train(vgae, train_loader, optimizer_vgae,epoch, device):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    vgae.train()
    loss_record = []
    loss_kl = []
    loss_reconstruction = []
    loss_dis = []
    loss_cd = []
    c = 0
    for step, data in enumerate(train_loader):
        #tqdm.write("step= {:3d}".format(step))
        data = data.to(device)

        x_recon, mean, logstd = vgae(data)
        #x_recon,pos_recon,real_pred, fake_pred,smiles_str,dec_all_graph,all_graph= vgae_dis(data)

        # 训练判别器
        #real_loss = F.binary_cross_entropy_with_logits(real_pred, torch.ones_like(real_pred))

        # Fake loss
        #fake_loss = F.binary_cross_entropy_with_logits(fake_pred, torch.zeros_like(fake_pred))

        #c_D_loss = F.mse_loss(y_pred, data.y, reduction='mean')

        # Discriminator total loss
        #discriminator_loss = real_loss + fake_loss + c_D_loss   # 对抗损失表示生成的既是分子，还必须与对应靶点有好的结合亲和力

        #### Train VgAE###
        reconstruction_loss = F.mse_loss(x_recon, data.x_smile,  reduction='mean')

        kl_divergence = -0.5 * torch.sum(1 + 2 * logstd - mean.pow(2) - logstd.exp().pow(2))

        vgae_loss = reconstruction_loss + kl_divergence  # 重构损失

        total_loss = vgae_loss #+ 0.1*discriminator_loss

        # 更新VGAE和判别器的参数
        optimizer_vgae.zero_grad()
        total_loss.backward(retain_graph=True)
        optimizer_vgae.step()

        loss_record.append(total_loss.detach().item())
        loss_kl.append(kl_divergence.detach().item())
        loss_reconstruction.append(reconstruction_loss.detach().item())
        #loss_dis.append(discriminator_loss.detach().item())
        #loss_cd.append(c_D_loss.detach().item())
        '''
        if step % 100 == 0 and step != 0:
            tqdm.write("*" * 50)
            tqdm.write(
                "#####epoch={}, step {:3d}  total_loss: {:5.2f}, reconstruction_loss: {:5.2f}, VGAE Loss: {:5.2f}, Discriminator Loss: {:5.2f},c-d Loss: {:5.2f} \n".format(
                    epoch,step, total_loss.item(), reconstruction_loss.item(), vgae_loss.item(), discriminator_loss.item(),c_D_loss.item()))
        '''
    avg_train_loss = sum(loss_record) / len(loss_record)
    avg_recon_loss = sum(loss_reconstruction) / len(loss_reconstruction)
    avg_kl_loss = sum(loss_kl) / len(loss_kl)
    #avg_dis_loss = sum(loss_dis) / len(loss_dis)
    #avg_cd_loss = sum(loss_cd) / len(loss_cd)


    return [avg_train_loss, avg_kl_loss ,avg_recon_loss ]#avg_dis_loss,avg_cd_loss]


@torch.no_grad()
def test(vgae, test_loader, epoch, device):
    vgae.eval()
    loss_kl = []
    loss_reconstruction = []
    loss_dis = []
    loss_cd=[]
    loss_record = []
    for step, data in enumerate(tqdm(test_loader)):

        x_recon, mean, logstd= vgae(data)
        #x_recon,pos_recon,real_pred, fake_pred,smiles_str,dec_all_graph,all_graph= vgae(data)


        #tqdm.write("*" * 50)
        #tqdm.write("####epoch={},step {:3d} \n data_x={} \n x_con={}".format(epoch,step,data.x_smile,x_recon))

        # 训练判别器
        #real_loss = F.binary_cross_entropy_with_logits(real_pred, torch.ones_like(real_pred))
        # Fake loss
        #fake_loss = F.binary_cross_entropy_with_logits(fake_pred, torch.zeros_like(fake_pred))

        # affinity score
        #c_D_loss =  F.mse_loss(y_pred, data.y,  reduction='mean')

        # Discriminator total loss
        #discriminator_loss = real_loss + fake_loss+c_D_loss   # 对抗损失
        # 训练VGAE
        #### Train VAE-Generator ###
        reconstruction_loss = F.mse_loss(x_recon, data.x_smile,  reduction='mean')
        kl_divergence = -0.5 * torch.sum(1 + 2 * logstd - mean.pow(2) - logstd.exp().pow(2))

        vgae_loss = reconstruction_loss + kl_divergence  # 重构损失

        total_loss = vgae_loss #+ 0.1 * discriminator_loss

        loss_record.append(total_loss.detach().item())
        loss_kl.append(kl_divergence.detach().item())
        loss_reconstruction.append(reconstruction_loss.detach().item())
        #loss_dis.append(discriminator_loss.detach().item())
        #loss_cd.append(c_D_loss.detach().item())

        '''
        if step % 100 == 0 and step != 0:
            tqdm.write("*" * 50)
            tqdm.write(
                "####epoch={},step {:3d}  total_loss: {:5.2f}, reconstruction_loss: {:5.2f}, VGAE Loss: {:5.2f}, Discriminator Loss: {:5.2f} \n".format(
                    epoch,step, total_loss.item(), reconstruction_loss.item(), vgae_loss.item(), discriminator_loss.item()))

        '''
    avg_test_loss = sum(loss_record) / len(loss_record)
    avg_recon_loss = sum(loss_reconstruction) / len(loss_reconstruction)
    avg_kl_loss = sum(loss_kl) / len(loss_kl)
    #avg_dis_loss = sum(loss_dis) / len(loss_dis)
    #avg_cd_loss = sum(loss_cd) / len(loss_cd)

    return [avg_test_loss,avg_kl_loss,avg_recon_loss]  # avg_dis_loss,avg_cd_loss]


