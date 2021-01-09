import torch
import torch.nn.functional as F
import skimage.restoration as skr

def loss_function(output, truth, mode="loglike", sigma=25, device="cpu"):
    if(mode == "mse"):
        loss = F.mse_loss(output, truth, reduction="sum") / (truth.size(0) * 2)
    elif(mode == "loglike"):
        eps = 1e-5
        N,C,H,W = truth.shape
        mean = output[0:N, 0:C, 0:H, 0:W].permute(0,2,3,1).reshape(N, H, W, C, 1)
        var = output[0:N, C:C+int(C*(C+1)/2), 0:H, 0:W].permute(0,2,3,1)
        truth = truth.permute(0,2,3,1).reshape(N, H, W, C, 1)
        ax = torch.zeros(N, H, W, int(C*C)).to(device)
        I = torch.eye(C).reshape(1,1,1,C,C).repeat(N, H, W, 1, 1).to(device)
        idx1 = 0
        for i in range(C):
            idx2 = idx1 + C-i
            ax[0:N, 0:H, 0:W, int(i*C):int(i*C)+C-i] = var[0:N, 0:H, 0:W, idx1:idx2]
            idx1 = idx2
        ax = ax.reshape(N, H, W, C, C)
        sigma2I = (((sigma**2)+eps)*I.permute(1,2,3,4,0)).permute(4,0,1,2,3)
        variance = torch.matmul(ax.transpose(3,4), ax) + sigma2I #(sigma**2)*I
        likelihood = 0.5*torch.matmul(torch.matmul((truth-mean).transpose(3,4), torch.inverse(variance)), (truth-mean))
        likelihood = likelihood.reshape(N,H,W)
        likelihood += 0.5*torch.log(torch.det(variance))
#         loss = torch.mean(likelihood)
        loss = torch.mean(likelihood.mean(dim=(1,2)) - 0.1*sigma) 
    return loss
