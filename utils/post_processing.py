import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import skimage.restoration as skr

def post_process(output, input, number=0, model="blind-spot-net" ,sigma=25, device="cpu"):
    if (model == "blind-spot-net"
        or model == "blind-video-net"
        or model == "blind-video-net-2"
        or model == "blind-video-net-d1"
        or model == "blind-video-net-d2"
        or model == "blind-spot-net-4"
        or model == "blind-video-net-4"
        or model == "blind-video-net-d1-4"
        or model == "blind-spot-net-2"):
        eps = 1e-5
        N,C,H,W = input.shape
        mean = output[0:N, 0:C, 0:H, 0:W].permute(0,2,3,1).reshape(N, H, W, C, 1)
        var = output[0:N, C:C+int(C*(C+1)/2), 0:H, 0:W].permute(0,2,3,1)
        input = input.permute(0,2,3,1).reshape(N, H, W, C, 1)
        ax = torch.zeros(N, H, W, int(C*C)).to(device)
        I = torch.eye(C).reshape(1,1,1,C,C).repeat(N, H, W, 1, 1).to(device)
        idx1 = 0
        for i in range(C):
            idx2 = idx1 + C-i
            ax[0:N, 0:H, 0:W, int(i*C):int(i*C)+C-i] = var[0:N, 0:H, 0:W, idx1:idx2]
            idx1 = idx2
        ax = ax.reshape(N, H, W, C, C)
        variance = torch.inverse(torch.matmul(ax.transpose(3,4), ax) + eps*I)
        Ibysigma2 = ((1/((sigma**2)+eps))*I.permute(1,2,3,4,0)).permute(4,0,1,2,3)
        inputbysigma2 = ((1/((sigma**2)+eps))*input.permute(1,2,3,4,0)).permute(4,0,1,2,3)
#         image = torch.matmul(torch.inverse(variance + (1/(sigma**2))*I),
#                              (torch.matmul(variance, mean) + (1/(sigma**2))*input)).reshape(N, H, W, C).permute(0,3,1,2)
        image = torch.matmul(torch.inverse(variance + Ibysigma2),
                             (torch.matmul(variance, mean) + inputbysigma2)).reshape(N, H, W, C).permute(0,3,1,2)
        input = input.reshape(N, H, W, C).permute(0,3,1,2)
        mean_image = output[0:N, 0:C, 0:H, 0:W]

        return image, mean_image
    else:
        return output, output
