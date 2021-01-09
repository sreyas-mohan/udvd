import torch
from torch.distributions import Poisson

def get_noise(data, dist='G', noise_std = float(25)/255.0, mode='S',
                    min_noise = float(5)/255., max_noise = float(55)/255.):
    if(dist == 'G'):
        noise_std /= 255.
        min_noise /= 255.
        max_noise /= 255.
        noise = torch.randn_like(data);
        if mode == 'B':
            n = noise.shape[0];
            noise_tensor_array = (max_noise - min_noise) * torch.rand(n) + min_noise;
            for i in range(n):
                noise.data[i] = noise.data[i] * noise_tensor_array[i];
        else:
            noise.data = noise.data * noise_std;
    elif(dist == 'P'):
        noise = torch.randn_like(data);
        if mode == 'S':
            noise_std /= 255.
            noise = torch.poisson(data*noise_std)/noise_std - data
    return noise
