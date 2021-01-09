import torch
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import cv2
from utils import get_noise, ssim, psnr

metrics_key = ['psnr', 'psnr_delta', 'ssim', 'ssim_delta'];

def tensor_to_image(torch_image, low=0.0, high = 1.0, clamp = True):
	if clamp:
		torch_image = torch.clamp(torch_image, low, high);
	return torch_image[0,0].cpu().data.numpy()


def normalize(data):
	return data/255.

def convert_dict_to_string(metrics):
	return_string = '';
	for x in metrics.keys():
		return_string += x+': '+str(round(metrics[x], 3))+' ';
	return return_string



def get_all_comparison_metrics(denoised, source, noisy = None,  return_title_string = False, clamp = True):

	
		
	metrics = {};
	metrics['psnr'] = np.zeros(len(denoised))
	metrics['ssim'] = np.zeros(len(denoised))
	if noisy is not None:
		metrics['psnr_delta'] = np.zeros(len(denoised))
		metrics['ssim_delta'] = np.zeros(len(denoised))

	if clamp:
		denoised = torch.clamp(denoised, 0.0, 1.0)

	
	metrics['psnr'] = psnr(source, denoised);
	metrics['ssim'] = ssim(source, denoised);

	if noisy is not None:
		metrics['psnr_delta'] = metrics['psnr'] - psnr(source, noisy);
		metrics['ssim_delta'] = metrics['ssim'] - ssim(source, noisy);

	


	if return_title_string:
		return convert_dict_to_string(metrics)
	else:
		return metrics


def average_on_folder(path_to_dataset, net, noise_std, 
			verbose=True, device = torch.device('cuda') ):

	if(verbose):
		print('Loading data info ...\n')
		print('Dataset: ', path_to_dataset)


	files_source = glob.glob(os.path.join(path_to_dataset, '*.png'))
	files_source.sort()
	
	avg_metrics = {};
	for x in metrics_key:
		avg_metrics[x] = 0.0;

	for f in files_source:
		# image
		Img = cv2.imread(f)
		Img = normalize(np.float32(Img[:,:,0]))
		Img = np.expand_dims(Img, 0)
		Img = np.expand_dims(Img, 1)
		ISource = torch.Tensor(Img)
		
		# noisy image
		INoisy = get_noise(ISource, noise_std = noise_std, mode ='S') + ISource
		ISource, INoisy = ISource.to(device), INoisy.to(device)
					  
		out = torch.clamp(net(INoisy), 0., 1.)
		
		ind_metrics = get_all_comparison_metrics(out, ISource, INoisy, return_title_string = False);
		for x in metrics_key:
			avg_metrics[x] += ind_metrics[x];

		if(verbose):
			print("%s %s" % (f, convert_dict_to_string(ind_metrics)))

	for x in metrics_key:
		avg_metrics[x] /= len(files_source);

	if(verbose):
		print("\n Average %s" % (convert_dict_to_string(avg_metrics)))

	if(not verbose):
		return avg_metrics


def metrics_avg_on_noise_range(net, path_to_dataset,noise_std_array, n_average = 1, device = torch.device('cuda')):
	
	print(path_to_dataset)

	array_metrics = {};
	for x in metrics_key:
		array_metrics[x] = np.zeros(len(noise_std_array))
   
	for j, noise_std in enumerate(noise_std_array):
		metric_list = [None]*n_average
		for i in range(n_average):
			metric_list[i] = average_on_folder(path_to_dataset, net, 
												noise_std = noise_std,
												verbose=False, device=device);

		for x in metrics_key:
			for i in range(n_average):
				array_metrics[x][j] += metric_list[i][x]
			array_metrics[x][j] /= n_average
			print('noise: ', int(noise_std*255), ' ', x, ': ', str(array_metrics[x][j]))

	return array_metrics


