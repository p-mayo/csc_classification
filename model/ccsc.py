# Second version of Cauchy Convolutional Sparse Coding
# This one attempts to use PyTorch and some other utilities to perform convolutions
# and some other operations

import os
import argparse 

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T 

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from sklearn.preprocessing import normalize
from scipy.interpolate import interp1d

from model import thresholding as th
from utils import cauchy
from utils import data as dt
from utils import display as dsp
from utils import progress_track as pt
from utils.math import get_avg_psnr
from utils.parameters import Parameters, log_params

MIN_VAL = 1e-10

def cauchy_penalty(z, gamma):
	# Other constants (pi) are removed as they do not influence the minima
	if "torch" in str(type(z)):
		return torch.sum(torch.log(gamma**2 + z**2))
	else:
		return np.sum(np.log(gamma**2 + z**2))

def l1_penalty(z):
	# L1 norm is only the sum ob absolute values
	return torch.sum(torch.abs(z))

def l0_penalty(z):
	return torch.sum(z != 0)

def sparsity_penalty (z, params, prior):
	# Function to generalise and choose sparse penalty (only L1 and Cauchy so far)
	if prior == "cauchy":
		return cauchy_penalty(z, params.param)
	elif prior == "laplace":
		return l1_penalty(z)
	else:
		return l0_penalty(z)

def cost_function(f, z, y, lmbda, params, return_all=False):
	y_hat = f(z)	# Convolution of filters with feature maps
	#print(z.shape, " ---- ", y_hat.shape, "----", y.shape)
	least_squares = torch.sum((y - y_hat)**2) # Reconstruction error (L2-norm)
	sparse_err = lmbda*sparsity_penalty(z, params, params.prior)
	cost = least_squares + sparse_err # Adding sparse penalty
	if return_all:
		return cost, least_squares, sparse_err
	else:
		return cost

def log_cost(cost, rec_err, sparse_err, it_out, it_in, prior, log_path=""):
	#overall = rec_err + sparsity_cost
	pt.log_event("%s [%04d %04d] COST = %f, Reconstruction error: %f, Sparsity penalty (%s prior): %f" 
				% (pt.get_time(), it_out, it_in, cost, rec_err, prior, sparse_err), log_path)

def save_feature_maps(z, idx, labels, path):
	for i in range(len(idx)):
		pt.save_progress(z[i].detach().numpy(), os.path.join(path, "fm_%06d_class_%s.fm" % (idx[i], str(labels[i].item()))))

def load_feature_maps(idx, labels, path):
	z = []
	for i in range(len(idx)):
		z.append(torch.unsqueeze(torch.tensor(pt.load_progress(os.path.join(path, "fm_%06d_class_%s.fm" % (idx[i], labels[i].item())))),0))
	return torch.cat(z, dim=0).double()

def learn_f_z(data_loader, params, imgdim, initial_f = [], initial_z=[], path = "", 
				learn_filters=True, seed=-1):
	if params.log_dir:
		pt.create_path(params.log_dir)
		mode = 'train' if learn_filters else 'valid'
		writer = SummaryWriter(os.path.join(params.log_dir, mode + '_' + params.prior + '_' + params.tag))
	data_shape = data_loader.dataset[0]['image'].shape
	n_samples = len(data_loader.dataset)
	in_channels = params.num_filters 
	out_channels = data_shape[0]
	kernel_size = params.filter_size
	initial_lr = [params.lr_d, params.lr_z]
	# Need to do something tricky here to have a convolution that outputs the
	# estimated input data, and not the feature maps
	# Defining filters
	if type(initial_f) in [torch.nn.modules.conv.ConvTranspose2d, torch.nn.modules.conv.ConvTranspose3d]:
		f = initial_f
	else:
		print("	IMAGE DIMENSION : ", imgdim)
		if seed >=0:
			torch.random.manual_seed(seed)
		if imgdim == 3:
			f = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, padding=0)
			#f = nn.Conv3d(in_channels, out_channels, kernel_size, padding=kernel_size-1)
			norm_dim = (2,3,4)
		else:
			f = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, padding=0)
			#f = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size-1)
			norm_dim = (2,3)
		f.weight.data = f.weight.double()
		f_shape = f.weight.shape

		with torch.no_grad():
			if len(initial_f) != 0:
				f.weight = torch.nn.Parameter(torch.tensor(initial_f).unsqueeze_(1))
			#f_aux = f.weight.detach().numpy()
			#f_aux = normalize(f_aux.reshape(f_shape[0],-1),axis=1).reshape(f_shape)
			#f.weight.data = torch.tensor(f_aux)
			f.weight.data = f.weight.data.reshape(f_shape[0],f_shape[1],-1).div_(torch.norm(f.weight.data.reshape(f_shape[0],f_shape[1],-1), dim=2, keepdim=True))
			f.weight.data = f.weight.data.reshape(f_shape)
			f.bias = torch.nn.Parameter(torch.tensor(np.zeros([out_channels])), requires_grad=params.learn_bias)
	
	th_params = {}
	th_params["lmbda"] = params.lmbda*params.lr_z

	if params.prior == "laplace" or params.prior == "hard":
		th_params["portion"] = 1.
	elif params.prior == "cauchy":
		th_params["gamma"] = params.param
		th_params["hard"] = 0.

	# Beginning of the learning for both the bank of filters and
	# the feature maps
	total_loads = len(data_loader)
	# -1 -- Estimate per batch
	estim_gamma = False
	if params.param == -1:
		estim_gamma = True
	fm_path = os.path.join(path, 'feature_maps')
	pt.create_path(fm_path)

	f.weight.requires_grad = True

	# Defining feature maps...
	if len(initial_z) == 0:
		z_shape = torch.tensor(data_shape[-imgdim:]) - torch.tensor(f.weight.shape[2:]) + 1
		if params.batch_size >= n_samples:
			z = torch.zeros([params.batch_size, in_channels] + list(z_shape), dtype=torch.float64, requires_grad = True)
			optimizer_z = optim.SGD([{'params':z}], lr=params.lr_z)
		else:
			lr_zs = int(np.ceil(n_samples/params.batch_size))
			lr_zs = params.lr_z*np.ones(lr_zs)
	else:
		z = torch.tensor(initial_z, dtype=torch.float64)

	# Defining the optimiser for the filters
	optimizer_f = optim.SGD([{'params':f.weight}], lr=params.lr_d)
	f_old = f.weight.clone().detach()
	curve = None
	update_curve = True
	# Learning starts
	for i in range(params.start, params.max_outer):
		pt.log_event("\n%s [---- ----] ============= Iteration %s"% (pt.get_time(), i), "")
		pt.log_event("%s [---- ----] Learning coefficients"% (pt.get_time()), "")

		f_aux = np.squeeze(f.weight.detach().numpy())
		for batch, data in enumerate(data_loader, 0):
			if params.batch_size < n_samples:
				params.lr_z = lr_zs[batch]
				if i == 0 and not exist_feature_maps(data['idx'], data['label'], fm_path):
					z = torch.zeros([len(data['idx']), in_channels] + list(z_shape), dtype=torch.float64, requires_grad = True)
					#z.requires_grad = True
				else:
					z = torch.zeros([len(data['idx']), in_channels] + list(z_shape), dtype=torch.float64, requires_grad = True)
					#z.requires_grad = True
					z.data = load_feature_maps(data['idx'], data['label'], fm_path).data
				optimizer_z = optim.SGD([{'params':z}], lr=params.lr_z)

			if estim_gamma:
				params.param = cauchy.estimate_gamma(data["image"].numpy().ravel())
			if params.gpu:
				imgs = data['image'].cuda()
			else:
				imgs = data['image']
			# -------------------------------------------
			# Learning the Feature Maps (z)
			# -------------------------------------------
			with torch.no_grad():
				z_old = z.clone().detach()
				old_cost = cost_function(f, z, imgs, params.lmbda, params).detach()
			#print(z.requires_grad, f.weight.requires_grad)
			#z.requires_grad = True
			#f.weight.requires_grad = False
			for j in range(params.max_inner):
				cost = cost_function(f, z, imgs, 0, params)
				#print(cost)
				cost.backward()
				#print(cost)
				#print(z.grad, f.weight.grad)
				with torch.no_grad():	
					z -= params.lr_z*z.grad
					#print(params.approximate)
					if params.approximate:
						if not curve or update_curve:
							x = np.linspace(z.min(), z.max(), 200)
							y = th.shrink(x, params.prior, th_params)
							curve = interp1d(x, y, fill_value='extrapolate') # , kind='cubic'
						z.data = torch.tensor(th.shrink(z.clone().detach().numpy(), params.prior, th_params, curve)).double().data
					else:
						z.data = torch.tensor(th.shrink(z.detach().numpy(), params.prior, th_params)).double().data
					new_cost, rec_err, sparse_err = cost_function(f, z, imgs, params.lmbda, params, True)
					log_cost(new_cost, rec_err, sparse_err, i, j, params.prior)
					if params.adapt_lr:
						if new_cost >= old_cost:
							params.lr_z /= 2
							if params.lr_z < MIN_VAL:
								params.lr_z = initial_lr[1]
							th_params["lmbda"] = params.lmbda*params.lr_z
							pt.log_event("%s [%04d %04d]        Updating learning rate for batch, new value: %f\t"
								% (pt.get_time(), i, j, params.lr_z), "")
							z.data = z_old.data
							update_curve = True
						else:
							z_old = z.clone().detach()
							old_cost = new_cost
							update_curve = False
				z.grad.zero_()
				f.zero_grad()
			if (params.batch_size < n_samples) or ((i == params.max_outer -1) and (j == params.max_inner - 1)):
				pt.log_event("%s [%04d %04d]        Saving feature maps (this might take a few minutes)"% (pt.get_time(), i, j), "")
				save_feature_maps(z, data["idx"], data['label'], fm_path)
			if params.batch_size < n_samples:
				lr_zs[batch] = params.lr_z
				update_curve = True
		# -------------------------------------------
		# Learning the Filters (f)
		# -------------------------------------------
		if learn_filters:
			pt.log_event("%s [---- ----] Learning filters"% (pt.get_time()), "")
			#z.requires_grad = False
			#f.weight.requires_grad = True
			#for batch, data in enumerate(data_loader, 0):
			#	with torch.no_grad():
			#		f_old = f.weight.detach().numpy()
			#		old_cost = cost_function(f, z, imgs, params.lmbda, params)
			#		if params.batch_size < n_samples:
			#			z = load_feature_maps(data['idx'], data['label'], fm_path)
			#	for j in range(params.max_inner):
			#		f.weight.requires_grad = True
			#		cost = cost_function(f, z, imgs, params.lmbda, params)
			#		cost.backward()
			#		f.weight.data -= params.lr_d*f.weight.grad
			#		with torch.no_grad():
			#			f_aux = f.weight.detach().numpy()
			#			f_aux = normalize(f_aux.reshape(f_shape[0],-1),axis=1).reshape(f_shape)
			#			f.weight.data = torch.tensor(f_aux)
			#			f.weight.data.div_(torch.norm(f.weight, dim=norm_dim,keepdim=True))
			#			new_cost, rec_err, sparse_err = cost_function(f, z, imgs, params.lmbda, params, True)
			#			log_cost(new_cost, rec_err, sparse_err, i, j, params.prior)
			#			if params.adapt_lr:
			#				if new_cost >= old_cost:
			#					params.lr_d /= 2
			#					if params.lr_d < MIN_VAL:
			#						params.lr_d = initial_lr[0]
			#					pt.log_event("%s [%04d %04d]        Updating learning rate, new value: %f\t"
			#						% (pt.get_time(), i, j, params.lr_d), "")
			#					f.weight.data = torch.tensor(f_old)
			#				else:
			#					f_old = f.weight.detach().numpy()
			#					old_cost = new_cost
			#		f.zero_grad()
			#		z.grad.zero_()
			for j in range(params.max_inner):
				for batch, data in enumerate(data_loader, 0):
					if params.batch_size < n_samples:
						z = torch.zeros([len(data['idx']), in_channels] + list(z_shape), dtype=torch.float64, requires_grad = True)
						#z.requires_grad = True
						z.data = load_feature_maps(data['idx'], data['label'], fm_path).data
					if params.gpu:
						imgs = data['image'].cuda()
					else:
						imgs = data['image']
					optimizer_f.zero_grad()
					cost = cost_function(f, z, imgs, params.lmbda, params)
					cost.backward()
					optimizer_f.step()
					with torch.no_grad():
						#f_aux = f.weight.detach().numpy()
						#f_aux = normalize(f_aux.reshape(f_shape[0],-1),axis=1).reshape(f_shape)
						#f.weight.data = torch.tensor(f_aux)
						#f.weight.data.div_(torch.norm(f.weight, dim=norm_dim,keepdim=True))
						f.weight.data = f.weight.data.reshape(f_shape[0],f_shape[1],-1).div_(torch.norm(f.weight.data.reshape(f_shape[0],f_shape[1],-1), dim=2, keepdim=True))
						f.weight.data = f.weight.data.reshape(f_shape)
						new_cost, rec_err, sparse_err = cost_function(f, z, imgs, params.lmbda, params, True)
						log_cost(new_cost, rec_err, sparse_err, i, j, params.prior)
						if params.adapt_lr:
							if new_cost >= old_cost:
								params.lr_d /= 2
								if params.lr_d < MIN_VAL:
									params.lr_d = initial_lr[0]
								pt.log_event("%s [%04d %04d]        Updating learning rate, new value: %f\t"
									% (pt.get_time(), i, j, params.lr_d), "")
								f.weight.data = f_old.data
								for param in optimizer_f.param_groups:
									param['lr'] = params.lr_d
							else:
								f_old = f.weight.clone().detach()
								old_cost = new_cost.detach()
			#		f.zero_grad()
			#		z.grad.zero_()
		#metrics = get_metrics(f, z, data_loader, params, imgdim, path, fm_path, i, learn_filters)
		#if params.log_dir:
		#	writer.add_scalar('average_psnr',metrics['hparam/psnr'], i+1)
		#	writer.add_scalar('average_l0', metrics['hparam/sparsity'], i+1)
	#if params.log_dir:
	#	log_params(writer, params, metrics, "layer_" + str(params.n_layer))
	#	writer.close()
	pt.log_event("\n%s Learning is over"% (pt.get_time()), "")
	return f, fm_path, list(z.shape[1:])

def get_metrics(f, z, data_loader, params, imgdim, path, fm_path, i, learn_filters, forze_load=False, mode=""):
	is_list = type(f) == list
	if is_list:
		out_channels = f[0].out_channels
	else:
		out_channels = f.out_channels
	with torch.no_grad():
		avg_psnr = 0
		avg_l0 = 0
		for batch, data in enumerate(data_loader, 0):
			#print(batch, params.num_samples, len(data['idx']))
			print(params.batch_size, params.num_samples, forze_load, params.num_samples)
			if (params.batch_size < params.num_samples) or forze_load or params.num_samples == 0:
				z = load_feature_maps(data['idx'], data['label'], fm_path)
			#y_hat = reconstruct_input(filter_bank, z)
			#print(batch, params.num_samples, len(data['idx']))
			if is_list:
				y_hat = reconstruct_input(f, z)
			else:
				y_hat = f(z).detach().numpy()
			#print(data['image'].numpy().shape, y_hat.shape, z.shape, f.weight.shape)
			avg_psnr += get_avg_psnr(data['image'].numpy(), y_hat)
			avg_l0 += l0_penalty(z)
		avg_psnr /= len(data_loader)
		avg_l0 /= len(data_loader.dataset)
		if out_channels == 1 and path != "":
			if imgdim == 2:
				img_path = os.path.join(path,"learning_recons_iter_%s%s.png" % (str(i+1), mode))
				img_title = "Reconstruction in iteration %s" % (str(i+1))
				dsp.show_imgs(np.squeeze(y_hat[0:25]), img_title, img_path)
				if learn_filters:
					pt.log_event("%s Saving filters"% (pt.get_time()), "")
					img_path = os.path.join(path,"learning_filters_iter_%s.png" % (str(i+1)))
					img_title = "Filter learned in iteration %s" % (str(i+1))
					dsp.show_imgs(np.squeeze(f.weight.detach().numpy()), img_title, img_path)
					pt.log_event("%s Filters saved in %s"% (pt.get_time(), img_path), "")
	return {'hparam/psnr':avg_psnr, 'hparam/sparsity':avg_l0}


# Learning is done in a greedy fashion, i.e. one layer at the time 
# "params" is a list of objexts of class "Parameters", each one specifying the 
# parameters for each layer -- this is experimental, hence it is subject to change
def deep_ccsc(params, dataset_path, imgdim, seed, initial_f="", learn_filters=True):
	f = []
	z = []
	z.append(dataset_path)
	layer = 0
	depth = len(params)

	if os.path.exists(os.path.join(params[0].output_dir,  "filter_bank.pckl")):
		initial_f = os.path.join(params[0].output_dir,  "filter_bank.pckl")
	if type(initial_f) == str:
		if initial_f != "":
			filter_bank = pt.load_progress(initial_f)
			filt = 0
	else:
		filter_bank = initial_f
		filt = 0

	mode = "train" if learn_filters else "valid"

	while layer < depth:
	#for l in range(params.n_layers):
		setattr(params[layer], 'n_layer', layer)
		if params[layer].l_type == "csc":
			if layer > 0:
				shuffle = False
			else:
				shuffle = True
			data = dt.Data(z[-1], n_samples=params[0].num_samples, seed=seed, 
							transformations=params[layer].transformations, shuffle=shuffle,
							diagnosis_code=params[layer].diagnosis_code)
			data_loader = DataLoader(
				data,
				batch_size=params[layer].batch_size,
				shuffle=False,
				num_workers=params[layer].num_workers,
				pin_memory=True
			)
			if params[layer].output_dir != "":
				res_path = os.path.join(params[layer].output_dir, mode ,"l_" + str(layer) + "_" + params[layer].l_type)
				pt.create_path(res_path)
			if len(initial_f) > 0:
				f0 = filter_bank[filt]
				filt += 1
			else:
				f0 = []
			if layer > 0:
				params[layer].param = params[0].param
			elif params[layer].param == "-1":
				params[layer].param = -1
			elif params[layer].param == 0:
				if params[0].prior == "cauchy":
					avg_gamma = 0
					for data in data_loader:
						avg_gamma += cauchy.estimate_gamma(data["image"].numpy().ravel())
					avg_gamma /= len(data_loader)
					params[layer].param = avg_gamma
					print("Gamma is: ", avg_gamma)
				else:
					params[layer].param= 0.01
			f_l, z_l, z_shape = learn_f_z(data_loader, params[layer], imgdim, initial_f=f0, path=res_path, 
											learn_filters=learn_filters, seed=seed)
			f.append(f_l)
			z.append(z_l)
		elif params[layer].l_type == "pool":
			if imgdim == 2:
				pool = nn.MaxPool2d(params[layer].filter_size, stride=params[layer].stride).double()
			elif imgdim == 2:
				pool = nn.MaxPool3d(params[layer].filter_size, stride=params[layer].stride).double()
			z_l, z_shape = process_feature_maps(pool, data_loader, res_path)
			z.append(z_l)
		elif params[layer].l_type in ["cnn", "svm"]:
			break
		layer += 1
	if params[0].output_dir != "" and learn_filters:
		pt.save_progress(f, os.path.join(params[0].output_dir, "filter_bank.pckl"))
	

	data = dt.Data(z[0], n_samples=params[0].num_samples, seed=seed, 
					transformations=params[0].transformations)
	data_loader = DataLoader(
		data,
		batch_size=params[0].batch_size,
		shuffle=False,
		num_workers=params[0].num_workers,
		pin_memory=True
	)

	#mode = 'train' if learn_filters else 'valid'
	#metrics = get_metrics(f, None, data_loader, params[0], imgdim, params[0].output_dir, z[-1], 0, False, True, "_" + mode)
	#writer = SummaryWriter(os.path.join(params[0].log_dir, mode + '_' + params[0].prior + '_' + params[0].tag))
	#log_params(writer, params[0], metrics,"final")
	return f, z, z_shape

def process_feature_maps(process, data_loader, res_path):
	fm_path = os.path.join(res_path,"feature_maps")
	pt.create_path(fm_path)
	for batch, data in enumerate(data_loader):
		z = process(data['idx'])
		save_feature_maps(z, data['idx'], data['label'], fm_path)
	return fm_path, z.shape[1:]

def exist_feature_maps(idx, labels, path):
	for i in range(len(idx)):
		if not os.path.exists(os.path.join(path, "fm_%06d_class_%s.fm" 
			% (idx[i], labels[i].item()))):
			return False
	return True

#def exist_feature_maps(data_loader, res_path):
#	fm_path = os.path.join(res_path,"feature_maps")
#	if not os.path.exists(fm_path):
#		return False
#	for batch, data in enumerate(data_loader):
#		for i in range(len(data['idx'])):
#			if not os.path.exists(os.path.join(fm_path, "fm_%06d_class_%s.fm" % 
#					(data['idx'][i], str(data['label'][i].item())))):
#				return False
#	return True

# The feature maps should correspond to the inner-most learned features
def reconstruct_input(filter_bank, feature_map):
	depth = len(filter_bank)
	with torch.no_grad():
		current_fm = feature_map
		for l in range(depth-1, -1, -1):
			current_fm = filter_bank[l](current_fm)
	return current_fm.detach().numpy()
