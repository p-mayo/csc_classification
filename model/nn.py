# Python file to contain all the process related to cnns

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from time import time
from tensorboardX import SummaryWriter

from scipy import interp
from sklearn import metrics as m
from sklearn.preprocessing import label_binarize
from collections import OrderedDict

from utils import progress_track as pt
from utils.parameters import log_params

class Flatten(nn.Module):
	def forward(self, data):
		return torch.flatten(data, start_dim=1)

class CauchyShrink(nn.Module):
	def forward(self, data):
		pass

class FC_NN(nn.Module):
	def __init__(self, params, data_shape, seed=-1):
		super(FC_NN, self).__init__()
		layers = OrderedDict()
		layer_order = list(params.order.upper())
		layer_idx = {}
		units = 0
		imgdim = len(data_shape[1:])
		if params.order.upper().startswith("F"):
			layers['flatten'] = Flatten().double()
			n_0 = torch.prod(torch.tensor(data_shape))
			print(n_0)
		else:
			n_0 = 0
			data_shape = torch.tensor(data_shape)
		conv = nn.Conv2d if imgdim == 2 else nn.Conv3d
		pool = nn.MaxPool2d if imgdim == 2 else nn.MaxPool3d
		batchnorm = nn.BatchNorm2d if imgdim == 2 else nn.BatchNorm3d
		if len(params.n_units) == 1:
			n_units = [n_0] + (params.depth-1)*params.n_units
		else:
			n_units = [n_0] + params.n_units + [0]
		n_units[-1] = len(params.classes)
		print(data_shape)
		for l in layer_order:
			if l in layer_idx:
				layer_idx[l] += 1
			else:
				layer_idx[l] = 1
			if l == "F":
				if "flatten" not in layers.keys():
					layers['flatten'] = Flatten().double()
					n_units[0] = torch.prod(data_shape)
				layers['linear%d' % layer_idx[l]] = nn.Linear(n_units[units], n_units[units+1]).double()
				units += 1
			elif l == "B":
				layers["batchnorm" + str(layer_idx[l])] = batchnorm(int(data_shape[0])).double()
			elif l == "A":
				layers[params.activation + str(layer_idx[l])] = get_activation(params.activation).double()
			elif l == "D":
				layers["drop%d" % layer_idx[l]] = nn.Dropout(params.dropout).double()
			#elif l == "S":
			#	layers["softmax%d" % layer_idx[l]] = nn.Softmax(dim=1).double()
			elif l=="C":
				if params.initial_f:
					filt_shape = params.initial_f[layer_idx[l]-1].weight.shape
					layers["conv%d" % layer_idx[l]] = conv(data_shape[0], filt_shape[0], 
														filt_shape[-1]).double()
					#data_shape[0] = params.conv[layer_idx[l]-1]['n_filters']
					layers["conv%d" % layer_idx[l]].weight.data = params.initial_f[layer_idx[l]-1].weight.data
					layers["conv%d" % layer_idx[l]].weight.data.requires_grad = params.learn_weights
					if not params.learn_bias:
						layers["conv%d" % layer_idx[l]].bias = torch.nn.Parameter(torch.Tensor(np.zeros([filt_shape[0]])), 
										requires_grad=params.learn_bias)
					data_shape[0] = params.initial_f[layer_idx[l]-1].weight.shape[0]
					data_shape[1:] = (data_shape[1:] - params.initial_f[layer_idx[l]-1].weight.shape[-1] + 1)
				else:
					if seed >= 0:
						torch.random.manual_seed(seed)
					layers["conv%d" % layer_idx[l]] = conv(data_shape[0], params.conv[layer_idx[l]-1]['n_filters'], 
														params.conv[layer_idx[l]-1]['filt_size']).double()
					data_shape[0] = params.conv[layer_idx[l]-1]['n_filters']
					data_shape[1:] = (data_shape[1:] - params.conv[layer_idx[l]-1]['filt_size'] + 1)
				print(data_shape)
				print(layers["conv%d" % layer_idx[l]].weight.shape)
			elif l == "M":
				layers["pool%d" % layer_idx[l]] = pool(params.pool[layer_idx[l]-1]['filt_size'], 
													stride=params.pool[layer_idx[l]-1]['stride']).double()
				data_shape[1:] = (data_shape[1:] - params.pool[layer_idx[l]-1]['filt_size'])/params.pool[layer_idx[l]-1]['stride'] + 1
		self.classifier = nn.Sequential(layers)

		#print(self.classifier.linear1.weight.shape)

	def forward(self, data):
		#for module in self.classifier:
		#	data = module(data)
		data = self.classifier(data)
		return data

def swap_in_out_channels(weights):
	in_channels = weights.shape[0]
	out_channels = weights.shape[1]
	swapped_weights = torch.zeros([out_channels, in_channels] + list(weights.shape[2:]))
	for o in range(out_channels):
		for i in range(in_channels):
			swapped_weights[o,i] = weights[i,o]
	return swapped_weights


def get_activation(activation):
	if activation.lower() == "relu":
		return nn.ReLU()
	elif activation.lower() == "prelu":
		return nn.PReLU()
	elif activation.lower() == "leakyrelu":
		return nn.LeakyReLU()
	elif activation.lower() == "cauchy":
		return CauchyShrink()

def train(model, train_loader, valid_loader, loss_function, optimizer, params):
	model.train()
	n_samples_train = len(train_loader.dataset)
	n_samples_valid = len(valid_loader.dataset)
	if params.log_dir:
		pt.create_path(params.log_dir)
		writer_train = SummaryWriter(os.path.join(params.log_dir, 'train_' + params.prior  + '_' + params.tag))
		writer_valid = SummaryWriter(os.path.join(params.log_dir, 'valid_' + params.prior  + '_' + params.tag))
	for epoch in range(1,params.epochs+1):
		print("%s \n======================== EPOCH %d" % (pt.get_time(), epoch) )
		processed = 0
		for i, data in enumerate(train_loader, 1):
			optimizer.zero_grad()
			if params.gpu:
				imgs, labels = data['image'].cuda(), data['label'].cuda()
			else:
				imgs, labels = data['image'], data['label']
			
			# Forward pass to get a prediction
			train_output = model(imgs)
			# Getting the actual prediction
			_, predict_batch = train_output.topk(1)
			
			# Computing the loss function
			loss = loss_function(train_output, labels)
			# Get the gradients
			loss.backward()
			# Update the weights of the model
			optimizer.step()
			processed += len(data['image'])
			if i%5 == 0 or i == len(train_loader):
				epoch_progress = processed*100./n_samples_train
				train_predicted, train_labels, train_loss = predict(model, train_loader, loss_function, params.gpu)
				mean_loss_train = train_loss / n_samples_train

				valid_predicted, valid_labels, valid_loss = predict(model, valid_loader, loss_function, params.gpu)
				#print(valid_loss, n_samples_valid)
				mean_loss_valid = valid_loss / n_samples_valid

				model.train()

				train_bal_acc = m.balanced_accuracy_score(train_labels, train_predicted)
				valid_bal_acc = m.balanced_accuracy_score(valid_labels, valid_predicted)

				pt.log_event("%s [%d/%d (%02d%%)] Train Bal Acc: %.4f | Valid Bal Acc: %.4f | Train Loss: %.4f | Validation Loss: %.4f" % 
					(pt.get_time(), processed, n_samples_train, epoch_progress, train_bal_acc, valid_bal_acc, mean_loss_train, mean_loss_valid), 
					"")

				if params.log_dir:
					global_step = i + epoch * len(train_loader)
					writer_train.add_scalar('balanced_accuracy', train_bal_acc, global_step, display_name=params.tag)
					writer_train.add_scalar('loss',  mean_loss_train, global_step, display_name=params.tag)
					writer_valid.add_scalar('balanced_accuracy', valid_bal_acc, global_step, display_name=params.tag)
					writer_valid.add_scalar('loss', mean_loss_valid, global_step, display_name=params.tag)
					
					if i == len(data['image']):
						metrics = {'hparam/balanced_accuracy': train_bal_acc, 'hparam/mean_loss':mean_loss_train}
						log_params(writer_train, params, metrics)

						metrics = {'hparam/balanced_accuracy': valid_bal_acc, 'hparam/mean_loss':mean_loss_valid}
						log_params(writer_valid, params, metrics)
	if params.log_dir:
		writer_valid.close()
		writer_train.close()
	with SummaryWriter() as w:
		w.add_hparams({'lr': 0.1*i, 'bsize': i},
					  {'hparam/accuracy': 10*i, 'hparam/loss': 10*i})
	return model

def predict(model, data_loader, loss_function, gpu, log_path="", get_scores = False):
	model.eval()
	labels = []
	softmax = torch.nn.Softmax(dim=1).double()
	total_loss = 0
	total_time = 0
	with torch.no_grad():
		for i, data in enumerate(data_loader):
			if gpu:
				inputs, lbls = data['image'].cuda(), data['label'].cuda()
			else:
				inputs, lbls = data['image'], data['label']
			predictions = model(inputs)
			loss = loss_function(predictions, lbls)
			if(loss.item() == np.nan):
				for p in range(predictions):
					if np.any(predictions[p] == np.nan):
						print(inputs[p])
						print(loss.item())
						print(predictions[p])
						print(lbls[p])
						print(data['idx'][p])
			total_loss += loss.item()
			_, pred = torch.max(predictions.data, 1)
			if i == 0:
				scores = softmax(predictions)
				predicted = pred
				labels = lbls
			else:
				scores = torch.cat((scores, softmax(predictions)), 0)
				predicted = torch.cat((predicted, pred), 0)
				labels = torch.cat((labels, lbls), 0)
			del inputs, predictions, loss
		#pt.log_event('%s \t\tMean time per batch loading (test): %s' % (pt.get_time(), str(total_time / len(dataloader) * dataloader.batch_size)), log_path)
		torch.cuda.empty_cache()
	if get_scores:
		return predicted.tolist(), labels.tolist(), total_loss, scores.tolist()
	else:
		return predicted.tolist(), labels.tolist(), total_loss


def evaluate_prediction(y_true, y_pred, y_pred_scores):
	n_classes = list(set(y_true))
	y_true_binary = label_binarize(y_true, n_classes)
	y_pred_binary = label_binarize(y_true, n_classes)

	metrics = {}

	fpr = dict()
	tpr = dict()
	roc_auc = dict()

	true_positive = dict()
	true_negative = dict()
	false_positive = dict()
	false_negative = dict()

	for i in n_classes:
		fpr[i], tpr[i], _ = m.roc_curve(y_true_binary[:, i], y_pred_binary[:, i])
		roc_auc[i] = m.auc(fpr[i], tpr[i])
		true_positive[i] = np.sum((y_true_binary[:, i] == 1) & (y_pred_binary[:, i] == 1))
		true_negative[i] = np.sum((y_true_binary[:, i] == 0) & (y_pred_binary[:, i] == 0))
		false_positive[i] = np.sum((y_true_binary[:, i] == 0) & (y_pred_binary[:, i] == 1))
		false_negative[i] = np.sum((y_true_binary[:, i] == 1) & (y_pred_binary[:, i] == 0))

	
	# Compute micro-average ROC curve and ROC area
	fpr["micro"], tpr["micro"], _ = m.roc_curve(y_true_binary.ravel(), y_pred_binary.ravel())
	roc_auc["micro"] = m.auc(fpr["micro"], tpr["micro"])

	all_fpr = np.unique(np.concatenate([fpr[i] for i in n_classes]))

	# Then interpolate all ROC curves at this points
	mean_tpr = np.zeros_like(all_fpr)
	for i in n_classes:
		mean_tpr += interp(all_fpr, fpr[i], tpr[i])

	# Finally average it and compute AUC
	mean_tpr /= len(n_classes)

	fpr["macro"] = all_fpr
	tpr["macro"] = mean_tpr
	roc_auc["macro"] = m.auc(fpr["macro"], tpr["macro"])

	if y_pred_scores:
		macro_roc_auc_ovo = m.roc_auc_score(y_true_binary, y_pred_scores, multi_class="ovo",
										average="macro")
		weighted_roc_auc_ovo = m.roc_auc_score(y_true_binary, y_pred_scores, multi_class="ovo",
										average="weighted")
		macro_roc_auc_ovr = m.roc_auc_score(y_true_binary, y_pred_scores, multi_class="ovr",
										average="macro")
		weighted_roc_auc_ovr = m.roc_auc_score(y_true_binary, y_pred_scores, multi_class="ovr",
										average="weighted")
	else:
		macro_roc_auc_ovo = None
		weighted_roc_auc_ovo = None
		macro_roc_auc_ovr = None
		weighted_roc_auc_ovr = None
	return {"fpr":fpr, "tpr":tpr, "roc_auc":roc_auc, "true_positive":true_positive,
			"true_negative":true_negative, "false_positive":false_positive,
			"false_negative":false_negative, "macro_roc_auc_ovo":macro_roc_auc_ovo,
			"weighted_roc_auc_ovo":weighted_roc_auc_ovo, "macro_roc_auc_ovr":macro_roc_auc_ovr,
			"weighted_roc_auc_ovr":weighted_roc_auc_ovr}