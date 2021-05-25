# Main file to call cauchy csc -- this is the start of the algorithm
import os
import argparse 

from sklearn.svm import SVC
from sklearn import metrics as m
from torch.utils.data import DataLoader

import torch

from model import nn
from model import svm
#from model import deep_csc as dcsc
from model import ccsc
from utils import data
from utils import progress_track as pt
from utils import parameters as p


# Available tasks:
# * csc -- Learn CSC only
# * clf -- Full learning (CSC + Classification)
# Example: 
# python run_task.py -xml C:\phd\src\phd_project\cauchy_csc\architectures\cauchy_depth_2.xml -s 0 -d 2
# python run_task.py -xml C:\phd\src\phd_project\cauchy_csc\architectures\cauchy_clf_fm.xml -s 0 -d 2
# python run_task.py -xml C:\phd\src\phd_project\cauchy_csc\architectures\laplace_depth_5.xml -s 0 -d 2
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-xml','--xmlpath', help='The path of the XML file to parse to arguments', type=str, required=True)
	parser.add_argument('-s','--seed', help='Seed for random numbers', type=int, default=0)
	parser.add_argument('-t', '--task', help='Train/Inference', type=str, default="train")
	parser.add_argument('-d', '--imgdim', help='Image dimension', type=int, default=3)
	parser.add_argument('-do', '--dropout', help='Dropout', type=float, default=None)
	parser.add_argument('-wd', '--weightdecay', help='Weight decay', type=float, default=None)
	parser.add_argument('-lr', '--learningrate', help='Learning rate', type=float, default=None)
	parser.add_argument('-mp', '--maxpoolingsize', help='MaxPooling size/stride', type=int, default=None)
	parser.add_argument('-od', '--outputdir', help='Output dir', type=str, default=None)
	parser.add_argument('-ld', '--logdir', help='Log dir', type=str, default=None)
	parser.add_argument('-td', '--traindir', help='Train dir', type=str, default=None)
	parser.add_argument('-vd', '--validdir', help='Valid dir', type=str, default=None)
	parser.add_argument('-dc', '--diagcode', help='Diagnosis Code', type=str, default=None)
	
	args = vars(parser.parse_args())

	xml_path       = args['xmlpath'] 
	seed           = args['seed']
	imgdim         = args['imgdim']
	dropout 	   = args['dropout']
	weightdecay    = args['weightdecay']
	lr 	  	  	   = args['learningrate']
	mpsize 	 	   = args['maxpoolingsize']
	output_dir 	   = args['outputdir']
	log_dir 	   = args['logdir']
	train_dir 	   = args['traindir']
	valid_dir 	   = args['validdir']
	diagnosis_code = args['diagcode']

	params = p.get_params(xml_path)
	exp_id = pt.get_time_fileformat()

	print("Experiment ID:", exp_id)
	for param in params:
		setattr(param, 'tag', pt.get_time_fileformat())
	csc_train_time = None
	csc_test_time = None
	clf_train_time = None
	start_time = pt.get_time()
	z_shape = None
	if params[0].l_type == "csc":
		if output_dir:
			params[0].output_dir = output_dir
		if log_dir:
			params[0].log_dir = log_dir
		if train_dir:
			params[0].train_input_dir = train_dir
		if valid_dir:
			params[0].test_input_dir = valid_dir
		filter_path = os.path.join(params[0].output_dir, 'filter_bank.pckl')
		filter_bank_exists = os.path.exists(filter_path)
		if (params[0].train_input_dir != "") and (not filter_bank_exists):
			#f, z_shape = dcsc.deep_ccsc(params, params[0].train_input_dir, imgdim=imgdim, seed=seed)

			f, z_train, z_shape = ccsc.deep_ccsc(params, params[0].train_input_dir, imgdim=imgdim, seed=seed)
			csc_train_time = pt.get_time()
		if params[0].test_input_dir != "":
			if 'f' in locals():
				#print(f)
				initial_f = f
			else:
				#print(params[0].initial_f)
				if params[0].initial_f != "":
					initial_f = params[0].initial_f
				else:
					if filter_bank_exists:
						initial_f = filter_path
			params[0].lr_z = 0.02
			__, z_valid, z_shape = ccsc.deep_ccsc(params, params[0].test_input_dir, initial_f=initial_f, imgdim=imgdim, seed=seed, learn_filters=False)
			csc_test_time = pt.get_time()
		else:
			z_valid = z_train
	else: 
		z_train = [params[0].train_input_dir]
		z_valid = [params[0].test_input_dir]
	if params[-1].l_type in ["clf", "svm"]:
		if output_dir:
			params[-1].output_dir = output_dir
		if log_dir:
			params[-1].log_dir = log_dir
		if train_dir:
			params[-1].train_input_dir = train_dir
			z_train = [params[-1].train_input_dir]
		if valid_dir:
			params[-1].test_input_dir = valid_dir
			z_valid = [params[-1].test_input_dir]
		if diagnosis_code:
			params[-1].diagnosis_code = p.get_true_type(diagnosis_code)
 	
		data_train = data.Data(z_train[-1], n_samples=params[0].num_samples, seed=seed, transformations=params[-1].transformations)
		data_valid = data.Data(z_valid[-1], n_samples=params[0].num_samples, seed=seed, transformations=params[-1].transformations)
		print("\n\nClassifier parameters")
		print(params[-1])
		print("\n\nDataset properties")
		print(data_train, "\n")
		print(data_valid)

		params[-1].classes = data_train.classes
		# shuffle set to False as the shuffle happens when the data is created
		train_loader = DataLoader(
			data_train,
			batch_size=params[-1].batch_size,
			shuffle=False,
			num_workers=params[-1].num_workers,
			pin_memory=True
		)
		#print(train_loader.dataset.labels)
		valid_loader = DataLoader(
			data_valid,
			batch_size=params[0].batch_size,
			shuffle=False,
			num_workers=params[0].num_workers,
			pin_memory=True
		)


		if params[-1].l_type == "clf":
			if params[-1].initial_f:
				params[-1].initial_f = pt.load_progress(params[-1].initial_f)
			if dropout:
				params[-1].dropout = dropout
			if weightdecay:
				params[-1].weight_decay = weightdecay
			if lr:
				params[-1].learning_rate = lr
			if lr:
				params[-1].learning_rate = lr
			if not z_shape:
				z_shape = data_train[0]['image'].shape
			#data_train = data.get_dataset(z, params[0].num_samples, seed)#, labels=True)
			#data_valid = data.get_dataset(params[0].test_input_dir, params[0].num_samples, seed)#, labels=True)
			if params[-1].prior != params[0].prior:
				params[-1].prior = params[0].prior
			model = nn.FC_NN(params[-1], z_shape, seed=seed)
			loss_function = torch.nn.CrossEntropyLoss()
			optimizer = eval("torch.optim." + params[-1].optimizer)(filter(lambda x: x.requires_grad, model.parameters()),
							lr=params[-1].learning_rate,
							weight_decay=params[-1].weight_decay)

			model = nn.train(model, train_loader, valid_loader, loss_function, optimizer, params[-1])
			clf_path = os.path.join(params[-1].output_dir, "classifier_%s.cnn" % exp_id)

			train_predicted, train_labels, train_loss, train_scores = nn.predict(model, train_loader, loss_function, params[-1].gpu, log_path="", get_scores = True)
			valid_predicted, valid_labels, valid_loss, valid_scores = nn.predict(model, valid_loader, loss_function, params[-1].gpu, log_path="", get_scores = True)
		else:
			print("Training with SVM")
			print(data_train.classes)
			#features_train = data_train[:]['image'].detach().numpy().reshape([len(data_train),-1])
			#features_valid = data_valid[:]['image'].detach().numpy().reshape([len(data_valid),-1])
			#train_labels = data_train[:]['label']
			#valid_labels = data_valid[:]['label']

			#clf = SVC(kernel=params[-1].kernel, C=params[-1].C, gamma=params[-1].gamma)
			#clf.fit(features_train, train_labels)
			model = svm.train(train_loader, params[-1].gpu, data_train.classes)
			train_predicted, train_labels = svm.predict(model, train_loader, params[-1].gpu)
			valid_predicted, valid_labels = svm.predict(model, valid_loader, params[-1].gpu)
			train_scores = None
			valid_scores = None
			print(model)
			clf_path = os.path.join(params[-1].output_dir, "classifier_%s.svm" % exp_id)
		if params[-1].output_dir != "":
			pt.create_path(params[-1].output_dir)
			pt.save_progress(model, clf_path)
			print("\n\nClassifier saved at ", clf_path)
			res = {}
			res['y_true_train'] = train_labels
			res['y_pred_train'] = train_predicted
			res['y_true_valid'] = valid_labels
			res['y_pred_valid'] = valid_predicted
			res['train_scores'] = train_scores
			res['valid_scores'] = valid_scores
			res_path = os.path.join(params[-1].output_dir, "res_%s.pckl" % exp_id)
			pt.save_progress(res, res_path)
			print("\n\nResults saved at ", res_path)

		train_bal_acc = m.balanced_accuracy_score(train_labels, train_predicted)
		valid_bal_acc = m.balanced_accuracy_score(valid_labels, valid_predicted)

		metrics_train = nn.evaluate_prediction(train_labels, train_predicted, train_scores)
		metrics_valid = nn.evaluate_prediction(valid_labels, valid_predicted, valid_scores)
		
		if params[-1].output_dir != "":
			metrics = {}
			metrics["exp_id"] = exp_id
			metrics["train_bal_acc"] = train_bal_acc
			metrics["valid_bal_acc"] = valid_bal_acc
			metrics["train"] = metrics_train
			metrics["valid"] = metrics_valid
			metrics["params"] = params
			metrics_path = os.path.join(params[-1].output_dir, "metrics_%s.pckl" %  exp_id)
			pt.save_progress(metrics, metrics_path)
			print("Metrics saved at ", metrics_path)
			#print(metrics)
		clf_train_time = pt.get_time()
	


	end_time = pt.get_time()
	processing_time = pt.get_time(start_time)
	pt.log_event("\n\nClassification ended.\nStart time: %s\nEnd time: %s\nElapsed time: %s" % 
		(str(start_time), str(end_time), str(processing_time)))
	if csc_train_time:
		pt.log_event("CSC Train time: %s" % (str(csc_train_time - start_time)))
	if csc_train_time:
		if csc_train_time:
			pt.log_event("CSC Test time: %s" % (str(csc_test_time - csc_train_time)))
		else:
			pt.log_event("CSC Test time: %s" % (str(csc_test_time - start_time)))
	if csc_train_time:
		if csc_test_time:
			pt.log_event("Classifier Train time: %s" % (str(clf_train_time - csc_test_time)))
		elif csc_train_time:
			pt.log_event("Classifier Train time: %s" % (str(clf_train_time - csc_train_time)))
		else:
			pt.log_event("Classifier Train time: %s" % (str(clf_train_time - start_time)))

