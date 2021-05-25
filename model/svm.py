# SVM implementation
import torch
from sklearn.linear_model import SGDClassifier

def train(train_loader, gpu, classes):
	clf = SGDClassifier()
	for i, data in enumerate(train_loader):
		if gpu:
			inputs, lbls = data['image'].cuda(), data['label'].cuda()
		else:
			inputs, lbls = data['image'], data['label']
		if i == 0:
			clf.partial_fit(inputs.reshape(len(inputs),-1),lbls, classes)
		else:
			clf.partial_fit(inputs.reshape(len(inputs),-1),lbls)
	return clf

def predict(model, data_loader, gpu, get_proba=False):
	for i, data in enumerate(data_loader):
		if gpu:
			inputs, lbls = data['image'].cuda(), data['label'].cuda()
		else:
			inputs, lbls = data['image'], data['label']
		pred = model.predict(inputs.reshape(len(inputs),-1))
		if i == 0:
			predicted = torch.Tensor(pred)
			labels = lbls
		else:
			predicted = torch.cat((predicted, torch.Tensor(pred)), 0)
			labels = torch.cat((labels, lbls), 0)
	return predicted.tolist(), labels.tolist()