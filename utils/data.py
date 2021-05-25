# Python file that allows the loading and processing of the data
import os
import cv2
import torch
import mnist
import random
import numpy as np
import nibabel as nib

from torch.utils.data import Dataset

from utils.parameters import get_true_type
from utils.progress_track import load_progress

def is_nifti(path):
	if path.strip().endswith('.nii') or path.strip().endswith('.nii.gz'):
		return True
	return False

def get_data_type(path, original_type = ""):
	if is_nifti(path) or original_type == "nifti":
		return "nifti", 3
	elif path.endswith(".fm"):
		sample = load_progress(path)
		imgdim = len(sample[2:])
		return "fm", imgdim
	else:
		return "img", 2

class Preprocess():
	def __call__(self,image):
		if torch.max(torch.abs(image)) != 0:
			image /= torch.max(image) - torch.min(image)
		return image - torch.mean(image)

class Flatten():
	def __call__(self, image):
		return torch.flatten(image, start_dim=1)

class Data(Dataset):
	def __init__(self, data_file, classes = [], n_samples=0, seed=0, shuffle=True, transformations=None,
					diagnosis_code = None):
		#self.diagnosis_code = {'cn': 0,'mci':1,'ad':2}
		paths, labels = get_dataset(data_file, n_samples, get_labels=True, paths_only=True, seed=seed, shuffle=shuffle)
		if transformations:
			if transformations.lower() == "preprocess":
				self.transformations = Preprocess()
			else:
				self.transformations = None
		else:
			self.transformations = None
		self.labels = labels
		self.paths = paths
		if is_mnist(data_file):
			self.file_type = 'mnist'
			self.imgdim = 2
		else:
			self.file_type, self.imgdim = get_data_type(paths[0])
		if len(classes) == 0:
			self.classes = list(set(labels))
		else:
			self.classes = classes
		if type(labels) == list:
			if not diagnosis_code:
				self.diagnosis_code = {'cn': 0,'mci':1,'ad':2}
			else:
				self.diagnosis_code = diagnosis_code
			if len(self.diagnosis_code) != len(self.classes):
				if len(self.classes) == 2:
					self.labels = np.array(self.labels)
					if 0 in self.classes and 1 in self.classes:	# CN vs MCI
						self.diagnosis_code = {'cn': 0,'mci':1}	
					elif 0 in self.classes and 2 in self.classes: # CN vs AD
						self.diagnosis_code = {'cn': 0,'ad':1}
						self.labels[self.labels==2] = 1
					elif 1 in self.classes and 2 in self.classes: # MCI vs AD
						self.diagnosis_code = {'ad': 0,'mci':1}
						self.labels[self.labels==2] = 0
					self.labels = self.labels.tolist()
			if type(self.labels[0]) == str:
				for i in range(len(self.labels)):
					self.labels[i] = self.diagnosis_code[labels[i]]


	def __len__(self):
		return len(self.labels) 

	def _get_meta_data(self, idx):
		image_idx = idx
		label = self.labels[idx]
		if self.file_type == "nifti":
			filename = self.paths[idx].split('_')
			participant_id = filename[1] + '_S_' + filename[3]
			image_id = filename[-1].split('.')[0]
			return participant_id, image_idx, image_id, label
		else:
			return image_idx, label

	def _get_path(self, idx):
		return self.paths[idx]

	def _get_full_image(self, idx):
		if self.file_type == "mnist" and type(self.paths[0]) != str:
			image = self.paths[idx]
		else:
			image = load_samples(self.paths[idx], self.file_type)
		if self.imgdim  == len(image.shape):
			image = torch.unsqueeze(image, 0)
		return image.double()

	def __getitem__(self, idx):
		if self.file_type == "nifti":
			participant_id, __, image_id, label = self._get_meta_data(idx)
			image_path = self._get_path(idx)
			image = self._get_full_image(idx)

			if self.transformations:
				image = self.transformations(image)
			sample = {'image': image, 'label': label, 'participant_id': participant_id, 'image_id': image_id,
					  'image_path': image_path, 'idx' : idx}
		else:
			image_idx, label = self._get_meta_data(idx)
			image = self._get_full_image(idx)
			if self.transformations:
				image = self.transformations(image)
			sample = {'image': image, 'label': label, 'idx' : idx}

		return sample

	def __str__(self):
		string = ""
		string += "total : " + str(len(self)) + "\n"
		string += "classes : " + str(self.classes) + "(type: " + str(type(self.classes)) + ")\n"
		if self.diagnosis_code:
			string += "diagnosis_code : " + str(self.diagnosis_code) + "\n"
		string += "file type : " + self.file_type

		return string


def load_sample(path_file, file_type):
	if file_type == "nifti":
		sample = torch.tensor(nib.load(path_file.strip()).get_data()).double()[11:108,13:136,0:107]
	elif file_type == "fm":
		sample = torch.tensor(load_progress(path_file))
	else:
		sample = torch.tensor(cv2.imread(path_file.strip(), 0)).double()
	return sample.clone()

def load_samples(path_file, file_type):
	if type(path_file) ==  str:
		sample = load_sample(path_file, file_type)
	else:
		sample = load_sample(path_file[0], file_type)
		for idx in range(1,len(path_file)):
			s = load_sample(path_file[idx], file_type)
			sample = torch.cat((sample, s),0)
	return sample.clone()


def is_mnist(dataset_path):
	if "mnist_" in dataset_path.lower():
		return True
	else:
		return False

def get_dataset(dataset_path, n_samples, get_labels=False, paths_only=False,seed=-1, shuffle=True):
	print("Dataset path is: ", dataset_path)
	if is_mnist(dataset_path):
		if "test" not in dataset_path:
			print("TRAIN")
			data = mnist.train_images()
			labels = mnist.train_labels()
		else:
			print("TEST")
			data = mnist.test_images()
			labels = mnist.test_labels()
		dataset_size = data.shape[0]
		if (n_samples > dataset_size) or (n_samples == 0):
			n_samples = dataset_size
		if shuffle:
			if seed >= 0:
				random.seed(seed)
			sample = random.sample(range(dataset_size), n_samples)
		else:
			sample = list(range(n_samples))
		data = torch.tensor(data[sample, :])
		labels = torch.tensor(labels[sample]).long()
	else:
		if os.path.isdir(dataset_path):
			files = [os.path.join(dataset_path, f)  for f in os.listdir(dataset_path)]
		else:
			paths_file = open(dataset_path, "r") 
			files = paths_file.readlines()
			paths_file.close()
		files.sort()
		dataset_size = len(files)
		if (n_samples > dataset_size) or (n_samples==0):
			n_samples = dataset_size
		if shuffle:
			if seed >= 0:
				random.seed(seed)
			samples = random.sample(range(dataset_size), n_samples)
		else:
			samples = list(range(n_samples))
		labels = []
		data_samples = []
		for i in samples:
			data_samples.append(files[i].strip())
			if os.path.isdir(dataset_path):
				labels.append(get_true_type(files[i].strip().split('_')[-1].split('.')[0]))
			else:
				labels.append(get_true_type(files[i].strip().split(os.sep)[-2]))
		#labels = np.array(labels)
		if paths_only:
			return data_samples, labels
		# Identifying the type of data to work with -- MRI or images
		file_type, __ = get_data_type(files[0])
		input_shape = load_sample(files[0], file_type).shape
		data = torch.zeros([n_samples] + list(input_shape)).double()
		c = 0
		for i in samples:
			img = load_sample(files[i], file_type)
			data[c,:] = img
			c += 1
	if get_labels:
		return data, labels
	else:
		return data

def get_shape(dataset_path):
	data = get_dataset(dataset_path, 1, paths_only=False)
	return data.shape[1:]