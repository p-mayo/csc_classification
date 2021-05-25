# Python class to have a better handling of the parameters
# the user can specify for the execution of the model

import xml.etree.ElementTree as ET

keys = set(["prior", "num_samples", "num_filters", "lmbda", "param", "lr_z", "lr_d", "max_outer", "max_inner", "batch_size",
	    "depth", "activation", "dropout", "epochs", "learning_rate", "weight_decay", "optimizer", "kernel", "C", "gamma"])

class Parameters:
	def __init__(self, l_type):
		self.output_dir = ""
		self.train_input_dir = ""
		self.test_input_dir = ""
		self.log_dir = ""
		self.l_type = l_type
		self.gpu = False
		self.transformations = None
		self.prior = "raw"
		self.diagnosis_code = None
		self.batch_size = 12
		self.num_workers = 1
		if l_type == "csc":
			self.init_csc_params()
		elif l_type == "clf":
			self.init_clf_params()
		elif l_type == "pool":
			self.init_pool_params()
		elif l_type == "svm":
			self.init_svm_params()

	def init_csc_params(self):
		self.prior = "cauchy"
		self.num_filters = 16
		self.filter_size = 5
		self.num_samples = 0
		self.lmbda = 1.
		self.param = 0
		self.lr_z = 0.015
		self.lr_d = 0.02
		self.start = 0
		self.max_outer = 10
		self.max_inner = 10
		self.learn_bias = False
		self.adapt_lr = True
		self.learn_filters = True
		self.initial_f = ""
		self.in_channels = 1
		self.approximate = False

	def init_clf_params(self):
		self.depth = 3
		self.order="DFAFAFS"
		self.n_units = [1300, 50]
		self.classes = []
		self.num_samples = 0
		self.activation = "relu"
		self.dropout = 0.5
		self.epochs = 20
		self.learning_rate = 1e-4
		self.weight_decay = 0
		self.optimizer = "Adam"
		self.accumulation_steps = 1
		self.patience = 10
		self.tolerance = 0.05
		self.evaluation_steps = 0
		self.selection_threshold = 0.0
		self.initial_f = ""
		self.learn_weights = True
		self.learn_bias = True

	def init_svm_params(self):
		self.kernel = "rbf"
		self.C = 1
		self.gamma = 1e-3
		self.num_samples = 0

	def init_pool_params(self):
		self.filter_size = 5
		self.stride = 1

	def init_dropout_params(self):
		self.probability = 0.5

	def __str__(self):
		params = [p for p in dir(self) if not (p.startswith("__") or p.startswith("init"))]
		string = ""
		for p in params:
			string += '\t'+ p + ' : ' + str(getattr(self, p)) + '\n'
		return string

	def __iter__(self):
		params = [p for p in dir(self) if not (p.startswith("__") or p.startswith("init"))]
		for p in params:
			yield(p, getattr(self, p))

def get_true_type(string):
	if string == None:
		return ""
	elif string.isnumeric():
		return int(string)
	elif "." in string and string.replace('.', '').isnumeric():
		return float(string)
	elif ":" in string and "c:" not in string.lower():
		diagnosis_code = {}
		for items in string.split(','):
			k,v = items.split(':')
			diagnosis_code[k] = int(v)
		return diagnosis_code
	elif "," in string:
		return [get_true_type(x) for x in string.split(',')]
	elif string.lower() == "true":
		return True
	elif string.lower() == "false":
		return False
	else:
		return string

def get_params(xml_path):
	tree = ET.parse(xml_path)
	root = tree.getroot()
	params = []
	conv_layers = []
	pool_layers = []
	for child in root:
		params_l = Parameters(child.attrib['name'])		
		for param in child:
			if "conv" in param.tag:
				conv_param = {}
				for p in param:
					conv_param[p.tag] = get_true_type(p.text)
				conv_layers.append(conv_param)
			elif "pool" in param.tag:
				pool_param = {}
				for p in param:
					pool_param[p.tag] = get_true_type(p.text)
				pool_layers.append(pool_param)
			else:
				if hasattr(params_l, param.tag) and param.text:
					setattr(params_l, param.tag, get_true_type(param.text))
		if child.attrib['name'] == "clf":
			setattr(params_l, "conv", conv_layers)
			setattr(params_l, "pool", pool_layers)
		params.append(params_l)
	return params

def log_params(writer, params, metrics, tag=""):
	keys_to_log = set(dict(params).keys()) & keys
	params_to_log = {}
	for key in keys_to_log:
		params_to_log[key] = getattr(params, key)
	writer.add_hparams(params_to_log, metrics, tag)