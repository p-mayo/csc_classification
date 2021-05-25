# Python code to manage miscelaneus functions for progress tracking.
import os 
import pickle 	
import datetime

from time import time

def __init__():
	datetime.datetime.now().strftime(get_str_format())

def save_progress(workspace, path):
	pickle.dump( workspace, open( path, "wb" ) )

def load_progress(path):
	workspace = pickle.load( open( path, "rb" ) )
	return workspace

def log_event(message, path=""):
	print(message)
	if path != "":
		with open(path, "a") as f:
			print("%s" % (message), file=f)

def create_path(path):
	if not os.path.exists(path):
		folders = os.path.split(path)
		create_path(folders[0])
		os.mkdir(path)

def get_str_format():
	time_format = "%Y-%m-%d %H:%M"
	return time_format

def get_time(start_time=None):
	if start_time==None:
		return datetime.datetime.now()
	else:
		return datetime.datetime.now() - start_time

def get_time_fileformat():
	timestamp = datetime.datetime.now()
	return timestamp.strftime("%Y%m%d_%H%M%S")