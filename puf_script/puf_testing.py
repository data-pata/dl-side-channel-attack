import os.path
import sys
import h5py
import random
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import time
from keras.models import Model, Sequential
from keras.layers import Flatten, Dense, Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D, AveragePooling1D
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras_applications.imagenet_utils import _obtain_input_shape
from keras.optimizers import Nadam, RMSprop
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from keras.models import load_model
import os
import fnmatch
 

##################################################
input_size = 150
 
data_folder = "../traces/puf0_avg100/"
model_folder = "../models/puf0/"
modelName = 'puf0'

#data_folder = "../traces/puf12_avg100/"
#model_folder = "../models/puf12/"
#modelName = 'puf12'											#

################################################### 

def load_sca_model(model_folder,model_file):
	for f_name in os.listdir(model_folder):
		if f_name.startswith(model_file):
			model = load_model(model_folder+f_name)
			print(f_name)
			#print(model.summary())
		else:
			continue
	return model

def rank_func(model, dataset, labels):

	# Get the input layer shape
	input_layer_shape = model.get_layer(index=0).input_shape

	if len(input_layer_shape) == 2:
		# This is a MLP
		input_data = dataset
	elif len(input_layer_shape) == 3:
		# This is a CNN: reshape the data
		input_data = dataset
		input_data = input_data.reshape((input_data.shape[0], input_data.shape[1], 1))
	else:
		print("Error: model input shape length %d is not expected ..." % len(input_layer_shape))
		sys.exit(-1)

	# Predict probabilities for 2-node output case
	predictions = model.predict(input_data)
	predictions = np.argmax(predictions,axis=1)

	# xor = 0 if the same
	ranks_prd = np.bitwise_xor(predictions, labels)

	return (ranks_prd)

# Load testing data
def load_traces(): 

	traces = np.load(data_folder+'trace9.npy')#[0:1000,:] 
	labels = np.load(data_folder+'label9.npy')#[0:1000]
	
	print('traces shape:', traces.shape)
	print('labels shape:', labels.shape)

	delimitedTraces = np.zeros(traces.shape)
	for x_index in range(traces.shape[0]):
		delimitedTraces[x_index,:] = -1+(traces[x_index,:]-np.min(traces[x_index,:]))*2/(np.max(traces[x_index,:])-np.min(traces[x_index,:]))

	return (delimitedTraces, labels)

# Check a saved model against test traces
def check_model(model_folder,model,count):
	# Load attack data
	(traces, labels) = load_traces()

	# Load model
	model_load = load_sca_model(model_folder,model)

	# coorect if 0
	num_attack = int(labels.shape[0])
	ranks_prd = np.zeros((num_attack, 1))
	ranks_prd = rank_func(model_load, traces, labels)

	a = 1-np.mean(ranks_prd)
	print('test accuracy:', a)
	print('mean of lables:', np.mean(labels))

	return
	
start = time.time()
print('input size =', input_size)
        					
count = 0
for file_name in os.listdir(model_folder):
	if fnmatch.fnmatch(file_name, 'puf0.h5'):
		model = file_name
		count = count + 1            												 																						#
		check_model(model_folder,model,count)

end = time.time()
print("The total running time was: ",((end-start)/60), " minutes.") 

