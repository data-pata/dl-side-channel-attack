import os.path
import sys
import h5py # as h5py??
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model, Sequential
from keras.layers import Dense, InputLayer, BatchNormalization, ReLU, Softmax
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras_applications.imagenet_utils import _obtain_input_shape
from keras.optimizers import Nadam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import to_categorical
from keras.models import load_model
import time

def create_model(classes=2, input_size=150):
	input_shape = (input_size,)

	# Create model.
	model = Sequential()
	model.add(BatchNormalization(input_shape=input_shape))

	model.add(Dense(4, kernel_initializer='he_uniform', input_shape=input_shape))
	model.add(BatchNormalization())
	model.add(ReLU())

	#model.add(Dense(2, kernel_initializer='he_uniform', input_shape=input_shape))
	#model.add(BatchNormalization())
	#model.add(ReLU())

	model.add(Dense(classes, kernel_initializer='he_uniform', input_dim=32))
	model.add(Softmax())

	optimizer = Nadam(lr=0.001, epsilon=1e-08)
	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

	# print(model.summary())
	return model

def check_file_exists(file_path):
	if os.path.exists(file_path) == False:
		print("Error: provided file path '%s' does not exist!" % file_path)
		sys.exit(-1)
	return

def load_traces(from_trace, to_trace):
	#Import our traces

	traces = np.load(data_folder + 'trace' + str(from_trace) + '.npy')
	labels = np.load(data_folder + 'label' + str(from_trace)+ '.npy')

	for i in range(from_trace+1, to_trace):  
		tempTrace = np.load(data_folder+'trace'+str(i)+'.npy')
		tempLabel = np.load(data_folder+'label'+str(i)+'.npy')

		traces = np.append(traces,tempTrace,axis=0)
		labels = np.append(labels,tempLabel,axis=0)
	
	print('traces shape:', traces.shape)
	print('labels shape:', labels.shape)

	delimitedTraces = np.zeros(traces.shape)
	for x_index in range(traces.shape[0]):
		delimitedTraces[x_index,:] = -1+(traces[x_index,:]-np.min(traces[x_index,:]))*2/(np.max(traces[x_index,:])-np.min(traces[x_index,:]))

	return (delimitedTraces[:,:], labels)
	###############################################################

def train_model(X_profiling, Y_profiling, model, save_file_name, epochs=50, batch_size=64):
	check_file_exists(os.path.dirname(save_file_name))
	# Save model every epoch
	save_model = ModelCheckpoint(save_file_name+'.h5',monitor='val_accuracy',verbose=1,save_best_only=True,mode='max')
	# Get the input layer shape
	input_layer_shape = model.get_layer(index=0).input_shape
	# Sanity check
	if input_layer_shape[1] != len(X_profiling[0]):
		print("Error: model input shape %d instead of %d is not expected ..." % (input_layer_shape[1], len(X_profiling[0])))
		sys.exit(-1)
	# Adapt the data shape according our model input
	if len(input_layer_shape) == 2:
		# This is a MLP
		Reshaped_X_profiling = X_profiling
	elif len(input_layer_shape) == 3:
		# This is a CNN: expand the dimensions
		Reshaped_X_profiling = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1], 1))
	else:
		print("Error: model input shape length %d is not expected ..." % len(input_layer_shape))
		sys.exit(-1)

	es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=20)

	history = model.fit(x=Reshaped_X_profiling, y=to_categorical(Y_profiling, num_classes=2), batch_size=batch_size, verbose=0, epochs=epochs, callbacks=[es,save_model], validation_split=0.3)
	
	# summarize history for accuracy
	plt.plot(history.history['accuracy'])
	plt.plot(history.history['val_accuracy'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.savefig('../history/' + modelName + '.pdf')
	plt.show()
	
	return history
	  
#######################################
# TRAINING OPTIONS 

# data_folder = "../traces/puf0_avg100/"
# model_folder = "../models/puf0/"
# modelName = 'puf0_trace_0-1_batchsize_128'
# batch_size = 128
# input_size = 150

# data_folder = "../traces/puf12_avg100/"
# model_folder = "../models/puf12/"
# modelName = 'puf12'
# batch_size = 
# input_size = 150

# data_folder = "../traces/puf0_avg0_blocks_of_1000/"
# model_folder = "../models/puf0_avg0_blocks_of_1000/"
# modelName = 'puf0_blocks_of_1000_trace_0-1_batchsize_64'
# batch_size = 64
# input_size = 150

from_trace = 0;
to_trace = 1;

print('data: {:s}\nfrom trace{:d} to trace{:d} (inclusive)\nbatch size: {:d}\nsave to:{:s}'
	.format(data_folder, from_trace, to_trace-1, batch_size, modelName))
#######################################
# Start of execution, the time parts are there for our own references so we know roughly how long training takes
start = time.time()

# Load the training traces
(traces, labels) = load_traces(from_trace, to_trace)

### MLP training

mlp = create_model(input_size=input_size)

train_model(traces, labels, mlp, model_folder+modelName, epochs=100, batch_size=batch_size) # 32 best batch size

end = time.time()
print("The total running time was: ",((end-start)/60), " minutes.") 
