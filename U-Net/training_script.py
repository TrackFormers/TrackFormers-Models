# Import modules
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import h5py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv
# Import utils
import models, utils
# Set the seed
import random
import gc
random.seed(666)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# General stuff
print("Is PyTorch using GPU?", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
DTYPE = torch.float32
events_per_batch = 1
batch_size = 16


# ========================= Load the data =========================
hf = h5py.File("../../../../../REDVID_Processed/interpolated_helical_50_to_100_tracks_batches_from_0_to_98304__events_per_batch_1_scale_factor_10.0_splitted_binning_30_30_expanded.h5", "r")

X_indices_training = torch.tensor(np.array(hf.get("X_indices_training")), dtype = torch.int32).to(device)
X_features_training = torch.tensor(np.array(hf.get("X_features_training")), dtype = torch.float32).to(device)

Y_indices_training = torch.tensor(np.array(hf.get("Y_indices_training")), dtype = torch.int32).to(device)
Y_features_training = torch.tensor(np.array(hf.get("Y_features_training")), dtype = torch.float32).to(device)
Y_features_binning_training = torch.tensor(np.array(hf.get("Y_features_binning_training")), dtype = torch.float32).to(torch.device("cpu"))

X_indices_validation = torch.tensor(np.array(hf.get("X_indices_validation")), dtype = torch.int32).to(device)
X_features_validation = torch.tensor(np.array(hf.get("X_features_validation")), dtype = torch.float32).to(device)

Y_indices_validation = torch.tensor(np.array(hf.get("Y_indices_validation")), dtype = torch.int32).to(device)
Y_features_validation = torch.tensor(np.array(hf.get("Y_features_validation")), dtype = torch.float32).to(device)
Y_features_binning_validation = torch.tensor(np.array(hf.get("Y_features_binning_validation")), dtype = torch.float32).to(torch.device("cpu"))

X_indices_test = torch.tensor(np.array(hf.get("X_indices_test")), dtype = torch.int32).to(device)
X_features_test = torch.tensor(np.array(hf.get("X_features_test")), dtype = torch.float32).to(device)

Y_indices_test = torch.tensor(np.array(hf.get("Y_indices_test")), dtype = torch.int32).to(device)
Y_features_test = torch.tensor(np.array(hf.get("Y_features_test")), dtype = torch.float32).to(device)
Y_features_binning_test = torch.tensor(np.array(hf.get("Y_features_binning_test")), dtype = torch.float32).to(torch.device("cpu"))

track_ids_training = torch.tensor(np.array(hf.get("track_ids_training")), dtype = torch.int32)
track_ids_validation = torch.tensor(np.array(hf.get("track_ids_validation")), dtype = torch.int32)
track_ids_test = torch.tensor(np.array(hf.get("track_ids_test")), dtype = torch.int32)

spatial_size = list(np.array(hf.get("spatial_size")))

param1_min = torch.tensor(np.array(hf.get("param1_min"))).to(device)
param1_max = torch.tensor(np.array(hf.get("param1_max"))).to(device)
param2_min = torch.tensor(np.array(hf.get("param2_min"))).to(device)
param2_max = torch.tensor(np.array(hf.get("param2_max"))).to(device)

num_bins = int(np.array(hf.get("num_bins")))

hf.close()


spatial_size[0] = spatial_size[0] + 1
spatial_size[1] = spatial_size[1] + 1
spatial_size[2] = spatial_size[2] + 1

print("\n" + "spatial_size: ", spatial_size)

# ========================= PREPROCESSING =========================

print("X_indices_training shape: ", X_indices_training.shape, "X_features_training shape: ", X_features_training.shape)
print("X_indices_validation shape: ", X_indices_validation.shape, "X_features_validation shape: ", X_features_validation.shape)
print("X_indices_test shape: ", X_indices_test.shape, "X_features_test shape: ", X_features_test.shape, "\n")

print("Y_indices_training shape: ", Y_indices_training.shape, "Y_features_training shape: ", Y_features_training.shape)
print("Y_indices_validation shape: ", Y_indices_validation.shape, "Y_features_validation shape: ", Y_features_validation.shape)
print("Y_indices_test shape: ", Y_indices_test.shape, "Y_features_test shape: ", Y_features_test.shape, "\n")

print("track_ids_training shape: ", track_ids_training.shape)
print("track_ids_validation shape: ", track_ids_validation.shape)
print("track_ids_test.shape: ", track_ids_test.shape)


# Generate the corresponding sparse tensors
X_training = spconv.SparseConvTensor(X_features_training, X_indices_training, spatial_size, int(torch.unique(X_indices_training[:,0:1]).shape[0]))
X_validation = spconv.SparseConvTensor(X_features_validation, X_indices_validation, spatial_size, int(torch.unique(X_indices_validation[:,0:1]).shape[0]))
X_test = spconv.SparseConvTensor(X_features_test, X_indices_test, spatial_size, int(torch.unique(X_indices_test[:,0:1]).shape[0]))

Y_training = spconv.SparseConvTensor(Y_features_training, Y_indices_training, spatial_size, int(torch.unique(Y_indices_training[:,0:1]).shape[0]))
Y_validation = spconv.SparseConvTensor(Y_features_validation, Y_indices_validation, spatial_size, int(torch.unique(Y_indices_validation[:,0:1]).shape[0]))
Y_test = spconv.SparseConvTensor(Y_features_test, Y_indices_test, spatial_size, int(torch.unique(Y_indices_test[:,0:1]).shape[0]))

Y_training_binning = spconv.SparseConvTensor(Y_features_binning_training, Y_indices_training, spatial_size, int(torch.unique(Y_indices_training[:,0:1]).shape[0]))
Y_validation_binning = spconv.SparseConvTensor(Y_features_binning_validation, Y_indices_validation, spatial_size, int(torch.unique(Y_indices_validation[:,0:1]).shape[0]))
Y_test_binning = spconv.SparseConvTensor(Y_features_binning_test, Y_indices_test, spatial_size, int(torch.unique(Y_indices_test[:,0:1]).shape[0]))

# Extract the batches that we have for each set
batches_train = list(torch.unique(X_training.indices[:,0:1]).detach().cpu().numpy())
batches_val = list(torch.unique(X_validation.indices[:,0:1]).detach().cpu().numpy())
batches_test = list(torch.unique(X_test.indices[:,0:1]).detach().cpu().numpy())

# ========================= MODEL AND TRAINING =========================

# Define the model
model = models.UNet_SCN_classifier(in_channels = 1, out_channels = num_bins, DTYPE = DTYPE, device = device, kernel_size = 3, spatial_size = spatial_size, scale_factor = 1/4).to(device)

# Define learning rate and optim
lr_w = 1e-3
optimizer = torch.optim.AdamW( model.parameters(), lr = lr_w )
model.optimizer = optimizer
model.param1_interval = [param1_min.item(), param1_max.item()]
model.param2_interval = [param2_min.item(), param2_max.item()]
model.n1, model.n2 = 30, 30

print("Starting optimization...")
epochs = 401
pbar = tqdm(range(epochs))
for epoch in pbar:
	model.epoch = epoch
	# Sloop over batches
	loss_training_batch, loss_val_batch = [], []
	accuracy_training_batch, accuracy_val_batch = [], []
	pbar_batch = tqdm(range( int(len(batches_train) / batch_size) ))
	for i in pbar_batch:
		# Filter the training dataset to that specific batch
		list_batches = batches_train[i*batch_size:(i+1)*batch_size]
		X_training_filtered, Y_training_filtered, Y_training_binning_filtered, track_ids_training_filtered = utils.filter_several_batches_sparse(X_training, Y_training, Y_training_binning, track_ids_training, list_batches, batch_size, spatial_size, device)
		# Do the train step and save the loss
		loss_training = model.train_step(x = X_training_filtered, y_true = Y_training_binning_filtered)
		loss_training_batch.append(loss_training.item())
		pbar_batch.set_postfix({'loss_training_batch' : loss_training.item()})
	model.loss_training_hist.append( float(np.mean(np.array(loss_training_batch))) )
	# In order to compute the trackML score, we need to filter just ONE single event.
	X_single_event_training, Y_single_event_training, Y_binning_single_event_training, track_ids_single_event_training = utils.filter_single_event_sparse_also_track_IDs(X_training, Y_training, Y_training_binning, track_ids_training, batches_train[0], spatial_size, events_per_batch, device)
	accuracy_training = model.compute_accuracy(x = X_single_event_training, y_true = Y_binning_single_event_training)
	trackML_training = model.compute_trackML(x = X_single_event_training, track_id = track_ids_single_event_training)
	model.trackML_training_hist.append(trackML_training)
	model.accuracy_training_hist.append(accuracy_training.item())
	
	# For the validation part we will do it in a similar way.
	pbar_batch = tqdm(range( int(len(batches_val) / batch_size) ))
	for i in pbar_batch:
		list_batches = batches_val[i*batch_size:(i+1)*batch_size]
		X_validation_filtered, Y_validation_filtered, Y_validation_binning_filtered, track_ids_validation_filtered = utils.filter_several_batches_sparse(X_validation, Y_validation, Y_validation_binning, track_ids_validation, list_batches, batch_size, spatial_size, device)
		loss_validation = model.compute_loss(x = X_validation_filtered, y_true = Y_validation_binning_filtered)
		loss_val_batch.append(loss_validation.item())
		pbar_batch.set_postfix({'loss_validation_batch' : loss_validation.item()})
	model.loss_validation_hist.append( float(np.mean(np.array(loss_val_batch))) )
	X_single_event_validation, Y_single_event_validation, Y_binning_single_event_validation, track_ids_single_event_validation = utils.filter_single_event_sparse_also_track_IDs(X_validation, Y_validation, Y_validation_binning, track_ids_validation, batches_val[0], spatial_size, events_per_batch, device)
	accuracy_validation = model.compute_accuracy(x = X_single_event_validation, y_true = Y_binning_single_event_validation)
	trackML_validation = model.compute_trackML(x = X_single_event_validation, track_id = track_ids_single_event_validation)
	model.trackML_validation_hist.append(trackML_validation)
	model.accuracy_validation_hist.append(accuracy_validation.item())
	
	# Free cache
	torch.cuda.empty_cache()
	gc.collect()












































































