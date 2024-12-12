import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import matplotlib.colors as mcolors
from collections import Counter





def compute_density_map(Y_binning, n1, n2):
	"""
	Computes a density map (heatmap) by binning data into a 2D grid based on input values.
	
	This function takes an input tensor `Y_binning` representing bin indices and computes 
	a heatmap with dimensions `(n1, n2)`. Each cell in the heatmap represents the count 
	of elements in `Y_binning` that fall into the respective bin. The heatmap is normalized 
	by the maximum count to scale values between 0 and 1.
	
	Args:
	    Y_binning (torch.Tensor): Tensor of binned data with bin indices for each element.
	    n1 (int): Number of rows in the output heatmap.
	    n2 (int): Number of columns in the output heatmap.
	
	Returns:
	    torch.Tensor: A normalized 2D heatmap tensor of shape `(n1, n2)`.
	"""
	# Define empty heatmap.
	heatmap = torch.zeros((n1,n2))
	# Run over all bins.
	for n_row in tqdm(range(n1)):
		for n_col in range(n2):
			heatmap[n_row, n_col] = (Y_binning == n_col + n2 * n_row).sum()
	heatmap = heatmap / torch.max(heatmap)
	return heatmap







class WeightedCrossEntropyLoss(nn.Module):
	"""
	Custom weighted cross-entropy loss function.
	
	This class implements a weighted version of the cross-entropy loss function, which 
	applies different loss weights to different classes during training. It extends 
	the PyTorch `nn.Module` class and allows optional weighting of the loss based on 
	the input class distribution.
	
	Args:
	    weight (torch.Tensor, optional): A tensor of weights to apply to each class. 
	        If None, no weighting is applied. Default is None.
	
	Methods:
	    forward(output, target):
	        Compute the weighted cross-entropy loss for a batch of predictions.
	"""
	def __init__(self, weight=None):
		super(WeightedCrossEntropyLoss, self).__init__()
		self.weight = weight
	def forward(self, output, target):
		# Calculate the logarithm of the softmax of the output
		log_softmax_output = torch.log_softmax(output, dim=1)
		# Extract the log probabilities corresponding to the target classes
		log_probabilities = log_softmax_output.gather(1, target.argmax(dim=1, keepdim=True))
		
		# Apply class weights if provided
		if self.weight is not None:
			weight = self.weight[target.argmax(dim=1)]
			weighted_loss = -log_probabilities.squeeze() * weight
		else:
			weighted_loss = -log_probabilities.squeeze()
		
		# Calculate the average loss
		loss = weighted_loss.mean()
		
		return loss







class FocalLoss(nn.Module):
	"""
	Focal Loss function for addressing class imbalance in classification tasks.
	
	This loss function is a modified version of cross-entropy loss that applies a 
	modulating factor to focus more on hard-to-classify examples, making it effective 
	for dealing with class imbalance. It is parameterized by an `alpha` value to balance 
	the importance of each class and a `gamma` value to control the focusing effect.
	
	Args:
	    alpha (float, optional): A scalar factor to balance the importance of different 
	        classes. Default is 1.
	    gamma (float, optional): Focusing parameter that adjusts the rate at which 
	        easy-to-classify examples are down-weighted. Default is 2.
	    reduction (str, optional): Specifies the reduction to apply to the output: 
	        `'none'`, `'mean'`, or `'sum'`. Default is `'mean'`.
	
	Methods:
	    forward(pred, target):
	        Compute the focal loss for a batch of predictions and target labels.
	"""
	def __init__(self, alpha=1, gamma=2, reduction='mean'):
		super(FocalLoss, self).__init__()
		self.alpha = alpha
		self.gamma = gamma
		self.reduction = reduction
	def forward(self, pred, target):
		# Convert target to one-hot encoding.
		target_onehot = F.one_hot(target.squeeze(), num_classes=pred.size(1)).float()
		# Calculate cross entropy.
		ce_loss = F.cross_entropy(pred, target.squeeze(), reduction='none')
		# Calculate focal loss.
		focal_loss = torch.pow(1 - torch.exp(-ce_loss), self.gamma) * ce_loss
		
		return focal_loss.mean()







class FocalTverskyLoss(nn.Module):
	"""
	Focal Tversky Loss function for imbalanced classification tasks, particularly in 
	segmentation problems. It combines the Tversky index with focal loss to focus 
	on hard-to-classify regions and mitigate the effect of class imbalance.
	
	The Tversky index is a generalization of the Dice coefficient, controlled by a parameter `alpha`. 
	Focal Tversky Loss adds a focusing parameter `gamma` to control how much emphasis is put on difficult examples.
	
	Args:
	    alpha (float): Controls the weight between false negatives and false positives. 
	        A higher `alpha` gives more weight to false negatives.
	    gamma (float): Focusing parameter that adjusts the impact of easy-to-classify 
	        examples. A higher `gamma` focuses more on hard examples.
	
	Methods:
	    forward(pred, target, density_map):
	        Compute the Focal Tversky loss for a batch of predictions, targets, 
	        and the given density map.
	"""
	def __init__(self, alpha, gamma):
		super(FocalTverskyLoss, self).__init__()
		self.alpha = alpha
		self.gamma = gamma
	def forward(self, pred, target, density_map):
		density_map = density_map / density_map.max()
		# Convert target to one-hot encoding and convert it to flatten.
		target_onehot = F.one_hot(target.squeeze(), num_classes=pred.size(1)).float()
		probs = nn.Softmax(dim=1)(pred)
		
		tp = torch.sum(torch.mul(target_onehot * probs, density_map.unsqueeze(0)), dim = 0)
		fn = torch.sum(torch.mul(target_onehot * (1 - probs), density_map.unsqueeze(0)), dim = 0)
		fp = torch.sum(torch.mul((1 - target_onehot) * probs, density_map.unsqueeze(0)), dim = 0)
		
		#tp = torch.sum(target_onehot * probs)
		#fn = torch.sum(target_onehot * (1 - probs))
		#fp = torch.sum((1 - target_onehot) * probs)
		
		tversky = (tp + 1e-7) / (tp + self.alpha * fn + (1 - self.alpha) * fp + 1e-7)
		tversky = torch.clamp(tversky, min = 0.0, max = 1.0)
		focal_tversky = (1 - tversky) ** (1 / self.gamma)
		
		loss = torch.mean(focal_tversky)
		
		return loss






def replace_feature(out, new_features):
	"""
	Replace the feature tensor of a given output object, handling both spconv 1.x and 2.x behavior.
	
	In spconv 2.x, the `replace_feature()` method is used to replace the features of a sparse tensor.
	For earlier versions, the features are replaced directly by assigning to the `features` attribute.
	
	Args:
	    out (spconv.SparseConvTensor): The sparse tensor whose features need to be replaced. 
	        It could either be from spconv 1.x or 2.x.
	    new_features (torch.Tensor): The new feature tensor to replace the existing features of `out`.
	
	Returns:
	    spconv.SparseConvTensor: The output tensor with the updated features.
	"""
	if "replace_feature" in out.__dir__():
		# spconv 2.x behaviour
		return out.replace_feature(new_features)
	else:
		out.features = new_features
		return out






def filter_single_event_sparse(X, Y, n, spatial_size, events_per_batch):
	"""
	Filters the input sparse tensors `X` and `Y` to extract features corresponding to a specific event, `n`, 
	and adjusts the batch indices accordingly.
	
	This function processes sparse tensors, `X` and `Y`, containing multiple events and filters them 
	to isolate the data corresponding to a single event. The batch index of the filtered data is set to zero.
	
	Args:
	    X (spconv.SparseConvTensor): Sparse tensor `X` containing input features and indices for multiple events.
	    Y (spconv.SparseConvTensor): Sparse tensor `Y` containing target features and indices for multiple events.
	    n (int): The index of the event to filter.
	    spatial_size (list or tuple of ints): The spatial dimensions of the sparse tensor.
	    events_per_batch (int): Number of events per batch, used in constructing the output sparse tensor.
	
	Returns:
	    tuple: A tuple containing:
	        - **X_filtered (spconv.SparseConvTensor)**: Filtered sparse tensor `X` for event `n`.
	        - **Y_filtered (spconv.SparseConvTensor)**: Filtered sparse tensor `Y` for event `n`.
	"""
	X_indices, X_features = X.indices, X.features
	Y_indices, Y_features = Y.indices, Y.features
	
	X_indices_filtered, X_features_filtered = X_indices[(X_indices[:,0:1] == n).view(-1),:], X_features[(X_indices[:,0:1] == n).view(-1),:]
	Y_indices_filtered, Y_features_filtered = Y_indices[(X_indices[:,0:1] == n).view(-1),:], Y_features[(X_indices[:,0:1] == n).view(-1),:]
	
	X_indices_filtered[:,0:1] = 0
	Y_indices_filtered[:,0:1] = 0
	
	X_filtered = spconv.SparseConvTensor(X_features_filtered, X_indices_filtered, spatial_size, events_per_batch)
	Y_filtered = spconv.SparseConvTensor(Y_features_filtered, Y_indices_filtered, spatial_size, events_per_batch)
	
	return X_filtered, Y_filtered







def filter_several_batches_sparse(X, Y, Y_binning, track_ids, list_indices, batch_size, spatial_size, device):
	X_indices, X_features = X.indices.to(device), X.features.to(device)
	Y_indices, Y_features = Y.indices.to(device), Y.features.to(device)
	Y_features_binning = Y_binning.features.to(device)
	track_ids = track_ids.to(device)
	
	mask = (X_indices[:, 0].unsqueeze(1) == torch.tensor(list_indices, device=X_indices.device)).any(dim=1)
	
	X_indices_filtered, X_features_filtered = X_indices[mask].to(torch.int32), X_features[mask]
	Y_indices_filtered, Y_features_filtered = Y_indices[mask].to(torch.int32), Y_features[mask]
	Y_features_binning_filtered = Y_features_binning[mask]
	track_ids_filtered = track_ids[mask].to(torch.int32)
	
	X_indices_filtered = reindex_indices(X_indices_filtered, 0, batch_size)
	Y_indices_filtered = reindex_indices(Y_indices_filtered, 0, batch_size)
	
	
	X_filtered = spconv.SparseConvTensor(X_features_filtered, X_indices_filtered, spatial_size, batch_size)
	Y_filtered = spconv.SparseConvTensor(Y_features_filtered, Y_indices_filtered, spatial_size, batch_size)
	Y_binning_filtered = spconv.SparseConvTensor(Y_features_binning_filtered, Y_indices_filtered, spatial_size, batch_size)
	
	return X_filtered, Y_filtered, Y_binning_filtered, track_ids_filtered






def filter_single_event_sparse_also_track_IDs(X, Y, Y_binning, track_ids, n, spatial_size, events_per_batch, device):
	"""
	Filters the input sparse tensors `X`, `Y`, and `Y_binning` to extract features corresponding to a 
	specific event, `n`, along with the associated track IDs.
	
	This function processes multiple sparse tensors containing various events and filters them 
	to isolate the data related to a single event. The batch indices of the filtered data are set to zero.
	
	Args:
	    X (spconv.SparseConvTensor): Sparse tensor `X` containing input features and indices for multiple events.
	    Y (spconv.SparseConvTensor): Sparse tensor `Y` containing target features and indices for multiple events.
	    Y_binning (spconv.SparseConvTensor): Sparse tensor `Y_binning` containing features for the binning of `Y`.
	    track_ids (torch.Tensor): Tensor containing track IDs corresponding to the events.
	    n (int): The index of the event to filter.
	    spatial_size (list or tuple of ints): The spatial dimensions of the sparse tensor.
	    events_per_batch (int): Number of events per batch, used in constructing the output sparse tensor.
	    device (torch.device): The device (CPU or GPU) on which to perform the operations.
	
	Returns:
	    tuple: A tuple containing:
	        - **X_filtered (spconv.SparseConvTensor)**: Filtered sparse tensor `X` for event `n`.
	        - **Y_filtered (spconv.SparseConvTensor)**: Filtered sparse tensor `Y` for event `n`.
	        - **Y_binning_filtered (spconv.SparseConvTensor)**: Filtered sparse tensor `Y_binning` for event `n`.
	        - **track_ids_filtered (torch.Tensor)**: Filtered track IDs corresponding to event `n`.
	"""
	X_indices, X_features = X.indices.to(device), X.features.to(device)
	Y_indices, Y_features = Y.indices.to(device), Y.features.to(device)
	track_ids = track_ids.to(device)
	Y_binning_features = Y_binning.features.to(device)
	
	X_indices_filtered, X_features_filtered = X_indices[(X_indices[:,0:1] == n).view(-1),:], X_features[(X_indices[:,0:1] == n).view(-1),:]
	Y_indices_filtered, Y_features_filtered = Y_indices[(X_indices[:,0:1] == n).view(-1),:], Y_features[(X_indices[:,0:1] == n).view(-1),:]
	Y_binning_features_filtered = Y_binning_features[(X_indices[:,0:1] == n).view(-1),:]
	track_ids_filtered = track_ids[(X_indices[:,0:1] == n).view(-1),:]
	
	X_indices_filtered[:,0:1] = 0
	Y_indices_filtered[:,0:1] = 0
	
	X_filtered = spconv.SparseConvTensor(X_features_filtered, X_indices_filtered, spatial_size, events_per_batch)
	Y_filtered = spconv.SparseConvTensor(Y_features_filtered, Y_indices_filtered, spatial_size, events_per_batch)
	Y_binning_filtered = spconv.SparseConvTensor(Y_binning_features_filtered, Y_indices_filtered, spatial_size, events_per_batch)
	
	return X_filtered, Y_filtered, Y_binning_filtered, track_ids_filtered






def reindex_indices(x, index_initial, index_final):
	"""
	Reindexes the first column of a tensor `x` based on unique values in that column.
	
	This function takes a tensor `x`, identifies unique indices in its first column, 
	and reindexes them within the specified range from `index_initial` to `index_final`. 
	Each unique index in `x` will be replaced with a new index corresponding to its 
	position in the specified range.
	
	Args:
	    x (torch.Tensor): A 2D tensor where the first column contains indices to be reindexed.
	    index_initial (int): The initial value of the new index range.
	    index_final (int): The final value (exclusive) of the new index range.
	
	Returns:
	    torch.Tensor: A new tensor with the same shape as `x`, but with the first column reindexed.
	
	Note:
	    The function assumes that the number of unique indices in the first column of `x` 
	    does not exceed the range from `index_initial` to `index_final`.
	"""
	list_known_indices = []
	indices = torch.unique(x[:,0:1])
	
	filtered_list = []
	for i in range(index_initial, index_final):
		filtered = x[x[:,0] == list(indices.detach().cpu().numpy())[i],:]
		filtered[:,0:1] = i
		filtered_list.append(filtered)
	
	x = torch.cat(filtered_list, dim = 0)
	return x






def scale_MinMax(Y_features, params_scale = [1.0, 1.0]):
	"""
	Scale the features of a tensor using Min-Max normalization.
	
	This function takes a 2D tensor `Y_features` and scales its first two columns 
	(assumed to be radial and pitch coefficients) to a specified range 
	using Min-Max normalization. The scaling is controlled by the `params_scale` 
	parameter, which determines the desired scale for each coefficient.
	
	Args:
	    Y_features (torch.Tensor): A 2D tensor where the first two columns represent 
	                                the features to be scaled.
	    params_scale (list, optional): A list containing two scaling factors for the 
	                                    radial and pitch coefficients, respectively. 
	                                    Defaults to [1.0, 1.0].
	
	Returns:
	    torch.Tensor: A 2D tensor with the same number of rows as `Y_features` and 
	                  the first two columns scaled to the specified range.
	
	Note:
	    The function assumes that the input tensor has at least two columns. 
	    It also assumes that the maximum and minimum values of each coefficient are 
	    not equal to avoid division by zero.
	"""
	radial_coeff, pitch_coeff = Y_features[:,0:1], Y_features[:,1:2]
	
	radial_coeff = params_scale[0] * (radial_coeff - radial_coeff.min()) / (radial_coeff.max() - radial_coeff.min())
	pitch_coeff = params_scale[1] * (pitch_coeff - pitch_coeff.min()) / (pitch_coeff.max() - pitch_coeff.min())
	
	## Rearange again the features
	Y_features = torch.cat( (radial_coeff, pitch_coeff), dim = 1 )
	
	return Y_features







def scale_Standard(Y_features_training, Y_features_validation, Y_features_test):
	"""
	Standardize the features of training, validation, and test datasets.
	
	This function applies standardization (z-score normalization) to the features of 
	the input tensors. Each feature (radial and pitch coefficients) is transformed 
	to have a mean of 0 and a standard deviation of 1 based on the training data.
	
	Args:
	    Y_features_training (torch.Tensor): A 2D tensor containing training features, 
	                                         where the first two columns are assumed 
	                                         to be radial and pitch coefficients.
	    Y_features_validation (torch.Tensor): A 2D tensor containing validation features, 
	                                           structured similarly to the training data.
	    Y_features_test (torch.Tensor): A 2D tensor containing test features, structured 
	                                     similarly to the training data.
	
	Returns:
	    tuple: A tuple containing three tensors:
	        - Y_features_training (torch.Tensor): The standardized training features.
	        - Y_features_validation (torch.Tensor): The standardized validation features.
	        - Y_features_test (torch.Tensor): The standardized test features.
	
	Note:
	    The standardization is based solely on the training data. 
	    It assumes that the input tensors have at least two columns for radial 
	    and pitch coefficients. 
	"""
	radial_coeff_training, pitch_coeff_training = Y_features_training[:,0:1], Y_features_training[:,1:2]
	radial_coeff_validation, pitch_coeff_validation = Y_features_validation[:,0:1], Y_features_validation[:,1:2]
	radial_coeff_test, pitch_coeff_test = Y_features_test[:,0:1], Y_features_test[:,1:2]
	
	radial_coeff_training = (radial_coeff_training - torch.mean(radial_coeff_training)) / torch.std(radial_coeff_training)
	radial_coeff_validation = (radial_coeff_validation - torch.mean(radial_coeff_validation)) / torch.std(radial_coeff_validation)
	radial_coeff_test = (radial_coeff_test - torch.mean(radial_coeff_test)) / torch.std(radial_coeff_test)
	
	pitch_coeff_training = (pitch_coeff_training - torch.mean(pitch_coeff_training)) / torch.std(pitch_coeff_training)
	pitch_coeff_validation = (pitch_coeff_validation - torch.mean(pitch_coeff_validation)) / torch.std(pitch_coeff_validation)
	pitch_coeff_test = (pitch_coeff_test - torch.mean(pitch_coeff_test)) / torch.std(pitch_coeff_test)
	
	## Rearange again the features
	Y_features_training = torch.cat( (radial_coeff_training, pitch_coeff_training), dim = 1 )
	Y_features_validation = torch.cat( (radial_coeff_validation, pitch_coeff_validation), dim = 1 )
	Y_features_test = torch.cat( (radial_coeff_test, pitch_coeff_test), dim = 1 )
	
	return Y_features_training, Y_features_validation, Y_features_test






def train_test_val_split_by_batches(X_indices, X_features, Y_indices, Y_features, device):
	"""
	Split input data into training, validation, and test sets by batches of events.
	
	This function divides the input data (features and indices) into three distinct
	sets: training, validation, and test, based on a random sampling of unique 
	event batch IDs. The data is processed in batches to efficiently handle large
	datasets while maintaining the integrity of the splits.
	
	Args:
	    X_indices (torch.Tensor): A tensor containing the indices of the input features.
	    X_features (torch.Tensor): A tensor containing the features of the input data.
	    Y_indices (torch.Tensor): A tensor containing the indices of the target features.
	    Y_features (torch.Tensor): A tensor containing the features of the target data.
	    device (torch.device): The device (CPU or GPU) to which the output tensors should be moved.
	
	Returns:
	    tuple: A tuple containing the training, validation, and test datasets:
	        - X_indices_training (torch.Tensor): Indices for the training dataset.
	        - X_indices_validation (torch.Tensor): Indices for the validation dataset.
	        - X_indices_test (torch.Tensor): Indices for the test dataset.
	        - X_features_training (torch.Tensor): Features for the training dataset.
	        - X_features_validation (torch.Tensor): Features for the validation dataset.
	        - X_features_test (torch.Tensor): Features for the test dataset.
	        - Y_indices_training (torch.Tensor): Target indices for the training dataset.
	        - Y_indices_validation (torch.Tensor): Target indices for the validation dataset.
	        - Y_indices_test (torch.Tensor): Target indices for the test dataset.
	        - Y_features_training (torch.Tensor): Target features for the training dataset.
	        - Y_features_validation (torch.Tensor): Target features for the validation dataset.
	        - Y_features_test (torch.Tensor): Target features for the test dataset.
	
	Note:
	    The function shuffles the unique batch IDs and splits them into training (70%),
	    validation (10%), and test (20%) sets before processing the data in batches.
	"""
	batch_size = 500
	# Split into training, validation and test datasets by batches of events
	unique_batches_IDs = torch.unique(X_indices[:,0:1])
	
	## Define the size of each of the three new lists
	integer_list = list(unique_batches_IDs.detach().cpu().numpy())
	random.shuffle(integer_list)
	size_training = int(len(integer_list) * 0.7)
	size_validation = int(len(integer_list) * 0.1)
	size_test = int(len(integer_list) * 0.2)
	
	## Split the shuffled list into three new lists
	list_training = integer_list[:size_training]
	list_validation = integer_list[size_training:size_training+size_validation]
	list_test = integer_list[size_training+size_validation:]
	
	## Generate the datasets
	list_training_tensor = torch.tensor(list_training, device=X_indices.device)
	list_validation_tensor = torch.tensor(list_validation, device=X_indices.device)
	list_test_tensor = torch.tensor(list_test, device=X_indices.device)
	
	
	# Initialize empty lists to store filtered data
	X_indices_training_list, X_features_training_list, Y_indices_training_list, Y_features_training_list = [], [], [], []
	X_indices_validation_list, X_features_validation_list, Y_indices_validation_list, Y_features_validation_list = [], [], [], []
	X_indices_test_list, X_features_test_list, Y_indices_test_list, Y_features_test_list = [], [], [], []
	
	# Iterate over the data in batches
	for i in tqdm(range(0, len(X_indices), batch_size)):
		# Extract a batch of data
		X_indices_batch = X_indices[i:i+batch_size]
		X_features_batch = X_features[i:i+batch_size]
		Y_indices_batch = Y_indices[i:i+batch_size]
		Y_features_batch = Y_features[i:i+batch_size]
		
		# Compare each element of list_training with the first column of X_indices_batch
		mask_training = (X_indices_batch[:, 0].unsqueeze(1) == list_training_tensor).any(dim=1)  # Use dim=1
		mask_validation = (X_indices_batch[:, 0].unsqueeze(1) == list_validation_tensor).any(dim=1)  # Use dim=1
		mask_test = (X_indices_batch[:, 0].unsqueeze(1) == list_test_tensor).any(dim=1)  # Use dim=1
		
		# Apply the mask to select the desired subset for this batch
		X_indices_training_batch = X_indices_batch[mask_training].to(torch.int32)
		X_features_training_batch = X_features_batch[mask_training]
		Y_indices_training_batch = Y_indices_batch[mask_training].to(torch.int32)
		Y_features_training_batch = Y_features_batch[mask_training]
		
		X_indices_validation_batch = X_indices_batch[mask_validation].to(torch.int32)
		X_features_validation_batch = X_features_batch[mask_validation]
		Y_indices_validation_batch = Y_indices_batch[mask_validation].to(torch.int32)
		Y_features_validation_batch = Y_features_batch[mask_validation]
		
		X_indices_test_batch = X_indices_batch[mask_test].to(torch.int32)
		X_features_test_batch = X_features_batch[mask_test]
		Y_indices_test_batch = Y_indices_batch[mask_test].to(torch.int32)
		Y_features_test_batch = Y_features_batch[mask_test]
		
		# Append the filtered data for this batch to the lists
		X_indices_training_list.append(X_indices_training_batch)
		X_features_training_list.append(X_features_training_batch)
		Y_indices_training_list.append(Y_indices_training_batch)
		Y_features_training_list.append(Y_features_training_batch)
		
		X_indices_validation_list.append(X_indices_validation_batch)
		X_features_validation_list.append(X_features_validation_batch)
		Y_indices_validation_list.append(Y_indices_validation_batch)
		Y_features_validation_list.append(Y_features_validation_batch)
		
		X_indices_test_list.append(X_indices_test_batch)
		X_features_test_list.append(X_features_test_batch)
		Y_indices_test_list.append(Y_indices_test_batch)
		Y_features_test_list.append(Y_features_test_batch)
		
	# Concatenate the lists of filtered data to get the final result
	X_indices_training = torch.cat(X_indices_training_list)
	X_features_training = torch.cat(X_features_training_list)
	Y_indices_training = torch.cat(Y_indices_training_list)
	Y_features_training = torch.cat(Y_features_training_list)
	
	X_indices_validation = torch.cat(X_indices_validation_list)
	X_features_validation = torch.cat(X_features_validation_list)
	Y_indices_validation = torch.cat(Y_indices_validation_list)
	Y_features_validation = torch.cat(Y_features_validation_list)
	
	X_indices_test = torch.cat(X_indices_test_list)
	X_features_test = torch.cat(X_features_test_list)
	Y_indices_test = torch.cat(Y_indices_test_list)
	Y_features_test = torch.cat(Y_features_test_list)
	
	return X_indices_training.to(device), X_indices_validation.to(device), X_indices_test.to(device), X_features_training.to(device), X_features_validation.to(device), X_features_test.to(device), Y_indices_training.to(device), Y_indices_validation.to(device), Y_indices_test.to(device), Y_features_training.to(device), Y_features_validation.to(device), Y_features_test.to(device)






def train_test_val_split_by_batches_also_track_IDs(X_indices, X_features, Y_indices, Y_features, Y_features_binning, track_ids, device):
	"""
	Split input data into training, validation, and test sets by batches of events, 
	including track IDs and binning features.
	
	This function divides the input data (features and indices) into three distinct
	sets: training, validation, and test, based on a random sampling of unique 
	event batch IDs. It also tracks the associated IDs and binning features. The 
	data is processed in batches to efficiently handle large datasets while 
	maintaining the integrity of the splits.
	
	Args:
	    X_indices (torch.Tensor): A tensor containing the indices of the input features.
	    X_features (torch.Tensor): A tensor containing the features of the input data.
	    Y_indices (torch.Tensor): A tensor containing the indices of the target features.
	    Y_features (torch.Tensor): A tensor containing the features of the target data.
	    Y_features_binning (torch.Tensor): A tensor containing the binning features of the target data.
	    track_ids (torch.Tensor): A tensor containing the track IDs associated with the input data.
	    device (torch.device): The device (CPU or GPU) to which the output tensors should be moved.
	
	Returns:
	    tuple: A tuple containing the training, validation, and test datasets, including track IDs and binning features:
	        - X_indices_training (torch.Tensor): Indices for the training dataset.
	        - X_indices_validation (torch.Tensor): Indices for the validation dataset.
	        - X_indices_test (torch.Tensor): Indices for the test dataset.
	        - X_features_training (torch.Tensor): Features for the training dataset.
	        - X_features_validation (torch.Tensor): Features for the validation dataset.
	        - X_features_test (torch.Tensor): Features for the test dataset.
	        - Y_indices_training (torch.Tensor): Target indices for the training dataset.
	        - Y_indices_validation (torch.Tensor): Target indices for the validation dataset.
	        - Y_indices_test (torch.Tensor): Target indices for the test dataset.
	        - Y_features_training (torch.Tensor): Target features for the training dataset.
	        - Y_features_validation (torch.Tensor): Target features for the validation dataset.
	        - Y_features_test (torch.Tensor): Target features for the test dataset.
	        - Y_features_binning_training (torch.Tensor): Binning features for the training dataset.
	        - Y_features_binning_validation (torch.Tensor): Binning features for the validation dataset.
	        - Y_features_binning_test (torch.Tensor): Binning features for the test dataset.
	        - track_ids_training (torch.Tensor): Track IDs for the training dataset.
	        - track_ids_validation (torch.Tensor): Track IDs for the validation dataset.
	        - track_ids_test (torch.Tensor): Track IDs for the test dataset.
	
	Note:
	    The function shuffles the unique batch IDs and splits them into training (70%),
	    validation (10%), and test (20%) sets before processing the data in batches.
	"""
	batch_size = 500
	# Split into training, validation and test datasets by batches of events
	unique_batches_IDs = torch.unique(X_indices[:,0:1])
	
	## Define the size of each of the three new lists
	integer_list = list(unique_batches_IDs.detach().cpu().numpy())
	random.shuffle(integer_list)
	size_training = int(len(integer_list) * 0.7)
	size_validation = int(len(integer_list) * 0.1)
	size_test = int(len(integer_list) * 0.2)
	
	## Split the shuffled list into three new lists
	list_training = integer_list[:size_training]
	list_validation = integer_list[size_training:size_training+size_validation]
	list_test = integer_list[size_training+size_validation:]
	
	## Generate the datasets
	list_training_tensor = torch.tensor(list_training, device=X_indices.device)
	list_validation_tensor = torch.tensor(list_validation, device=X_indices.device)
	list_test_tensor = torch.tensor(list_test, device=X_indices.device)
	
	
	# Initialize empty lists to store filtered data
	X_indices_training_list, X_features_training_list, Y_indices_training_list, Y_features_training_list, Y_features_binning_training_list = [], [], [], [], []
	X_indices_validation_list, X_features_validation_list, Y_indices_validation_list, Y_features_validation_list, Y_features_binning_validation_list = [], [], [], [], []
	X_indices_test_list, X_features_test_list, Y_indices_test_list, Y_features_test_list, Y_features_binning_test_list = [], [], [], [], []
	track_ids_training_list, track_ids_validation_list, track_ids_test_list = [], [], []
	
	# Iterate over the data in batches
	for i in tqdm(range(0, len(X_indices), batch_size)):
		# Extract a batch of data
		X_indices_batch = X_indices[i:i+batch_size]
		X_features_batch = X_features[i:i+batch_size]
		Y_indices_batch = Y_indices[i:i+batch_size]
		Y_features_batch = Y_features[i:i+batch_size]
		Y_features_binning_batch = Y_features_binning[i:i+batch_size]
		track_ids_batch = track_ids[i:i+batch_size]
		
		# Compare each element of list_training with the first column of X_indices_batch
		mask_training = (X_indices_batch[:, 0].unsqueeze(1) == list_training_tensor).any(dim=1)  # Use dim=1
		mask_validation = (X_indices_batch[:, 0].unsqueeze(1) == list_validation_tensor).any(dim=1)  # Use dim=1
		mask_test = (X_indices_batch[:, 0].unsqueeze(1) == list_test_tensor).any(dim=1)  # Use dim=1
		
		# Apply the mask to select the desired subset for this batch
		X_indices_training_batch = X_indices_batch[mask_training].to(torch.int32)
		X_features_training_batch = X_features_batch[mask_training]
		Y_indices_training_batch = Y_indices_batch[mask_training].to(torch.int32)
		Y_features_training_batch = Y_features_batch[mask_training]
		Y_features_binning_training_batch = Y_features_binning_batch[mask_training]
		
		X_indices_validation_batch = X_indices_batch[mask_validation].to(torch.int32)
		X_features_validation_batch = X_features_batch[mask_validation]
		Y_indices_validation_batch = Y_indices_batch[mask_validation].to(torch.int32)
		Y_features_validation_batch = Y_features_batch[mask_validation]
		Y_features_binning_validation_batch = Y_features_binning_batch[mask_validation]
		
		X_indices_test_batch = X_indices_batch[mask_test].to(torch.int32)
		X_features_test_batch = X_features_batch[mask_test]
		Y_indices_test_batch = Y_indices_batch[mask_test].to(torch.int32)
		Y_features_test_batch = Y_features_batch[mask_test]
		Y_features_binning_test_batch = Y_features_binning_batch[mask_test]
		
		track_ids_training_batch = track_ids_batch[mask_training].to(torch.int32)
		track_ids_validation_batch = track_ids_batch[mask_validation].to(torch.int32)
		track_ids_test_batch = track_ids_batch[mask_test].to(torch.int32)
		
		# Append the filtered data for this batch to the lists
		X_indices_training_list.append(X_indices_training_batch)
		X_features_training_list.append(X_features_training_batch)
		Y_indices_training_list.append(Y_indices_training_batch)
		Y_features_training_list.append(Y_features_training_batch)
		Y_features_binning_training_list.append(Y_features_binning_training_batch)
		
		X_indices_validation_list.append(X_indices_validation_batch)
		X_features_validation_list.append(X_features_validation_batch)
		Y_indices_validation_list.append(Y_indices_validation_batch)
		Y_features_validation_list.append(Y_features_validation_batch)
		Y_features_binning_validation_list.append(Y_features_binning_validation_batch)
		
		X_indices_test_list.append(X_indices_test_batch)
		X_features_test_list.append(X_features_test_batch)
		Y_indices_test_list.append(Y_indices_test_batch)
		Y_features_test_list.append(Y_features_test_batch)
		Y_features_binning_test_list.append(Y_features_binning_test_batch)
		
		track_ids_training_list.append(track_ids_training_batch)
		track_ids_validation_list.append(track_ids_validation_batch)
		track_ids_test_list.append(track_ids_test_batch)
		
	# Concatenate the lists of filtered data to get the final result
	X_indices_training = torch.cat(X_indices_training_list)
	X_features_training = torch.cat(X_features_training_list)
	Y_indices_training = torch.cat(Y_indices_training_list)
	Y_features_training = torch.cat(Y_features_training_list)
	Y_features_binning_training = torch.cat(Y_features_binning_training_list)
	
	X_indices_validation = torch.cat(X_indices_validation_list)
	X_features_validation = torch.cat(X_features_validation_list)
	Y_indices_validation = torch.cat(Y_indices_validation_list)
	Y_features_validation = torch.cat(Y_features_validation_list)
	Y_features_binning_validation = torch.cat(Y_features_binning_validation_list)
	
	X_indices_test = torch.cat(X_indices_test_list)
	X_features_test = torch.cat(X_features_test_list)
	Y_indices_test = torch.cat(Y_indices_test_list)
	Y_features_test = torch.cat(Y_features_test_list)
	Y_features_binning_test = torch.cat(Y_features_binning_test_list)
	
	track_ids_training = torch.cat(track_ids_training_list)
	track_ids_validation = torch.cat(track_ids_validation_list)
	track_ids_test = torch.cat(track_ids_test_list)
	
	return X_indices_training.to(device), X_indices_validation.to(device), X_indices_test.to(device), X_features_training.to(device), X_features_validation.to(device), X_features_test.to(device), Y_indices_training.to(device), Y_indices_validation.to(device), Y_indices_test.to(device), Y_features_training.to(device), Y_features_validation.to(device), Y_features_test.to(device), Y_features_binning_training.to(device), Y_features_binning_validation.to(device), Y_features_binning_test.to(device), track_ids_training.to(device), track_ids_validation.to(device), track_ids_test.to(device)






def get_sparse_tensor_2D(x, DTYPE, device):
	"""
	Convert a dense tensor to a 2D sparse tensor representation.
	
	This function takes a dense tensor, converts it to a specified data type, 
	and permutes its dimensions before creating a sparse tensor representation. 
	It is useful for handling high-dimensional data in sparse formats to 
	optimize memory usage and computational efficiency.
	
	Args:
	    x (torch.Tensor): The input dense tensor to be converted.
	    DTYPE (torch.dtype): The desired data type for the sparse tensor.
	    device (torch.device): The device (CPU or GPU) to which the tensor 
	                          should be moved before conversion.
	
	Returns:
	    spconv.SparseConvTensor: A 2D sparse tensor representation of the input 
	                              dense tensor.
	"""
	return spconv.SparseConvTensor.from_dense( torch.permute(x.type(DTYPE).to(device), (0, 2, 3, 1)) )






def get_sparse_tensor_3D(x, DTYPE, device):
	"""
	Convert a dense tensor to a 3D sparse tensor representation.
	
	This function takes a dense tensor, converts it to a specified data type, 
	and permutes its dimensions before creating a sparse tensor representation. 
	It is useful for handling high-dimensional data in sparse formats to 
	optimize memory usage and computational efficiency.
	
	Args:
	    x (torch.Tensor): The input dense tensor to be converted.
	    DTYPE (torch.dtype): The desired data type for the sparse tensor.
	    device (torch.device): The device (CPU or GPU) to which the tensor 
	                          should be moved before conversion.
	
	Returns:
	    spconv.SparseConvTensor: A 3D sparse tensor representation of the input 
	                              dense tensor.
	"""
	return spconv.SparseConvTensor.from_dense( torch.permute(x.type(DTYPE).to(device), (0, 2, 3, 4, 1)) )






def one_hot_mask(tensor, DTYPE):
	"""
	Generate a one-hot encoded mask from a tensor of class labels.
	
	This function takes a tensor of class labels and converts it into a 
	one-hot encoded format. Each unique class label in the input tensor 
	is represented as a separate binary channel, where a value of 1.0 
	indicates the presence of the class and 0.0 indicates its absence.
	
	Args:
	    tensor (torch.Tensor): A tensor containing class labels. Each element 
	                           should be an integer representing a class.
	    DTYPE (torch.dtype): The desired data type for the output one-hot 
	                         encoded tensor.
	
	Returns:
	    torch.Tensor: A tensor of shape (N, C) where N is the number of 
	                  elements in the input tensor and C is the number of 
	                  unique classes, containing the one-hot encoded 
	                  representation of the input tensor.
	"""
	n_classes = torch.unique( tensor ).shape[0]
	sample_onehot = []
	for label in range(n_classes):
		sample_onehot.append( torch.where( tensor == label, 1.0, 0.0 ) )
	sample_onehot = torch.cat(sample_onehot, dim = 1).type(DTYPE)
	return sample_onehot







def get_mask_from_prob(tensor, DTYPE):
	"""
	Generate a one-hot mask from the most probable class indices of a probability tensor.
	
	This function takes a tensor of probabilities across multiple classes, 
	determines the class with the highest probability for each sample, 
	and converts the resulting class indices into a one-hot encoded mask.
	
	Args:
	    tensor (torch.Tensor): A tensor containing probabilities of shape 
	                           (N, C), where N is the number of samples and 
	                           C is the number of classes.
	    DTYPE (torch.dtype): The desired data type for the output one-hot 
	                         encoded mask.
	
	Returns:
	    torch.Tensor: A one-hot encoded tensor of shape (1, N, C), where N 
	                  is the number of samples and C is the number of 
	                  classes, indicating the most probable class for each 
	                  sample.
	"""
	tensor = torch.argmax(tensor, dim = 1).unsqueeze(0)
	tensor = one_hot_mask(tensor, DTYPE)
	return tensor





def categorize_binning(x, DF):
	"""
	Categorize input data into bins based on radial and pitch coefficients.
	
	This function takes a multi-dimensional tensor and categorizes its 
	elements into bins according to specified radial and pitch coefficient 
	ranges. The binning is performed using a defined number of bins for 
	both coefficients, and the function returns a tensor of class IDs 
	indicating the bin each input falls into.
	
	Args:
	    x (torch.Tensor): A multi-dimensional tensor with shape 
	                      (N, C, D1, D2, D3) where N is the number of samples, 
	                      C is the number of features (expected to be at least 2 for 
	                      radial and pitch coefficients), and D1, D2, D3 are the 
	                      spatial dimensions.
	    DF (pandas.DataFrame): A DataFrame containing the minimum and maximum 
	                           values for the radial and pitch coefficients, 
	                           specifically with columns 'radial_coeff' and 
	                           'pitch_coeff'.
	
	Returns:
	    torch.Tensor: A tensor of class IDs with the same spatial dimensions as 
	                  the input tensor, where each element indicates the bin 
	                  category or is set to 0 if the corresponding input values 
	                  are not significant (less than or equal to 1e-2).
	"""
	## Regarding the binning
	n_bins_radial_coeff = 20
	n_bins_pitch_coeff = 20
	radial_coeff_min, radial_coeff_max = DF['radial_coeff'].min(), DF['radial_coeff'].max()
	pitch_coeff_min, pitch_coeff_max = DF['pitch_coeff'].min(), DF['pitch_coeff'].max()
	radial_coeff_bin_size = ((radial_coeff_max - radial_coeff_min) / n_bins_radial_coeff) + 1e-7
	pitch_coeff_bin_size = ((pitch_coeff_max - pitch_coeff_min) / n_bins_pitch_coeff) + 1e-7
	
	radial_coeff_bin = ((x[:,0:1,:,:,:] - radial_coeff_min) // radial_coeff_bin_size).to(torch.int)
	pitch_coeff_bin = ((x[:,1:2,:,:,:] - pitch_coeff_min) // pitch_coeff_bin_size).to(torch.int)
	
	class_id = torch.where( ( torch.abs(x[:,0:1,:,:,:]) > 1e-2 ).to(x.device) & ( torch.abs(x[:,1:2,:,:,:]) > 1e-2 ).to(x.device), (1 + (radial_coeff_bin * n_bins_pitch_coeff + pitch_coeff_bin)).to(x.dtype).to(x.device), torch.tensor(0.0, dtype = x.dtype, device = x.device) )
	#class_id = 1 + (radial_coeff_bin * n_bins_pitch_coeff + pitch_coeff_bin)
	
	return class_id






# =================  Define extra Losses  =================


class CrossEntropyLoss(nn.Module):
	"""
	CrossEntropyLoss class for computing the cross-entropy loss between 
	predicted and true feature distributions.
	
	This class inherits from PyTorch's nn.Module and provides a custom 
	implementation of cross-entropy loss suitable for sparse representations 
	of predictions and ground truths.
	
	Args:
	    DTYPE (torch.dtype): The desired data type for tensor computations 
	                         (e.g., torch.float32).
	    device (torch.device): The device (CPU or GPU) on which the tensors 
	                           will be allocated.
	
	Methods:
	    forward(y_true, y_pred): Computes the cross-entropy loss between 
	                              true and predicted values.
	
	    Args:
	        y_true (SparseConvTensor): The ground truth values, expected 
	                                    to have 'features' and 'indices' 
	                                    attributes.
	        y_pred (SparseConvTensor): The predicted values, also expected 
	                                    to have 'features' and 'indices' 
	                                    attributes.
	
	    Returns:
	        torch.Tensor: The computed cross-entropy loss, averaged over 
	                       the batch size.
	
	Example:
	    >>> criterion = CrossEntropyLoss(DTYPE=torch.float32, device='cuda')
	    >>> y_true = ...  # SparseConvTensor with ground truth
	    >>> y_pred = ...  # SparseConvTensor with predictions
	    >>> loss = criterion(y_true, y_pred)
	"""
	def __init__(self, DTYPE, device):
		super(CrossEntropyLoss, self).__init__()
		self.DTYPE, self.device = DTYPE, device
	def forward(self, y_true, y_pred):
		y_true_features, y_pred_features = y_true.features, y_pred.features
		y_true_indices, y_pred_indices = y_true.indices, y_pred.indices
		# Set the batch size
		batch_size = 3000
		indices_list = []
		# Split y_true_indices into batches
		num_batches = (y_true_indices.shape[0] + batch_size - 1) // batch_size
		for i in tqdm(range(num_batches)):
			start_idx = i * batch_size
			end_idx = min((i + 1) * batch_size, y_true_indices.shape[0])
			batch = y_true_indices[start_idx:end_idx]
			# Create a boolean mask indicating the rows of the batch that are included in y_pred_indices
			mask = (batch.unsqueeze(1) == y_pred_indices).all(dim=2).any(dim=1)
			# Get the indices where the entire rows are included in the batch
			batch_indices = (torch.nonzero(mask) + start_idx).flatten().tolist()
			indices_list.extend(batch_indices)
		y_true_indices_filtered = y_true_indices[indices_list,:]
		y_true_features_filtered = y_true_features[indices_list,:]
		
		y_true_features_filtered_not_BG = y_true_features_filtered[ y_true_features_filtered[:,0] != 1 ]
		y_pred_features_filtered_not_BG = y_pred_features[ y_true_features_filtered[:,0] != 1 ]
		
		y_true_features_filtered_not_BG = y_true_features_filtered_not_BG[:,1:]
		y_pred_features_filtered_not_BG = y_pred_features_filtered_not_BG
		
		loss = -torch.mean( y_true_features_filtered_not_BG * y_pred_features_filtered_not_BG, dim = 0 ).sum()
		return loss








def _analyze_tracks(truth, submission):
	"""Compute the majority particle, hit counts, and weight for each track.
	
	Parameters
	----------
	truth : pandas.DataFrame
	    Truth information. Must have hit_id, particle_id, and weight columns.
	submission : pandas.DataFrame
	    Proposed hit/track association. Must have hit_id and track_id columns.
	
	Returns
	-------
	pandas.DataFrame
	    Contains track_id, nhits, major_particle_id, major_particle_nhits,
	    major_nhits, and major_weight columns.
	"""
	# true number of hits for each particle_id
	particles_nhits = truth['particle_id'].value_counts(sort=False)
	total_weight = truth['weight'].sum()
	# combined event with minimal reconstructed and truth information
	event = pd.merge(truth[['hit_id', 'particle_id', 'weight']],
	                     submission[['hit_id', 'track_id']],
	                     on=['hit_id'], how='left', validate='one_to_one')
	event.drop('hit_id', axis=1, inplace=True)
	event.sort_values(by=['track_id', 'particle_id'], inplace=True)
	
	# ASSUMPTIONs: 0 <= track_id, 0 <= particle_id
	
	tracks = []
	# running sum for the reconstructed track we are currently in
	rec_track_id = -1
	rec_nhits = 0
	# running sum for the particle we are currently in (in this track_id)
	cur_particle_id = -1
	cur_nhits = 0
	cur_weight = 0
	# majority particle with most hits up to now (in this track_id)
	maj_particle_id = -1
	maj_nhits = 0
	maj_weight = 0
	
	for hit in event.itertuples(index=False):
		# we reached the next track so we need to finish the current one
		if (rec_track_id != -1) and (rec_track_id != hit.track_id):
			# could be that the current particle is the majority one
			if maj_nhits < cur_nhits:
				maj_particle_id = cur_particle_id
				maj_nhits = cur_nhits
				maj_weight = cur_weight
			# store values for this track
			tracks.append((rec_track_id, rec_nhits, maj_particle_id, particles_nhits[maj_particle_id], maj_nhits, maj_weight / total_weight))
		
		# setup running values for next track (or first)
		if rec_track_id != hit.track_id:
			rec_track_id = hit.track_id
			rec_nhits = 1
			cur_particle_id = hit.particle_id
			cur_nhits = 1
			cur_weight = hit.weight
			maj_particle_id = -1
			maj_nhits = 0
			maj_weights = 0
			continue
		
		# hit is part of the current reconstructed track
		rec_nhits += 1
		
		# reached new particle within the same reconstructed track
		if cur_particle_id != hit.particle_id:
			# check if last particle has more hits than the majority one
			# if yes, set the last particle as the new majority particle
			if maj_nhits < cur_nhits:
				maj_particle_id = cur_particle_id
				maj_nhits = cur_nhits
				maj_weight = cur_weight
			# reset runnig values for current particle
			cur_particle_id = hit.particle_id
			cur_nhits = 1
			cur_weight = hit.weight
		# hit belongs to the same particle within the same reconstructed track
		else:
			cur_nhits += 1
			cur_weight += hit.weight
	# last track is not handled inside the loop
	if maj_nhits < cur_nhits:
		maj_particle_id = cur_particle_id
		maj_nhits = cur_nhits
		maj_weight = cur_weight
	# store values for the last track
	tracks.append((rec_track_id, rec_nhits, maj_particle_id, particles_nhits[maj_particle_id], maj_nhits, maj_weight / total_weight))
	
	cols = ['track_id', 'nhits', 'major_particle_id', 'major_particle_nhits', 'major_nhits', 'major_weight']
	return pd.DataFrame.from_records(tracks, columns=cols)






def score_event(truth, submission):
	"""Compute the TrackML event score for a single event.
	Parameters
	----------
	truth : pandas.DataFrame
	    Truth information. Must have hit_id, particle_id, and weight columns.
	submission : pandas.DataFrame
	    Proposed hit/track association. Must have hit_id and track_id columns.
	"""
	tracks = _analyze_tracks(truth, submission)
	purity_rec = np.true_divide(tracks['major_nhits'], tracks['nhits'])
	purity_maj = np.true_divide(tracks['major_nhits'], tracks['major_particle_nhits'])
	good_track = (0.5 < purity_rec) & (0.5 < purity_maj)
	return tracks['major_weight'][good_track].sum()






def fp_fn_rate(pred_lbl, true_lbl):
	"""
	Calculate the false positive and false negative rates based on predicted 
	and true labels.
	
	This function takes predicted labels and true labels for particles and 
	computes the rates of false positives (FP rate) and false negatives (FN 
	rate). It uses a majority voting mechanism to determine the cluster ID 
	for each true label.
	
	Args:
	    pred_lbl (torch.Tensor): A tensor containing the predicted labels for 
	                             each hit.
	    true_lbl (torch.Tensor): A tensor containing the true labels for each 
	                             hit.
	
	Returns:
	    tuple: A tuple containing two float values:
	        - fp_rate (float): The rate of false positives, defined as the 
	                           ratio of incorrectly assigned hits to the total 
	                           number of hits.
	        - fn_rate (float): The rate of false negatives, defined as the 
	                           ratio of missed true hits to the total number of 
	                           hits.
	
	Example:
	    >>> fp_rate, fn_rate = fp_fn_rate(predicted_labels, true_labels)
	"""
	truth_rows, pred_rows = [], []
	for ind, part in enumerate(true_lbl):
		truth_rows.append((ind, part.item(), 1))
	
	for _, pred in enumerate(pred_lbl):
		pred_rows.append(pred.item())
	
	df = pd.DataFrame(truth_rows)
	df.columns = ['hit_id', 'particle_id', 'weight']
	df['cluster_id'] = pred_rows
	
	total_nr_hits = len(df['hit_id'])
	
	grouped_df = df.groupby('cluster_id')
	
	def extract_fps(rows):
		label_data = rows["particle_id"].to_numpy(dtype=np.float32)
		data = Counter(label_data)
		majority_label = data.most_common(1)[0][0]
		track_belonging = np.array([label == majority_label for label in label_data], dtype=int).astype(np.float32)
		fps = len(track_belonging[track_belonging == False])
		return fps
	
	fp_rate = (grouped_df.apply(extract_fps)).sum()/total_nr_hits
	
	grouped_df = df.groupby('particle_id')
	
	def extract_fns(rows):
		label_data = rows["cluster_id"].to_numpy(dtype=np.float32)
		data = Counter(label_data)
		majority_label = data.most_common(1)[0][0]
		track_belonging = np.array([label == majority_label for label in label_data], dtype=int).astype(np.float32)
		fns = len(track_belonging[track_belonging == False])
		return fns
	
	fn_rate = (grouped_df.apply(extract_fns)).sum()/total_nr_hits
	return fp_rate, fn_rate




        

def calc_score(pred_lbl, true_lbl):
	"""
	Calculate the score for a set of predictions against the true labels.
	
	This function constructs data frames for the true labels and predicted 
	labels, then computes the score based on the provided evaluation method. 
	
	Args:
	    pred_lbl (torch.Tensor): A tensor containing the predicted track IDs 
	                             for each hit.
	    true_lbl (torch.Tensor): A tensor containing the true particle IDs 
	                             for each hit.
	
	Returns:
	    float: The score calculated based on the true and predicted labels, 
	           as evaluated by the `score_event` function.
	
	Example:
	    >>> score = calc_score(predicted_labels, true_labels)
	"""
	truth_rows, pred_rows = [], []
	for ind, part in enumerate(true_lbl):
		truth_rows.append((ind, part.item(), 1))
	
	for ind, pred in enumerate(pred_lbl):
		pred_rows.append((ind, pred.item()))
	
	truth = pd.DataFrame(truth_rows)
	truth.columns = ['hit_id', 'particle_id', 'weight']
	submission = pd.DataFrame(pred_rows)
	submission.columns = ['hit_id', 'track_id']
	return score_event(truth, submission)






def calc_score_trackml(pred_lbl, true_lbl):
	"""
	Calculate the score for a set of predictions against the true labels for the TrackML dataset.
	
	This function constructs data frames for the true labels (including hit IDs, 
	particle IDs, and weights) and predicted track IDs, then computes the score 
	based on the provided evaluation method.
	
	Args:
	    pred_lbl (torch.Tensor): A tensor containing the predicted track IDs 
	                             for each hit.
	    true_lbl (torch.Tensor): A tensor containing the true labels for each 
	                             hit, where each label consists of a tuple 
	                             containing the particle ID and its weight.
	
	Returns:
	    float: The score calculated based on the true and predicted labels, 
	           as evaluated by the `score_event` function.
	"""
	truth_rows, pred_rows = [], []
	for ind, part in enumerate(true_lbl):
		truth_rows.append((ind, part[0].item(), part[1].item()))
	
	for ind, pred in enumerate(pred_lbl):
		pred_rows.append((ind, pred.item()))
	
	truth = pd.DataFrame(truth_rows)
	truth.columns = ['hit_id', 'particle_id', 'weight']
	submission = pd.DataFrame(pred_rows)
	submission.columns = ['hit_id', 'track_id']
	return score_event(truth, submission)























































































