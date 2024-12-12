from functools import partial
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv
import utils

##################################################################################################################

# Sparse U-Net in 3D
## Basic block of convolutionals



def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0, conv_type='subm', norm_fn=None):
	"""
	Creates a post-activation block with convolution, normalization, and Tanh activation. 
	
	This block supports different types of sparse convolution, including Submanifold, 
	Sparse, and Sparse Inverse convolutions. The block also includes normalization and 
	a Tanh activation function.
	
	:param in_channels: 
	    Number of input channels for the convolution layer.
	:type in_channels: int
	:param out_channels: 
	    Number of output channels for the convolution layer.
	:type out_channels: int
	:param kernel_size: 
	    Size of the convolution kernel.
	:type kernel_size: int or tuple
	:param indice_key: 
	    Indice key to use for sparse convolution (used for weight sharing), defaults to None.
	:type indice_key: str or None, optional
	:param stride: 
	    Stride for the convolution layer, defaults to 1.
	:type stride: int, optional
	:param padding: 
	    Padding for the convolution layer, defaults to 0.
	:type padding: int, optional
	:param conv_type: 
	    Type of convolution to use. Choices are 'subm' (SubMConv3d), 'spconv' (SparseConv3d), 
	    or 'inverseconv' (SparseInverseConv3d). Defaults to 'subm'.
	:type conv_type: str, optional
	:param norm_fn: 
	    Normalization function to apply after the convolution layer.
	:type norm_fn: callable
	:raises NotImplementedError: 
	    If an invalid `conv_type` is provided.
	:return: 
	    A SparseSequential module consisting of the convolution, normalization, 
	    and Tanh activation function.
	:rtype: spconv.SparseSequential
	"""
	if conv_type == 'subm':
		conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
	elif conv_type == 'spconv':
		conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
	                               bias=False, indice_key=indice_key)
	elif conv_type == 'inverseconv':
		conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
	else:
		raise NotImplementedError
	m = spconv.SparseSequential(
		conv,
		norm_fn(out_channels),
		nn.Tanh(),
	)
	return m







class SparseBasicBlock(spconv.SparseModule):
	"""
	A SparseBasicBlock that consists of two sparse 3D convolutional layers, batch normalization, 
	and Tanh activations. This block can optionally downsample the input to match dimensions 
	when the stride is applied.
	
	:cvar expansion: 
	    A factor to expand the number of output channels (set to 1 for this basic block).
	:type expansion: int
	
	:param inplanes: 
	    Number of input channels for the first convolutional layer.
	:type inplanes: int
	:param planes: 
	    Number of output channels for both convolutional layers.
	:type planes: int
	:param kernel_size: 
	    Size of the convolution kernel for both convolutional layers.
	:type kernel_size: int or tuple
	:param stride: 
	    Stride for the convolutional layers. Defaults to 1.
	:type stride: int, optional
	:param bias: 
	    Whether or not to use bias in the convolutional layers. If not provided, it is 
	    automatically set based on whether normalization is used. Defaults to None.
	:type bias: bool or None, optional
	:param norm_fn: 
	    Normalization function to apply after each convolutional layer (e.g., batch normalization).
	:type norm_fn: callable
	:param downsample: 
	    A downsampling layer applied to the input to match output dimensions if needed. Defaults to None.
	:type downsample: callable or None, optional
	:param indice_key: 
	    Indice key used for sparse convolution (helps with weight sharing between layers). Defaults to None.
	:type indice_key: str or None, optional
	
	:raises AssertionError: 
	    If `norm_fn` is not provided.
	"""
	expansion = 1
	def __init__(self, inplanes, planes, kernel_size, stride=1, bias=None, norm_fn=None, downsample=None, indice_key=None):
		super(SparseBasicBlock, self).__init__()
		assert norm_fn is not None
		if bias is None:
			bias = norm_fn is not None
		self.conv1 = spconv.SubMConv3d(
			inplanes, planes, kernel_size=kernel_size, stride=stride, padding=1, bias=bias, indice_key=indice_key
		)
		self.bn1 = norm_fn(planes)
		self.act = nn.Tanh()
		self.conv2 = spconv.SubMConv3d(
			planes, planes, kernel_size=kernel_size, stride=stride, padding=1, bias=bias, indice_key=indice_key
		)
		self.bn2 = norm_fn(planes)
		self.downsample = downsample
		self.stride = stride
	def forward(self, x):
		identity = x
		out = self.conv1(x)
		out = utils.replace_feature(out, self.bn1(out.features))
		out = utils.replace_feature(out, self.act(out.features))
		out = self.conv2(out)
		out = utils.replace_feature(out, self.bn2(out.features))
		if self.downsample is not None:
			identity = self.downsample(x)
		out = utils.replace_feature(out, out.features + identity.features)
		out = utils.replace_feature(out, self.act(out.features))
		return out







# UNet_vanilla with submanifold convolutions for regression
class UNet_SCN_regressor(nn.Module):
	"""
	A 3D U-Net architecture built using sparse convolutions for regression tasks, particularly suited 
	for tasks with sparse input data. This model includes an encoder-decoder structure with 
	residual connections, Tanh activations, and batch normalization.
	
	:param in_channels: 
	    Number of input channels for the first convolutional layer.
	:type in_channels: int
	:param out_channels: 
	    Number of output channels for the final layer.
	:type out_channels: int
	:param DTYPE: 
	    The data type used for the model's parameters.
	:type DTYPE: torch.dtype
	:param device: 
	    The device (CPU or GPU) where the model will be run.
	:type device: torch.device
	:param kernel_size: 
	    Kernel size for the convolutional layers.
	:type kernel_size: int or tuple
	:param spatial_size: 
	    The spatial size of the input data.
	:type spatial_size: int or tuple
	"""
	def __init__(self, in_channels, out_channels, DTYPE, device, kernel_size, spatial_size):
		super(UNet_SCN_regressor, self).__init__()
		# Parameters
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_size = kernel_size
		self.spatial_size = spatial_size
		self.device = device
		self.DTYPE = DTYPE
		# Neural architecture
		norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
		## Encoder =======================================================================================================
		
		self.conv_input = spconv.SparseSequential(
			spconv.SubMConv3d(self.in_channels, 32, 3, padding=1, bias=False, indice_key='subm1'),
			norm_fn(32),
			nn.Tanh(),
		)
		
		block = post_act_block
		self.conv1 = spconv.SparseSequential(
			block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm1', conv_type='subm'),
		)
		
		self.conv2 = spconv.SparseSequential(
			# [1600, 1408, 41] <- [800, 704, 21]
			block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
			block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
			block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
		)
		
		self.conv3 = spconv.SparseSequential(
			# [800, 704, 21] <- [400, 352, 11]
			block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
			block(128, 128, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
			block(128, 128, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
		)
		
		self.conv4 = spconv.SparseSequential(
			# [400, 352, 11] <- [200, 176, 5]
			block(128, 128, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
			block(128, 128, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
			block(128, 128, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
		)
		
		## Decoder =======================================================================================================
		
		self.conv_up_t4 = SparseBasicBlock(128, 128, indice_key='subm4', norm_fn=norm_fn, kernel_size=self.kernel_size)
		self.conv_up_m4 = block(256, 128, 3, norm_fn=norm_fn, padding=1, indice_key='subm4')
		self.inv_conv4 = block(128, 128, 3, norm_fn=norm_fn, indice_key='spconv4', conv_type='inverseconv')
		
		self.conv_up_t3 = SparseBasicBlock(128, 128, indice_key='subm3', norm_fn=norm_fn, kernel_size=self.kernel_size)
		self.conv_up_m3 = block(256, 128, 3, norm_fn=norm_fn, padding=1, indice_key='subm3')
		self.inv_conv3 = block(128, 64, 3, norm_fn=norm_fn, indice_key='spconv3', conv_type='inverseconv')
		
		self.conv_up_t2 = SparseBasicBlock(64, 64, indice_key='subm2', norm_fn=norm_fn, kernel_size=self.kernel_size)
		self.conv_up_m2 = block(128, 64, 3, norm_fn=norm_fn, indice_key='subm2')
		self.inv_conv2 = block(64, 32, 3, norm_fn=norm_fn, indice_key='spconv2', conv_type='inverseconv')
		
		self.conv_up_t1 = SparseBasicBlock(32, 32, indice_key='subm1', norm_fn=norm_fn, kernel_size=self.kernel_size)
		self.conv_up_m1 = block(64, 32, 3, norm_fn=norm_fn, indice_key='subm1')
		
		self.conv5 = spconv.SparseSequential(
			block(32, self.out_channels, 3, norm_fn=norm_fn, padding=1, indice_key='subm1')
		)
		
		# Loss
		self.criterion = nn.MSELoss()
		# Create lists to save loss
		self.loss_training_hist, self.loss_validation_hist = [], []
	def UR_block_forward(self, x_lateral, x_bottom, conv_t, conv_m, conv_inv):
		x_trans = conv_t(x_lateral)
		x = x_trans
		x = utils.replace_feature(x, torch.cat((x_bottom.features, x_trans.features), dim=1))
		x_m = conv_m(x)
		x = self.channel_reduction(x, x_m.features.shape[1])
		x = utils.replace_feature(x, x_m.features + x.features)
		x = conv_inv(x)
		return x
	@staticmethod
	def channel_reduction(x, out_channels):
		"""
		Args:
		    x: x.features (N, C1)
		    out_channels: C2
		Returns:
		"""
		features = x.features
		n, in_channels = features.shape
		assert (in_channels % out_channels == 0) and (in_channels >= out_channels)
		x = utils.replace_feature(x, features.view(n, out_channels, -1).sum(dim=2))
		return x
	def forward(self, x):
		# Embedding input
		x = self.conv_input(x)
		# Encoder
		x_conv1 = self.conv1(x)
		x_conv2 = self.conv2(x_conv1)
		x_conv3 = self.conv3(x_conv2)
		x_conv4 = self.conv4(x_conv3)
		# Decoder
		x_up4 = self.UR_block_forward(x_conv4, x_conv4, self.conv_up_t4, self.conv_up_m4, self.inv_conv4)
		x_up3 = self.UR_block_forward(x_conv3, x_up4, self.conv_up_t3, self.conv_up_m3, self.inv_conv3)
		x_up2 = self.UR_block_forward(x_conv2, x_up3, self.conv_up_t2, self.conv_up_m2, self.inv_conv2)
		x_up1 = self.UR_block_forward(x_conv1, x_up2, self.conv_up_t1, self.conv_up_m1, self.conv5)
		
		indices_pred, features_pred, = x_up1.indices, torch.cat( (nn.ReLU()(x_up1.features[:,0:1]), nn.ReLU()(x_up1.features[:,1:2]) ), dim = 1 )
		
		return indices_pred, features_pred
	def compute_loss(self, x, y_true):
		indices_pred, features_pred = self(x)
		features_truth = y_true.features
		
		# Loss computation
		loss = self.criterion(features_pred, features_truth)
		del features_pred, features_truth
		return loss
	def train_step(self, x, y_true):
		# Compute loss
		loss = self.compute_loss(x, y_true)
		loss.backward(retain_graph = True)
		self.optimizer.step()
		self.optimizer.zero_grad(set_to_none = True)
		
		del x, y_true
		
		return loss







# UNet_vanilla with submanifold convolutions for classification
class UNet_SCN_classifier(nn.Module):
	"""
	UNet-based Sparse Convolutional Neural Network (SCN) Classifier.
	
	This class defines a UNet-based architecture using sparse convolutions for 3D data.
	It includes both an encoder-decoder structure with upsampling and skip connections.
	It is designed for classification tasks with input channels, output channels, and 
	a defined spatial size.
	
	Args:
	    in_channels (int): Number of input channels.
	    out_channels (int): Number of output channels (number of classes).
	    DTYPE (torch.dtype): Data type of the network (e.g., torch.float32).
	    device (torch.device): Device where the network is loaded (e.g., "cuda" or "cpu").
	    kernel_size (int): Size of the convolutional kernel.
	    spatial_size (tuple): Dimensions of the spatial input (e.g., (x, y, z)).
	    density_map (float): Density map for 3D input, used in scaling.
	    scale_factor (float): Factor by which to scale the model layers.
	"""
	def __init__(self, in_channels, out_channels, DTYPE, device, kernel_size, spatial_size, scale_factor):
		super(UNet_SCN_classifier, self).__init__()
		# Parameters
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_size = kernel_size
		self.spatial_size = spatial_size
		self.device = device
		self.DTYPE = DTYPE
		self.scale_factor = scale_factor
		# Neural architecture
		norm_fn = partial(nn.BatchNorm1d, eps=1e-4, momentum=0.001)
		## Encoder =======================================================================================================
		
		self.conv_input = spconv.SparseSequential(
			spconv.SubMConv3d(self.in_channels, int(128*self.scale_factor), self.kernel_size, padding=1, bias=False, indice_key='subm1'),
			norm_fn(int(128*self.scale_factor)),
			nn.Tanh(),
		)
		
		block = post_act_block
		self.conv1 = spconv.SparseSequential(
			block(int(128*self.scale_factor), int(128*self.scale_factor), self.kernel_size, norm_fn=norm_fn, padding=1, indice_key='subm1', conv_type='subm'),
		)
		
		self.conv2 = spconv.SparseSequential(
			# [1600, 1408, 41] <- [800, 704, 21]
			block(int(128*self.scale_factor), int(256*self.scale_factor), self.kernel_size, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
			block(int(256*self.scale_factor), int(256*self.scale_factor), self.kernel_size, norm_fn=norm_fn, padding=1, indice_key='subm2'),
			block(int(256*self.scale_factor), int(256*self.scale_factor), self.kernel_size, norm_fn=norm_fn, padding=1, indice_key='subm2'),
		)
		
		self.conv3 = spconv.SparseSequential(
			# [800, 704, 21] <- [400, 352, 11]
			block(int(256*self.scale_factor), int(512*self.scale_factor), self.kernel_size, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
			block(int(512*self.scale_factor), int(512*self.scale_factor), self.kernel_size, norm_fn=norm_fn, padding=1, indice_key='subm3'),
			block(int(512*self.scale_factor), int(512*self.scale_factor), self.kernel_size, norm_fn=norm_fn, padding=1, indice_key='subm3'),
		)
		
		self.conv4 = spconv.SparseSequential(
			# [400, 352, 11] <- [200, 176, 5]
			block(int(512*self.scale_factor), int(512*self.scale_factor), self.kernel_size, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
			block(int(512*self.scale_factor), int(512*self.scale_factor), self.kernel_size, norm_fn=norm_fn, padding=1, indice_key='subm4'),
			block(int(512*self.scale_factor), int(512*self.scale_factor), self.kernel_size, norm_fn=norm_fn, padding=1, indice_key='subm4'),
		)
		
		## Decoder =======================================================================================================
		
		self.conv_up_t4 = SparseBasicBlock(int(512*self.scale_factor), int(512*self.scale_factor), indice_key='subm4', norm_fn=norm_fn, kernel_size=self.kernel_size)
		self.conv_up_m4 = block(int(1024*self.scale_factor), int(512*self.scale_factor), self.kernel_size, norm_fn=norm_fn, padding=1, indice_key='subm4')
		self.inv_conv4 = block(int(512*self.scale_factor), int(512*self.scale_factor), self.kernel_size, norm_fn=norm_fn, indice_key='spconv4', conv_type='inverseconv')
		
		self.conv_up_t3 = SparseBasicBlock(int(512*self.scale_factor), int(512*self.scale_factor), indice_key='subm3', norm_fn=norm_fn, kernel_size=self.kernel_size)
		self.conv_up_m3 = block(int(1024*self.scale_factor), int(512*self.scale_factor), self.kernel_size, norm_fn=norm_fn, padding=1, indice_key='subm3')
		self.inv_conv3 = block(int(512*self.scale_factor), int(256*self.scale_factor), self.kernel_size, norm_fn=norm_fn, indice_key='spconv3', conv_type='inverseconv')
		
		self.conv_up_t2 = SparseBasicBlock(int(256*self.scale_factor), int(256*self.scale_factor), indice_key='subm2', norm_fn=norm_fn, kernel_size=self.kernel_size)
		self.conv_up_m2 = block(int(512*self.scale_factor), int(256*self.scale_factor), self.kernel_size, norm_fn=norm_fn, indice_key='subm2')
		self.inv_conv2 = block(int(256*self.scale_factor), int(128*self.scale_factor), self.kernel_size, norm_fn=norm_fn, indice_key='spconv2', conv_type='inverseconv')
		
		self.conv_up_t1 = SparseBasicBlock(int(128*self.scale_factor), int(128*self.scale_factor), indice_key='subm1', norm_fn=norm_fn, kernel_size=self.kernel_size)
		self.conv_up_m1 = block(int(256*self.scale_factor), int(128*self.scale_factor), self.kernel_size, norm_fn=norm_fn, indice_key='subm1')
		
		self.conv5 = spconv.SparseSequential(
			block(int(128*self.scale_factor), self.out_channels, self.kernel_size, norm_fn=norm_fn, padding=1, indice_key='subm1')
		)
		
		# Loss
		self.criterion = nn.CrossEntropyLoss()
		# Create lists to save loss
		self.loss_training_hist, self.loss_validation_hist = [], []
		# Create lists to save trackML_score
		self.trackML_training_hist, self.trackML_validation_hist, self.trackML_test_hist = [], [], []
		# Create lists to save accuracy
		self.accuracy_training_hist, self.accuracy_validation_hist, self.accuracy_validation_test = [], [], []
	def UR_block_forward(self, x_lateral, x_bottom, conv_t, conv_m, conv_inv):
		x_trans = conv_t(x_lateral)
		x = x_trans
		x = utils.replace_feature(x, torch.cat((x_bottom.features, x_trans.features), dim=1))
		x_m = conv_m(x)
		x = self.channel_reduction(x, x_m.features.shape[1])
		x = utils.replace_feature(x, x_m.features + x.features)
		x = conv_inv(x)
		return x
	@staticmethod
	def channel_reduction(x, out_channels):
		"""
		Args:
		    x: x.features (N, C1)
		    out_channels: C2
		Returns:
		"""
		features = x.features
		n, in_channels = features.shape
		assert (in_channels % out_channels == 0) and (in_channels >= out_channels)
		x = utils.replace_feature(x, features.view(n, out_channels, -1).sum(dim=2))
		return x
	def forward(self, x):
		# Embedding input
		x = self.conv_input(x)
		# Encoder
		x_conv1 = self.conv1(x)
		x_conv2 = self.conv2(x_conv1)
		x_conv3 = self.conv3(x_conv2)
		x_conv4 = self.conv4(x_conv3)
		# Decoder
		x_up4 = self.UR_block_forward(x_conv4, x_conv4, self.conv_up_t4, self.conv_up_m4, self.inv_conv4)
		x_up3 = self.UR_block_forward(x_conv3, x_up4, self.conv_up_t3, self.conv_up_m3, self.inv_conv3)
		x_up2 = self.UR_block_forward(x_conv2, x_up3, self.conv_up_t2, self.conv_up_m2, self.inv_conv2)
		x_up1 = self.UR_block_forward(x_conv1, x_up2, self.conv_up_t1, self.conv_up_m1, self.conv5)
		
		indices_pred, features_pred, = x_up1.indices, x_up1.features
		
		return indices_pred, features_pred
	def compute_trackML(self, x, track_id):
		indices_pred, features_pred = self(x)
		classes_pred = torch.argmax(nn.Softmax(dim=1)(features_pred), dim = 1)
		
		track_id = track_id.to(torch.long)
		
		trackML = utils.calc_score(pred_lbl = classes_pred.view(-1).detach().cpu().numpy(), true_lbl = track_id.view(-1).detach().cpu().numpy())
		return trackML
	def compute_accuracy(self, x, y_true):
		features_truth = y_true.features.to(torch.long)
		indices_pred, features_pred = self(x)
		classes_pred = torch.argmax(nn.Softmax(dim=1)(features_pred), dim = 1).view(-1,1)
		accuracy = (features_truth == classes_pred).sum() / features_truth.shape[0]
		
		del features_pred, features_truth
		
		return accuracy
	def compute_loss(self, x, y_true):
		indices_pred, features_pred = self(x)
		
		classes_pred = torch.argmax(features_pred, dim = 1).view(-1,1)
		features_truth = y_true.features.to(torch.long)
		
		# Loss computation
		loss = self.criterion(features_pred, features_truth.squeeze())
		
		del features_pred, features_truth
		return loss
	def train_step(self, x, y_true):
		# Compute loss
		loss = self.compute_loss(x, y_true)
		loss.backward(retain_graph = True)
		self.optimizer.step()
		self.optimizer.zero_grad(set_to_none = True)
		
		del x, y_true
		
		return loss









































































































