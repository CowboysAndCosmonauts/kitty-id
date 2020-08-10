"""
Network Class for Neural Network

"""


###Libraries
import numpy as np 


class Network(object):

	def __init__(self, layer_sizes):
		"""List of sizes contains the nodes of the neural network.
		Weights and biases initialized randomly
		"""
		self.num_layers = len(layer_sizes)
		self.layer_sizes = layer_sizes

	def train(self, X, Y, learning_rate = 0.0075, num_iterations = 3000, print_cost = False):
		"""
		Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

		Arguments:
		X -- data, numpy array of shape (num_px * num_px * 3, number of examples)
		Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
		layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
		learning_rate -- learning rate of the gradient descent update rule
		num_iterations -- number of iterations of the optimization loop
		print_cost -- if True, it prints the cost every 100 steps

		Returns:
		parameters -- parameters learnt by the model. They can then be used to predict.
		"""

		assert(X.shape[0] == self.layer_sizes[0])

		costs = []

		parameters = self.initialize_parameters_deep(self.layer_sizes)

		for i in range(0, num_iterations):
			#Forward propagation
			AL, caches = self.L_model_forward(X, parameters)

			#Compute cost
			cost = self.compute_cost(AL, Y)

			#Backwards propagation
			grads = self.L_model_backward(AL, Y, caches)

			#Update Parameters
			self.parameters = self.update_parameters(parameters, grads, learning_rate)

			#print costs
			if print_cost and i % 100 == 0:
				print ("Cost after iteration %i: %f" %(i, cost))
			if print_cost and i % 100 == 0:
				costs.append(cost)

		return parameters

	def predict(self, X, y, parameters):
		"""
		This function is used to predict the results of a  L-layer neural network.

		Arguments:
		X -- data set of examples you would like to label
		parameters -- parameters of the trained model

		Returns:
		p -- predictions for the given dataset X
		"""

		#assert(X.shape[0] == self.num_layers[0])

		m = X.shape[1]
		n = len(parameters) // 2 # number of layers in the neural network
		p = np.zeros((1,m))

		# Forward propagation
		probas, caches = self.L_model_forward(X, parameters)


		# convert probas to 0/1 predictions
		for i in range(0, probas.shape[1]):
			if probas[0,i] > 0.5:
				p[0,i] = 1
			else:
				p[0,i] = 0

		#print results
		#print ("predictions: " + str(p))
		#print ("true labels: " + str(y))
		print("Accuracy: "  + str(np.sum((p == y)/m)))

		return p



	def initialize_parameters_deep(self, layer_sizes):
		"""
		Arguments: 
		num_layers -- python list

		Returns:
		parameters -- python dictionary containing the parameters "W1", "b1",...,"WL", "bL"
		Wl -- weight matrix of shape (layer_sizes[l], layer_sizes[l-1])
		bl -- bias vector of shape (lNayer_sizes[l], 1)
		"""
		parameters = {}
		L = self.num_layers

		for l in range(1, L):
			parameters["W" + str(l)] = np.random.randn(layer_sizes[l], layer_sizes[l-1]) / np.sqrt(layer_sizes[l-1])
			parameters["b" + str(l)] = np.zeros((layer_sizes[l], 1))

			assert(parameters["W" + str(l)].shape == (layer_sizes[l], layer_sizes[l-1]))
			assert(parameters["b" + str(l)].shape == (layer_sizes[l], 1))

		return parameters

	## FORWARD PROPAGATION

	## Helper functions for forward propagation	
	def linear_forward(self, A, W, b):
		"""
		Implement the linear equation of a layer's foward propoagation

		Arguments: 
		A -- activations from previous layer (or input data). Shape: (size of previous layer, number of examples)
		W -- weights matrix: numpy array. Shape: (size of current layer, size of previous layer)
		b -- bias vector: numpy array. Shape: (Size of current layer, 1)

		Outputs:
		Z -- the input of the activation function (pre-activation parameter)
		cache: python tuple containing "A", "W", "b". Stored for computer backprop efficiently
		"""

		Z = np.dot(W, A) + b

		assert(Z.shape == (W.shape[0], A.shape[1]))

		cache = (A, W, b)

		return Z, cache

	def linear_activation_forward(self, A_prev, W, b, activation):
		"""
		Implement forwardprop for Linear->activation layer

		Arguments:
		A -- activations from previous layer (or input data). Shape: (size of previous layer, number of examples)
		W -- weights matrix: numpy array. Shape: (size of current layer, size of previous layer)
		b -- bias vector: numpy array. Shape: (Size of current layer, 1)
		activation -- the activation function, stored as string. Either: "sigmoid" or "relu"

		Outputs:
		A -- the output of the activation function, post-activation value
		cache -- python tuple containing linear cache and activation cache, stored for computing backprop 
		"""

		if activation == "sigmoid":
			Z, linear_cache = self.linear_forward(A_prev, W, b)
			A, activation_cache = sigmoid(Z)

		if activation == "relu":
			Z, linear_cache = self.linear_forward(A_prev, W, b)
			A, activation_cache = relu(Z)

		assert(A.shape == (W.shape[0], A_prev.shape[1]))
		cache = (linear_cache, activation_cache)

		return A, cache


	## main feed forward method
	## NOTE this model performs RELU (L-1) times and ends with a sigmoid
	## Good for binary image classification
	def L_model_forward(self, X, parameters):
		"""	
		Implements forward propagation
		
		Arguments: 
		X -- data, numpy shape (input size, number of examples)
		parameters -- output of initialize_parameters_deep
		
		Outputs: 
		AL -- last post-activation value 
		caches: list of tuple from linear_activation_forward() indexed from 0 to L-1
		"""
		caches = []
		A = X 
		L = len(parameters) // 2


		##Implement relu for L-1 layers
		for l in range(1, L):
			A_prev = A
			A, cache = self.linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], "relu")
			caches.append(cache)

		##Implement sigmoid for output layer
		AL, cache = self.linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], "sigmoid")
		caches.append(cache)

		assert(AL.shape == (1, X.shape[1]))

		return AL, caches

	### BACKPROPAGATION
	def linear_backward(self, dZ, cache):
		"""
		Implements the linear portion of backprop for a single layer

		Arguments:
		dZ -- Gradient of the cost with respect to linear output of current layer l 
		cache --- tuple of 
		
		Outputs: 
		dA_prev -- Gradient of the cost wrt activaton of layer l-1
		dW -- gradient of the cost wrt W of current layer 
		db -- gradient of the cost wrt b of current layer
		"""
		A_prev, W, b = cache
		m = A_prev.shape[1]

		dW = (1/m)*np.dot(dZ, A_prev.T)
		db = (1/m)*np.sum(dZ, axis=1, keepdims=True)
		dA_prev = np.dot(W.T, dZ)

		assert (dA_prev.shape == A_prev.shape)
		assert (dW.shape == W.shape)
		assert (db.shape == b.shape)

		return dA_prev, dW, db

	def linear_activation_backward(self, dA, cache, activation):
		"""
		Implement the backward propagation for the LINEAR->ACTIVATION layer.

		Arguments:
		dA -- post-activation gradient for current layer l 
		cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
		activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

		Returns:
		dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
		dW -- Gradient of the cost with respect to W (current layer l), same shape as W
		db -- Gradient of the cost with respect to b (current layer l), same shape as b
		"""

		linear_cache, activation_cache = cache

		if activation == "relu":
			dZ = relu_backward(dA, activation_cache)
			dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

		if activation == "sigmoid":
			dZ = sigmoid_backward(dA, activation_cache)
			dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

		return dA_prev, dW, db


	def L_model_backward(self, AL, Y, caches):
		"""
		Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

		Arguments:
		AL -- probability vector, output of the forward propagation (L_model_forward())
		Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
		caches -- list of caches containing:
		            every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
		            the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])

		Returns:
		grads -- A dictionary with the gradients
		         grads["dA" + str(l)] = ... 
		         grads["dW" + str(l)] = ...
		         grads["db" + str(l)] = ... 
		"""

		grads = {}
		L = len(caches)
		m = AL.shape[1]
		Y = Y.reshape(AL.shape)

		#Initializing the backpropagation
		dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

		##Lth layer (Sigmoid -> Linear gradients)
		current_cache = caches[L-1]
		grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = self.linear_activation_backward(dAL, current_cache, "sigmoid")

		##Loop from l = L-2 to l = 0
		for l in reversed(range(L-1)):
			current_cache = caches[l]
			dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(grads["dA" + str(l+1)], current_cache, "relu")

			grads["dA" + str(l)] = dA_prev_temp
			grads["dW" + str(l + 1)] = dW_temp
			grads["db" + str(l + 1)] = db_temp

		return grads

	##UPDATE PARAMETERS
	def update_parameters(self, parameters, grads, learning_rate):
		"""
		Update parameters using gradient descent

		Arguments:
		parameters -- python dictionary containing your parameters 
		grads -- python dictionary containing your gradients, output of L_model_backward

		Returns:
		parameters -- python dictionary containing your updated parameters 
		              parameters["W" + str(l)] = ... 
		              parameters["b" + str(l)] = ...
		"""

		L = len(parameters) // 2

		#update for each layer
		for l in range(L):
			parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
			parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]

		return parameters


	## COST FUNCTIONS 
	def compute_cost(self, AL, Y):
		"""
		Implements cost function
		
		Inputs:
		AL -- probability vector corresponding to predictions, shape (1, number of examples)
		Y -- true label vector, shape (1, number of examples)

		cost -- cross-entropy cost
		"""
		m = Y.shape[1]

		##Compute loss
		cost = (-1/m) * (np.dot(Y, np.log(AL).T)+np.dot((1-Y), np.log(1-AL).T))

		cost = np.squeeze(cost)
		assert(cost.shape == ())

		return cost

### Helper Activation Functions
def sigmoid(Z):
	"""
	Implements sigmoid function 

	Arguments:
	Z -- pre-activation value

	Outputs: 
	A -- output of sigmoid(Z)
	cache -- returns Z as well
	"""

	A = 1/(1+np.exp(-Z))
	cache = Z

	return A, cache

def relu(Z):
	"""
	Implements RELU function 

	Arguments:
	Z -- pre-activation value

	Outputs: 
	A -- output of sigmoid(Z)
	cache -- returns Z as well
	"""

	A = np.maximum(0, Z)
	assert(A.shape == Z.shape)
	cache = Z

	return A, cache

def relu_backward(dA, cache):
	"""
	Implement the backward propagation for a single RELU unit.

	Arguments:
	dA -- post-activation gradient, of any shape
	cache -- 'Z' where we store for computing backward propagation efficiently

	Returns:
	dZ -- Gradient of the cost with respect to Z
	"""

	Z = cache
	dZ = np.array(dA, copy=True) # just converting dz to a correct object.

	# When z <= 0, you should set dz to 0 as well. 
	dZ[Z <= 0] = 0

	assert (dZ.shape == Z.shape)

	return dZ

def sigmoid_backward(dA, cache):
	"""
	Implement the backward propagation for a single SIGMOID unit.

	Arguments:
	dA -- post-activation gradient, of any shape
	cache -- 'Z' where we store for computing backward propagation efficiently

	Returns:
	dZ -- Gradient of the cost with respect to Z
	"""

	Z = cache

	s = 1/(1+np.exp(-Z))
	dZ = dA * s * (1-s)

	assert (dZ.shape == Z.shape)

	return dZ


