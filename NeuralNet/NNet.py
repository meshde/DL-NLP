import numpy as np

class NeuralNet(object):
	SIGMOID = 0
	TANH = 1
	ReLU = 2
	SOFTMAX = 3
	CROSS_ENTROPY = 4
	def __init__(self,dimensions,N):
		self.Dx = dimensions[0]
		self.H = dimensions[1]
		self.Dy = dimensions[2]
		self.N = N
		self.activation = NeuralNet.SIGMOID
		self.output = NeuralNet.SOFTMAX
		self.error = NeuralNet.CROSS_ENTROPY
	def set_input(self,X,Y=None):
		self.X = X
		self.Y = Y
		return
	def set_alpha(self,alpha):
		self.alpha = alpha
		return
	def initialise_parameters(self):
		# W1 has dimensions: Dx x H
		self.W1 = numpy.random.randn(self.Dx,self.H)
		# b1 has dimensions: 1 x H
		self.b1 = numpy.random.randn(1,self.H)
		# W2 has diemsions: H x Dy
		self.W2 = numpy.random.randn(self.H,self.Dy)
		# b2 has diemsions: 1 x Dy
		self.b2 = numpy.random.randn(1,self.Dy)
		return
	def forward_propogate_batch(self):
		""""""
		z = self.X.dot(self.W1)
		self.h = self.activation_function(z)
		# h has dimensions: N x H
		assert self.h.shape == (self.N,self.H)
		z = self.h.dot(self.W2)
		self.ycap = self.output_function(z)
		return self.ycap
	def activation_function(self,mat):
		if self.activation == NeuralNet.SIGMOID:
			return NeuralNet.sigmoid(mat)
		if self.activation == NeuralNet.TANH:
			return np.tanh(mat)
	@staticmethod
	def sigmoid(mat):
		return 1/(1+np.exp(mat))
	@staticmethod
	def sigmoid_grad(mat):
		return sigmoid(mat)-np.power(sigmoid(mat),2)
	def output_function(self):
		if self.output == NeuralNet.SOFTMAX:
			return NeuralNet.softmax(mat)
	@staticmethod
	def softmax(mat):
		xmax = np.max(mat,axis=1,keepdims=True)
		z = np.exp(mat-xmax)
		return z/np.sum(z,axis=1,keepdims=True)
	@staticmethod
	def cross_entropy(ycap,y):
		return np.sum(np.multiply(y,np.log(ycap)))
	@staticmethod
	def grad_cross_softmax(ycap,y):
		return ycap-y
	def backward_propogate_batch(self):
		""""""
		grad_o = self.output_grad()
		grad_W2 = self.h.T.dot(grad_o)
		grad_b2 = np.sum(grad_o,axis=0,keepdims=True)
		hprime = self.activation_grad()
		z = grad_o.dot(self.W2.T)
		z = np.multiply(z,hprime)
		grad_W1 = self.X.T.dot(z)
		grad_b1 = self.np.sum(z,axis=0,keepdims=True)

		W1 -= np.multiply(grad_W1,self.alpha)
		W2 -= np.multiply(grad_W2,self.alpha)
		b1 -= np.multiply(grad_b1,self.alpha)
		b2 -= np.multiply(grad_b2,self.alpha)
		
	def activation_grad(self):
		if self.activation == NeuralNet.SIGMOID:
			return NeuralNet.sigmoid_grad(self.h)
	def output_grad(self):
		if self.output == NeuralNet.SOFTMAX and self.error == NeuralNet.CROSS_ENTROPY:
			return NeuralNet.grad_cross_softmax(self.ycap,self.Y)



class RNN(object):
	def __init__(self,dimensions):
		self.Dx = dimesions[0]
		self.H = dimensions[1]
		n1 = np.sqrt(1/self.Dx)
		n2 = np.sqrt(1/self.H)
		self.activation = NeuralNet.SIGMOID
		self.U = np.random.uniform(-n1,n1,(self.H,self.Dx))
		self.W = np.random.uniform(-n2,n2,(self.H,self.H))
		self.V = np.random.uniform(-n2,n2,(self.Dx,self.H))
	def set_input(self,x,y=None):
		self.X = x
		self.T = len(self.X)
		return
	def set_activation(self,method):
		self.activation = method
		return
	def activation_function(self,mat):
		if self.activation == NeuralNet.SIGMOID:
			return NeuralNet.sigmoid(mat)
		if self.activation == NeuralNet.TANH:
			return np.tanh(mat)
	def forward_propogate(self):
		self.s = np.zeros(self.T+1,self.H)
		self.o = np.zeros(self.T,self.Dx)
		for t in range(1,self.T+1):
			self.s[t] = self.activation_function(np.dot(self.U,self.X)+np.dot(self.W,self.s[t-1]))
			self.o[t] = NeuralNet.softmax(np.dot(self.V,self.s[t]))
		return
	def predict(self,x):
		self.set_input(x)
		self.forward_propogate()
		return np.argmax(self.o,axis=1)
	def loss_function(self):
		return NeuralNet.cross_entropy(self.X,self.Y)