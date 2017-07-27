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


import theano
import theano.tensor as T
import time
import pickle
import sys

debug = False
if debug:
	theano.config.optimizer='fast_compile'
	theano.config.exception_verbosity='high'
	theano.config.compute_test_value = 'warn'

class RNN_Theano(object):
	def __init__(self,word_dim,hidden_dim):
		U = np.random.uniform(-np.sqrt(1./word_dim),np.sqrt(1./word_dim),(word_dim,hidden_dim))
		V = np.random.uniform(-np.sqrt(1./hidden_dim),np.sqrt(1./hidden_dim),(hidden_dim,word_dim))
		W = np.random.uniform(-np.sqrt(1./hidden_dim),np.sqrt(1./hidden_dim),(hidden_dim,hidden_dim))
		state = np.zeros((1,hidden_dim))
		
		self.hidden_dim = hidden_dim
		self.word_dim = word_dim

		self.U = theano.shared(name='U',value=U.astype(theano.config.floatX))
		self.V = theano.shared(name='V',value=V.astype(theano.config.floatX))
		self.W = theano.shared(name='W',value=W.astype(theano.config.floatX))
		self.state = theano.shared(name='S',value=state.astype(theano.config.floatX))

		self.__theano_build__()

	@staticmethod
	def step_forward(xt,ot_prev,st_prev,U,V,W):
		# print st_prev.dimshuffle('x',0).type()
		# print st_prev.type()
		# print xt.type()
		# print U.type()
		# print V.type()
		# print W.type()
		st = T.tanh(T.dot(xt,U)+T.dot(st_prev,W))
		ot = T.nnet.softmax(T.dot(st,V))
		# print ot_prev.type()
		# print ot.type()
		# print st.type()
		# a = T.dot(xt,U)
		# if debug: print a.tag.test_value.shape
		# b = T.dot(st_prev,W)
		# if debug: print b.tag.test_value.shape
		# c = a+b
		# st = T.tanh(c)
		# ot = T.nnet.softmax(T.dot(st,V))
		return ot,st

	def __theano_build__(self):
		x = T.dtensor3('x')
		if debug: x.tag.test_value = np.random.random((10,1,self.word_dim))
		y = T.lvector('y')
		if debug:
			tst = np.zeros((self.word_dim),dtype=np.int)
			tst[np.random.randint(self.word_dim)] = 1
			y.tag.test_value = tst

		# s = T.matrix('s')

		results, updates = theano.scan(RNN_Theano.step_forward,sequences=x,
			outputs_info=[dict(initial=T.patternbroadcast(T.zeros((1,self.word_dim)),(False,False))),
			dict(initial=self.state)],
			non_sequences=[self.U,self.V,self.W],strict=True,truncate_gradient=9)

		o = results[0][-1]
		s = results[1]
		# print o.type(),y.type()
		if debug: print o.tag.test_value
		# theano.pp(o)
		# time.sleep(1)
		self.prediction = T.argmax(o,axis=1)
		o_error = T.sum(T.nnet.categorical_crossentropy(o,y))

		self.dU = T.grad(o_error,self.U)
		self.dV = T.grad(o_error,self.V)
		self.dW = T.grad(o_error,self.W)

		self.forward_propogation = theano.function([x],[o,s],updates=[(self.state,s[-1])])
		self.predict = theano.function([x],self.prediction)
		self.ce_error = theano.function([x,y],o_error)
		self.bptt = theano.function([x,y],[self.dU,self.dV,self.dW])

		learning_rate = T.scalar('learning_rate')
		learning_rate.tag.test_value = 0.1
		self.sgd_step = theano.function([x,y,learning_rate],[],
			updates=[(self.U, self.U - learning_rate * self.dU),
					(self.V,self.V - learning_rate * self.dV),
					(self.W, self.W - learning_rate * self.dW),
					(self.state,s[-1])])
	def save(self,file):
		sys.setrecursionlimit(5000)
		with open(file,'wb') as f:
			pickle.dump(self,f)
		return
