import numpy as np
import sklearn.datasets
# import sklearn.
from NNet import NeuralNet as NN
import sys

N = int(sys.argv[1])
H = int(sys.argv[2])

X,Y = sklearn.datasets.make_moons((4*N)//3,noise=0.2)
trainX,trainY = X[:N],Y[:N]
testX,testY = X[N:],Y[N:]

net = NN((2,H,2),N)
net.set_input_output(trainX,trainY)
net.initialise_parameters()

y = net.forward_propogate_batch()

