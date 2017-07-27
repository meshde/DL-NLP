from NNet import RNN_Theano as RNN
from helper import form_one_hot
from helper import get_one_hot
from helper import load_dic
import numpy as np
import theano
import theano.tensor as T
import time
import sys
import traceback
import pickle


def main():
	dic = load_dic('shake.pkl')
	DIM = len(dic)
	model = RNN(DIM,64)
	with open('shake.txt','r') as f:
		for line in f:
			line = line.strip()
			for i,char in enumerate(line):
				try:
					x = get_one_hot(char,dic)
					y = get_one_hot(line[i+1],dic)
				except IndexError:
					# print Exception
					continue
				learning_rate = 0.4
				x = np.reshape(x,(1,1,DIM))
				model.sgd_step(x,y,learning_rate)
	model.save('tst5.pkl')
	return

if __name__ == '__main__':
	s = time.time()
	main()
	print time.time()-s