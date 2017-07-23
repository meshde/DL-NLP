from NNet import RNN_Theano as RNN
import numpy as np
import theano
import theano.tensor as T
import time
import sys
import traceback

DIM = 27

def form_one_hot(char):
	char = char.upper()
	result = np.zeros(DIM,dtype=np.int64)
	if char == ' ':
		result[-1] = 1
		return result
	if ord('Z') >= ord(char) and ord('A') <= ord(char):
		result[ord(char)-ord('A')] = 1
		return result
	return result

def main():
	global DIM
	model = RNN(DIM,64)
	with open('shake.txt','r') as f:
		for line in f:
			line = line.strip()
			if line.endswith(':'):
				continue
			for i,char in enumerate(line):
				try:
					x = form_one_hot(char)
					if np.array_equal(x,np.zeros(DIM,dtype=np.int64)):
						continue
					y = form_one_hot(line[i+1])
				except Exception:
					# traceback.print_exc(file=sys.stdout)
					continue
				learning_rate = 0.4
				# print x
				x = np.reshape(x,(1,1,DIM))
				model.sgd_step(x,y,learning_rate)
	model.save('tst2.pkl')
	return

if __name__ == '__main__':
	s = time.time()
	main()
	print time.time()-s