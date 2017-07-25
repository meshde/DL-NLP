import numpy as np
import pickle
from helper import form_one_hot
from helper import get_one_hot
from helper import get_char
from helper import load_dic

FILE = 'tst4.pkl'
INIT = 'We are accounted poor citizens, the patricians good.What authority surfeits on would relieve us'
DIM = 27
def get_model(filename):
	with open(filename,'rb') as f:
		result = pickle.load(f)
	return result

def vectorize_statement(line):
	result = []
	count = 0
	for char in line:
		# print char
		one_hot = form_one_hot(char)
		# print type(one_hot)
		if np.array_equal(one_hot,np.zeros(DIM,dtype=np.int64)):
			continue
		# print one_hot
		result.append(one_hot)
		count += 1
	# print count
	return np.array(result).reshape((count,1,DIM))

def get_word(index):
	if index == 26:
		return ' '
	return chr(index+ord('a'))
def generate(model,inp):
	result = []
	for i in range(100):
		vector = vectorize_statement(inp)
		temp = model.predict(vector)
		out = get_word(temp)
		result += out
		inp = out
	return result

def test(model,inp):
	result = []
	for i in range(100):
		vector = vectorize_statement(inp)
		res,state = model.forward_propogation(vector)
		print 'Result:'
		print res
		print 'State:'
		print state
		out = get_word(np.argmax(res[0]))
		result += out
		inp = out
	return result 

def main():
	global FILE
	global INIT
	model = get_model(FILE)
	# print generate(model,INIT)
	print test(model,INIT)
if __name__ == '__main__':
	main()
