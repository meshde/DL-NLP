import numpy as np
import pickle
from character_train import form_one_hot 

FILE = 'tst2.pkl'
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
	print count
	return np.array(result).reshape((count,1,DIM))

def get_word(index):
	if index == 26:
		return ' '
	return chr(index+ord('a'))
def generate(model,inp):
	result = []
	for i in range(100):
		vector = vectorize_statement(inp)
		out = get_word(model.predict(vector)[0])
		result += out
		inp = out
	return result

def main():
	global FILE
	global INIT
	model = get_model(FILE)
	print generate(model,INIT)
if __name__ == '__main__':
	main()
