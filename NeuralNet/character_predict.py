import numpy as np
import pickle
from helper import form_one_hot
from helper import get_one_hot
from helper import get_char
from helper import load_dic

FILE = 'tst6.pkl'
INIT = 'We are accounted poor citizens, the patricians good.What authority surfeits on would relieve us'

def get_model(filename):
	with open(filename,'rb') as f:
		result = pickle.load(f)
	return result

def vectorize_statement(line):
	result = []
	count = 0
	for char in line:
		one_hot = get_one_hot(char,dic)
		result.append(one_hot)
		count += 1
	return np.array(result).reshape((count,1,len(dic)))

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
		out = get_char(np.argmax(res[0]),dic)
		result += out
		inp = out
	return result 

def main():
	global FILE
	global INIT
	global dic
	dic = load_dic('shake.pkl')
	model = get_model(FILE)
	# print generate(model,INIT)
	print test(model,INIT)
if __name__ == '__main__':
	main()
