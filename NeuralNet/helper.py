import time
import pickle

def create_word_dic(input,output):
	d = dict()
	count = 0
	with open(input,'r') as f:
		for line in f:
			for char in line:
				if char not in d:
					d[char] = count
					count += 1
	print d
	# print time.time()-s

	with open(output,'w') as f:
		pickle.dump(d,f)
	return

def form_one_hot(char,DIM):
	char = char.upper()
	result = np.zeros(DIM,dtype=np.int64)
	if char == ' ':
		result[-1] = 1
		return result
	if ord('Z') >= ord(char) and ord('A') <= ord(char):
		result[ord(char)-ord('A')] = 1
		return result
	return result

def load_dic(file):
	with open(file,'r') as f:
		result = pickle.load(f)
	return result

def get_one_hot(char,dic):
	result = np.zeros(len(dic),dtype=np.int64)
	result[dic[char]] = 1
	return result

def get_char(index,dic):
	for key in dic:
		if dic[key] == index:
			return key
	return None