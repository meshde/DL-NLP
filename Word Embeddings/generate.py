import gensim
import sys
import os
import time
class MySentences(object):
	def __init__(self,path):
		self.path = path
	def __iter__(self):
		if os.path.isdir(self.path):
			for file in os.listdir(self.path):
				for line in open(os.path.join(self.path,file)):
					yield line.strip().split()
		else:
			for line in open(self.path):
				yield line.strip().split()
def get_details(file):
	dic = dict()
	with open(file,'r') as f:
		for line in f:
			lhs,rhs = line.strip().split()
			dic[lhs] = rhs
	return dic
def get_filename(corpus,dic):
	name = 'corpus-'+corpus.split('.')[0]
	for key in sorted(dic.keys()):
		name += '+'
		name += key+'-'+dic[key]
	return name #+'+tst'
def main():
	corpus = os.path.abspath(sys.argv[2])
	dic = get_details(sys.argv[1])
	sentences = MySentences(corpus)
	# sentences = []
	# for line in open(corpus,'r'):
		# sentences.append(line.split())
	fname = get_filename(corpus.split('/')[-1],dic)
	# dic['sentences'] = sentences
	start = time.time()
	model = gensim.models.Word2Vec(sentences,**dic)
	model.save(os.path.join('Models',fname))
	print "Time Taken:",time.time()-start
	# model.doesnt_match("breakfast cereal dinner lunch".split())
	# model.similarity('woman','man')
	# print (model.wv.vocab)

if __name__ == "__main__":
	main()