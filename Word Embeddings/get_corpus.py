import os
import sys
global BASE
def get_from_derictory(dire):
	# print "Hello"
	for file in os.listdir(dire):
		path = os.path.join(dire,file)
		if os.path.isdir(path):
			get_from_derictory(path)
		else:
			ext = os.path.splitext(file)[-1]
			if ext == '.txt':
				# print "HERE"
				os.system('cp '+path+' '+BASE)
	return

def main():
	path = os.path.abspath(sys.argv[1])
	get_from_derictory(path)

if __name__ == '__main__':
	global BASE
	BASE = "~/Mehmood/DL-NLP/Word\ Embeddings/Corpus/"
	BASE = os.path.join(BASE,sys.argv[2])
	os.system('mkdir '+BASE)
	main()