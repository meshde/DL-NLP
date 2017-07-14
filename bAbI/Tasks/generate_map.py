from optparse import OptionParser
import pickle
import os

def main():
	
	parser = OptionParser()
	parser.add_option('-t','--type',dest='type',action='store')
	options,args = parser.parse_args()

	BASE = os.path.abspath('tasks_1-20_v1-2')
	path = os.path.join(BASE,options.type)

	dic = dict()

	for fname in os.listdir(path):
		index,task,_ = fname.split('_')
		index = index[2:]
		if index not in dic:
			dic[index] = 'qa'+index+'_'+task+'_{}.txt'
			dic[index] = os.path.join(os.path.join('Tasks/tasks_1-20_v1-2',options.type),dic[index])
	# print dic

	with open(options.type,'wb') as f:
		pickle.dump(dic,f)

	return


if __name__ == '__main__':
	main()
