import gensim
import os
import pickle
import re
import numpy as np
import sys
from optparse import OptionParser
from keras.models import Sequential
from keras.models import Model
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Input
from keras.layers import Dense
from keras.layers import dot
from keras.layers import concatenate
from keras.preprocessing.sequence import pad_sequences

usage = ''

def get_options():
    parser = OptionParser()

    parser.add_option('-w','--word2vec',dest='word2vec',action='store',help="location of pre-trained word2vec model")
    parser.add_option('-t','--task',dest='task',action='store',help="bAbI task id")
    parser.add_option('-m','--map',dest='map',action='store',default='Tasks/en',help="location of the file that maps task id to it's file location")
    parser.add_option('-s','--onlysupporting',dest='only_supporting',action='store_true',default=False,
        help="set if only supporting sentences should be used to train the model")
    parser.add_option('-i','--model-index',dest='model_index',action='store',default='1',help="Selects the model to be trained."+usage)
    
    options,args = parser.parse_args()
    return options,args

def get_word2vec_model(file):
    path = os.path.abspath('../Word Embeddings/Models')
    path = os.path.join(path,file)
    model = gensim.models.Word2Vec.load(path)
    return model

def get_challenge(file,task):
    """ 'file' is the location of the file where the dictionary which maps task id to task loaction was pickled """
    with open(file,'rb') as f:
        dic = pickle.load(f)
    return dic[task]

def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]

def parse_stories(lines,only_supporting=False):
    """ This function parses the bAbI task file and returns a dictionary containing three 2D matrices labeled 'story', 'question' and 'answer' 
    In each of the three matrices, the rows represent separate training examples and the columns represent the sequence of words.
    A training example is a triplet of story,question and answer. """
    data = dict()
    data['story'] = []
    data['question'] = []
    data['answer'] = []
    for line in lines:
        nid,line = line.split(' ',1)
        nid = int(nid)
        if nid == 1:
            """ In the bAbI task a line with id as 1 indicates the start of a new story """
            story = []
        if '\t' in line:
            """ This means that the line contains a question, it's answer and the supporting statement's line no, all separated by a tab """
            question,answer,supporting = line.split('\t')
            if only_supporting:
                supporting = map(int,supporting.split())
                substory = [story[i-1] for i in supporting]

            else:
                """ Else it's just a statement """
                substory = [x for x in story if x]
            substory = reduce(lambda x,y:x+y,substory)
            data['story'].append(substory)
            data['question'].append(tokenize(question))
            data['answer'].append(answer)
            story.append('')    # This is added so that the indices in our story matrix match that given in the task

            # data.append({'story':substory,'question':tokenize(question),'answer':answer})
        else:
            sent = tokenize(line)
            story.append(sent)
    return data

def vectorize(data,word2vec,story_maxlen,question_maxlen):
    """ This function replaces the words in the story and question sequences with their vectors """
    res = dict()
    res['story'] = []
    res['question'] = []
    res['answer'] = []
    for key in ['story','question']:
        for example in data[key]:
            entry = []
            for word in example:
                entry.append(word2vec[word])
            res[key].append(entry)
        res[key] = np.array(res[key])
    for ans in data['answer']:
        res['answer'].append(word2vec[ans])
    res['answer'] = np.array(res['answer'])
    res['story'] = pad_sequences(res['story'],story_maxlen)
    # try:
    #     res['story'] = pad_sequences(res['story'],story_maxlen)
    # except:
    #     print res['story']
    #     exit()
    res['question'] = pad_sequences(res['question'],question_maxlen)
    return res

def test():
    print os.getcwd()
	# print os.path.abspath('../')

def set_path():
    if not os.getcwd().endswith('bAbI'):
        os.chdir('bAbI')

def get_model(index,story_maxlen,question_maxlen,word2vec):
    if index == '1':
        input_story = Input((story_maxlen,word2vec.vector_size))
        input_question = Input((question_maxlen,word2vec.vector_size))

        story_embedding = LSTM(64)(input_story)
        story_embedding = Dropout(0.3)(story_embedding)
        question_embedding = LSTM(64)(input_question)
        question_embedding = Dropout(0.3)(question_embedding)

        output = Dense(100)(concatenate([story_embedding,question_embedding]))
        output = Dropout(0.2)(output)


        model = Model([input_story,input_question],output)
        model.compile(optimizer='rmsprop',loss='cosine_proximity',metrics=['accuracy'])

        return model
    
    if index == '2':
        input_story = Input((story_maxlen,word2vec.vector_size))
        input_question = Input((question_maxlen,word2vec.vector_size))

        story_embedding = LSTM(64)(input_story)
        story_embedding = Dropout(0.3)(story_embedding)
        question_embedding = LSTM(64)(input_question)
        question_embedding = Dropout(0.3)(question_embedding)

        output = Dense(100)(concatenate([story_embedding,question_embedding]))
        output = Dropout(0.2)(output)

        output = word2vec.most_similar(positive=[output],negative=[],topn=1)
        output = word2vec[output[0][0]]

        model = Model([input_story,input_question],output)
        model.compile(optimizer='rmsprop',loss='cosine_proximity',metrics=['accuracy'])

        print model

        return model

def main():
    set_path()
    options,args = get_options()
    wv = get_word2vec_model(options.word2vec)
    challenge = get_challenge(options.map,options.task)

    print options.model_index

    with open(challenge.format('train'),'r')as train,open(challenge.format('test'),'r') as test:
        train_data = parse_stories(train.readlines(),only_supporting=options.only_supporting)
        test_data = parse_stories(test.readlines(),only_supporting=options.only_supporting)
    
    story_maxlen = max(map(len,[x for x in train_data['story']+test_data['story']]))
    question_maxlen = max(map(len,[x for x in train_data['question']+test_data['question']]))

    train_data  = vectorize(train_data,wv,story_maxlen,question_maxlen)
    test_data = vectorize(test_data,wv,story_maxlen,question_maxlen)

    model = get_model(options.model_index,story_maxlen,question_maxlen,wv)

    model.fit([train_data['story'],train_data['question']],train_data['answer'],batch_size=32,
        epochs=100,validation_data=([test_data['story'],test_data['question']],test_data['answer']))
	# test()

if __name__ == '__main__':
    global usage
    usage = """The models are as follows:
    1. Story passed through an LSTM, Question passed through another LSTM, States of both LSTMs are then concatenated and then passed through a hidden layer, Output of hidden layer then compared with anser using cosine proximity loss function
    2. Similar to Model 1, except of output if hidden layer is used to obtain the most similar word in word2vec model, the vector of this word is then used for the cosine loss function.
    3. """
    main()