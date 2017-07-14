from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

import numpy as np
import sys
import time
# import nltk
last = time.time()

def get_time():
	global last
	now = time.time()
	print now-last
	last = now

file = sys.argv[1]
txt = open(file,'r').read().lower()
chars = sorted(list(set(txt)))
char_to_int = dict((c,i) for i,c in enumerate(chars))
n_chars = len(txt)
n_vocab = len(chars)
print "Total Characters:",n_chars
print "Total Vocab:",n_vocab
get_time()
seq_length = int(sys.argv[2])
dataX = []
dataY = []

for i in range(0,n_chars - seq_length):
	dataX.append([char_to_int[c] for c in txt[i:i+seq_length]])
	dataY.append(char_to_int[txt[i+seq_length]])
n_patterns = len(dataX)
print "Total Patterns:",n_patterns

get_time()

X = np.reshape(dataX,(n_patterns,seq_length,1))
X = X/float(n_vocab)

y = np_utils.to_categorical(dataY)
print y.shape

# sys.exit(0)

model = Sequential()
model.add(LSTM(256,input_shape=(X.shape[1],X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1],activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam')
print "Model Compiled"
get_time()

filepath = "weights-ml-mastery-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath,monitor='loss',verbose=1,save_best_only=True,mode='min')
callbacks_list = [checkpoint]

model.fit(X,y,epochs=20,batch_size=128,callbacks=callbacks_list)
print "Done"
get_time()