#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 23:43:54 2019

@author: sarahpell
"""

from keras import backend as K
K.clear_session()

import numpy
import sys
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
import pandas as pd
import numpy as np

def select(preds, temperature=1.0):
# helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


data1 = pd.read_json(r'recipes_raw_nosource_ar.json')
# data2 = pd.read_json(r'recipes_raw_nosource_epi.json')
# data3 = pd.read_json(r'recipes_raw_nosource_fn.json')

X = []
y = []
recipes = ''
ct=0
# len(data1.keys())
for r1 in data1.keys()[:1000]:
    if type(data1[r1]['instructions']) is str:
        recipes += 'instructions: ' + data1[r1]['instructions']
# for r2 in data2.keys():
#     if type(data2[r2]['instructions']) is str:
#         recipes += 'instructions: ' + data2[r2]['instructions'].rstrip()
# for r3 in data3.keys():
#     if type(data3[r3]['instructions']) is str:
#         recipes += 'instructions: ' + data3[r3]['instructions'].rstrip()
# print(recipes)
               
        
       
        
tokenized_recipes = tokenize_txt(recipes)

char_list = sorted(list(set(tokenized_recipes)))
print(char_list)
char_cts = dict((c, i) for i, c in enumerate(char_list))
print(char_cts)

input_len = len(tokenized_recipes)
num_chars = len(char_list)
print ("Total number of characters:", input_len)
print ("Total vocab:", num_chars)


seq_length = 100
Xdata = []
ydata = []

for i in range(0, input_len - seq_length, 1):
    in_seq = tokenized_recipes[i:i + seq_length]
    out_seq = tokenized_recipes[i + seq_length]

    Xdata.append([char_cts[char] for char in in_seq])
    ydata.append(char_cts[out_seq])

num_patterns = len(Xdata)
print('number of patterns: ', num_patterns)

X = numpy.reshape(Xdata, (num_patterns, seq_length, 1))
X = X/float(num_chars)
y = np_utils.to_categorical(ydata)

model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', 
              optimizer='adam')


filepath = "model_weights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
desired_callbacks = [checkpoint]

scores = model.fit(X, y, epochs=40, batch_size=256, callbacks=desired_callbacks)

num_to_char = dict((i, c) for i, c in enumerate(char_list))

start = numpy.random.randint(0, len(Xdata) - 1)
pattern = Xdata[start]
print("Random Seed:")
print("\"", ''.join([num_to_char[value] for value in pattern]), "\"")


for i in range(1000):
    x = numpy.reshape(pattern, (1, len(pattern), 1))
    x = x / float(num_chars)
    prediction = model.predict(x, verbose=0)[0]
    diversity = 0.2
    index = select(prediction, diversity)
#     index = numpy.argmax(prediction)
    result = num_to_char[index]
    seq_in = [num_to_char[value] for value in pattern]

    sys.stdout.write(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
    
    
#model.summary()