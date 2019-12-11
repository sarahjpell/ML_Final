#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 12:41:19 2019

@author: sarahpell
"""
#following guide from: 
#https://stackabuse.com/text-generation-with-python-and-tensorflow-keras/
from __future__ import absolute_import, division, print_function, unicode_literals

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import re
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout, LSTM, Bidirectional, Activation
from keras.utils import to_categorical

from nltk.tokenize import word_tokenize

X = []
y = []
#salad, soup, cookies, sandwich, cake, pasta
data = pd.read_csv(r'RAW_recipes.csv')
instructions = data['steps']
#print(descs)

for i in instructions:
    if type(i) is str:
        i = re.sub(r'\[[0-9]*\]', ' ', i)
        i = re.sub(r'\s+', ' ', i)
        i = re.sub('[^a-zA-Z]', ' ', i)
        i = re.sub(r'\s+', ' ', i)
        
        if 'salad' in i:
            y.append('salad')
            X.append(i)
        elif 'soup' in i:
            y.append('soup')
            X.append(i)
        elif 'cookies' in i:
            y.append('cookies')
            X.append(i)
        elif 'sandwich' in i:
            y.append('sandwich')
            X.append(i)
        elif 'cake' in i:
            y.append('cake')
            X.append(i)
        elif 'pasta' in i:
            y.append('pasta')
            X.append(i)
        else:
            pass
    else:
        pass
    

le = preprocessing.LabelEncoder()
y = le.fit_transform(y)
y = to_categorical(y, num_classes=None)

all_words = []
for val in X:
    tokenized_sent = word_tokenize(val)
    for word in tokenized_sent:
        all_words.append(word)
        
unique_words = set(all_words)
#print(len(unique_words))

vocab_len = len(unique_words) + 10
embedded_sentences = [one_hot(sent, vocab_len) for sent in X]
#print(embedded_sentences )

word_ct = lambda sentence: len(word_tokenize(sentence))
long_sent = max(X, key=word_ct)
len_long_sent = len(word_tokenize(long_sent))


padded_sentences = pad_sequences(embedded_sentences, len_long_sent, padding='post')
padded_sentences = padded_sentences.reshape(1009,len_long_sent,1)


p = []
for i in range(0,len(padded_sentences)):
    p.append(padded_sentences[i])

#cr_val_score = cross_val_score(model, padded_sentences, y, cv = 10)
#model = Sequential()
#model.add(LSTM(256, input_shape=(padded_sentences[1], padded_sentences.shape[2]), return_sequences=True))
#model.add(Dropout(0.2))
#model.add(LSTM(256, return_sequences=True))
#model.add(Dropout(0.2))
#model.add(LSTM(128))
#model.add(Dropout(0.2))
#model.add(Dense(y.shape[1], activation='softmax'))
#model.add(Flatten())
#model.add(Dense(6, activation='sigmoid'))



model = Sequential()

cells = [
LSTM(100),
LSTM(100),
LSTM(100),
]

model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=((len_long_sent,1))))
model.add(Bidirectional(LSTM(128)))
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(6))
model.add(Activation('softmax'))

#filepath = "recipe_class_model3.hdf5"
#checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1,
#                             save_best_only=True, mode='min')
#desired_callbacks = [checkpoint]
#
#
#filename = "recipe_class_model2.hdf5"
#model.load_weights(filename)
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
X_train, X_test, y_train, y_test = train_test_split(padded_sentences, y, test_size=0.2)
history = model.fit(X_train, y_train, epochs=20, verbose=1, callbacks=desired_callbacks, validation_data=(X_test,y_test))
# loss, accuracy = model.evaluate(padded_sentences, y, verbose=0)
# print('Accuracy: %f' % (accuracy*100))
#
#import pickle
#file_Name = "history3.pkl"
## open the file for writing
#fileObject = open(file_Name,'wb') 
#pickle.dump(history,fileObject) 
#

#history1 = pickle.load( open( "history.pkl", "rb" ) )
#history2 = pickle.load( open( "history2.pkl", "rb" ) )
#
#history_acc = []
#history_loss = []
#
#for acc in history1.history['accuracy']:
##    print(acc)
#    history_acc.append(acc)
#    
#for acc2 in history2.history['accuracy']:
##    print(acc2)
#    history_acc.append(acc2)
##print('-------------------------------')
#for loss in history1.history['loss']:
##    print(loss)
#    history_loss.append(loss)
#    
#for loss2 in history2.history['loss']:
##    print(loss2)
#    history_loss.append(loss2)
#
#fig, axes = plt.subplots(figsize=(13, 4.5), ncols=2)
##print(history.history.keys())
#axes[0].plot(history_acc, color='blue', label='accuracy')
## plt.legend(loc="upper left")
#axes[1].plot(history_loss, color='orange', label='loss')
## plt.legend(loc="upper left")
#for ax in axes:
#    ax.set(xlabel='Epochs')
#axes[0].set(ylabel='accuracy')
#axes[1].set(ylabel='loss')
#    
