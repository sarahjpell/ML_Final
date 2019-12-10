#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 12:41:19 2019

@author: sarahpell
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import pandas as pd

from sklearn import preprocessing
import re
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout, LSTM, Bidirectional, Activation
from keras.utils import to_categorical


X = []
y = []
#salad, soup, cookies, sandwich, cake, pasta
data = pd.read_csv(r'RAW_recipes.csv')
data.head()
descs = data['description'][:5000]

#print(descs)


for d in descs:
    if type(d) is str:
        d = re.sub(r'\[[0-9]*\]', ' ', d)
        d = re.sub(r'\s+', ' ', d)
        d = re.sub('[^a-zA-Z]', ' ', d )
        d = re.sub(r'\s+', ' ', d)
        
        if 'salad' in d:
            y.append('salad')
            X.append(d)
        elif 'soup' in d:
            y.append('soup')
            X.append(d)
        elif 'cookies' in d:
            y.append('cookies')
            X.append(d)
        elif 'sandwich' in d:
            y.append('sandwich')
            X.append(d)
        elif 'cake' in d:
            y.append('cake')
            X.append(d)
        elif 'pasta' in d:
            y.append('pasta')
            X.append(d)
        else:
            pass
    else:
        pass
    




from nltk.tokenize import word_tokenize
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

word_count = lambda sentence: len(word_tokenize(sentence))
longest_sentence = max(X, key=word_count)
length_long_sentence = len(word_tokenize(longest_sentence))


padded_sentences = pad_sequences(embedded_sentences, length_long_sentence, padding='post')
padded_sentences = padded_sentences.reshape(1009,length_long_sentence,1)


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

model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=((length_long_sentence,1))))
model.add(Bidirectional(LSTM(128)))
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(6))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

model.fit(padded_sentences, y, epochs=1, verbose=1)
loss, accuracy = model.evaluate(padded_sentences, y, verbose=0)
print('Accuracy: %f' % (accuracy*100))

