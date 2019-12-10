import json
import os
import numpy
import sys
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
# import nltk
# nltk.download('stopwords')
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
# import tensorflow
import matplotlib as plt
import pickle

##############FUNCTIONS################################
def tokenize_txt(txt):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(txt)
    filtered = filter(lambda token: token not in stopwords.words('english'), tokens)
    return " ".join(filtered)

########################################################

data_directory = 'recipes'
directory = os.fsencode(data_directory)  # establish directory
recipes = ''
ct = 0
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    filename = data_directory + '/' + filename
    # # don't parse .DS_Store file
    # if filename != "{}/.DS_Store".format(data_directory):
    if ct < 1:
        try:
            # try parsing as json
            f = open(filename)
            data = json.loads(f.read())
            for k in data.keys():
                try:
                    txt = (data[k]["instructions"])
                    if txt != None:
                        recipes = recipes + ' ' + txt.rstrip()
                        ct = ct + 1
                except KeyError:
                    pass

            f.close()
        except FileNotFoundError:
            pass
print('here!!!!!!!')
tokenized_recipes = tokenize_txt(recipes)


print(tokenized_recipes)
chars = sorted(list(set(tokenized_recipes)))
print(chars)
char_to_num = dict((c, i) for i, c in enumerate(chars))
print(char_to_num)

input_len = len(tokenized_recipes)
vocab_len = len(chars)
print ("Total number of characters:", input_len)
print ("Total vocab:", vocab_len)


seq_length = 100
Xdata = []
ydata = []

for i in range(0, input_len - seq_length):
    in_seq = tokenized_recipes[i:i + seq_length]
    out_seq = tokenized_recipes[i + seq_length]

    Xdata.append([char_to_num[char]] for char in in_seq)
    ydata.append([char_to_num[out_seq]])

num_patterns = len(Xdata)
print(num_patterns)

X = numpy.reshape(Xdata, (num_patterns, seq_length, 1))
X = X/float(vocab_len)
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
              optimizer='adam',
              metrics=['accuracy'])


filepath = "model_weights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1,
                             save_best_only=True, mode='min')
desired_callbacks = [checkpoint]


scores = model.fit(X, y, epochs = 10, batch_size = 32, callbacks = desired_callbacks, save_scores=True)
fig, axes = plt.subplots(figsize=(15, 6), ncols=2)
scores['accuracy'].plot(ax=axes[0], title='Train Accuracy')
scores['loss'].plot(ax=axes[1], title='Train Loss')
for ax in axes:
    ax.set(xlabel='Steps')

# filename = "model_weights_saved.hdf5"
# model.load_weights(filename)
# model.compile(loss='categorical_crossentropy', optimizer='adam')
#
# num_to_char = dict((i, c) for i, c in enumerate(chars))
#
# start = numpy.random.randint(0, len(Xdata) - 1)
# pattern = Xdata[start]
# print("Random Seed:")
# print("\"", ''.join([num_to_char[value] for value in pattern]), "\"")
#
# for i in range(1000):
#     x = numpy.reshape(pattern, (1, len(pattern), 1))
#     x = x / float(vocab_len)
#     prediction = model.predict(x, verbose=0)
#     index = numpy.argmax(prediction)
#     result = num_to_char[index]
#     seq_in = [num_to_char[value] for value in pattern]
#
#     sys.stdout.write(result)
#
#     pattern.append(index)
#     pattern = pattern[1:len(pattern)]
