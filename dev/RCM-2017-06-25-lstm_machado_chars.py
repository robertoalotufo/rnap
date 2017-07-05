# -*- coding: utf-8 -*-
# ----------------------------------------------
# Project:  ...
# Filename: xxx.py
#
#                     Rubens Machado, 2017-06-28
# ----------------------------------------------
# -*- coding: utf-8 -*-
from __future__ import print_function
import matplotlib.pyplot as plot

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM, BatchNormalization
from keras.optimizers import Adam
import keras.backend as K
from sklearn.model_selection import train_test_split

import numpy as np
import random
import glob
import sys
import os
import re

sys.path.append('../src')
from my_keras_utilities import (get_available_gpus,
                                load_model_and_history,
                                save_model_and_history,
                                TrainingPlotter)

os.makedirs('../../models', exist_ok=True)
np.set_printoptions(precision=3, linewidth=120)


# K.set_image_data_format('channels_first')
K.set_floatx('float32')

print('Backend:        {}'.format(K.backend()))
print('Data format:    {}'.format(K.image_data_format()))
print('Available GPUS:', get_available_gpus())
print(sys.executable, sys.getdefaultencoding())


class MyCb(TrainingPlotter):
    
    def on_epoch_end(self, epoch, logs={}):
        super().on_epoch_end(epoch, logs)


def train_network(model, model_name, X_train, y_train, Xval=None, yval=None, 
                  opt='rmsprop', batch_size=60, nepochs=50000, patience=500, nr_seed=20170522, 
                  reset=False, ploss=1.0):

    do_plot = (ploss > 0.0)
    
    model_fn = model_name + '.model'
    if reset and os.path.isfile(model_fn):
        os.unlink(model_name + '.model')
        
    if not os.path.isfile(model_fn):
        # initialize the optimizer and model
        print("[INFO] compiling model...")
        model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])    

        # History, checkpoint, earlystop, plot losses:
        cb = MyCb(n=1, filepath=model_name, patience=patience, plot_losses=do_plot)
        
    else:
        print("[INFO] loading model...")
        model, cb = load_model_and_history(model_name)
        cb.patience = patience

    past_epochs = cb.get_nepochs()
    tr_epochs = nepochs - past_epochs
    
    if do_plot:
        vv = 0
        fig = plot.figure(figsize=(15,6))
        plot.ylim(0.0, ploss)
        plot.xlim(0, nepochs)
        plot.grid(True)
    else:
        vv = 2
        
    if Xval is not None:
        val_data = (Xval, yval)
    else:
        val_data = None

    print("[INFO] training for {} epochs...".format(tr_epochs))
    try:
        h = model.fit(X_train, y_train, batch_size=batch_size, epochs=tr_epochs, verbose=vv, 
                      validation_data=val_data, callbacks=[cb])
    except KeyboardInterrupt:
        pass

    return model, cb


def test_network(model_name, X_test, y_test):
    model, histo = load_model_and_history(model_name)
    print('Model from epoch {}'.format(histo.best_epoch))
    print("[INFO] evaluating in the test data set ...")
    loss, accuracy = model.evaluate(X_test, y_test, batch_size=128, verbose=1)
    print("\n[INFO] accuracy on the test data set: {:.2f}% [{:.5f}]".format(accuracy * 100, loss))


# In[4]:

data_dir = '../../datasets/'

def clean_text(text):
    txt = re.sub('\n\n+', '\07', text)
    txt = re.sub('\n', ' ', txt)
    txt = re.sub('\07', '\n', txt)
    txt = re.sub('  +', ' ', txt)
    txt = re.sub('\nCAPÍTULO [^\n]*\n', '\n', txt)    
    return txt.lower()

book_texts = []
book_titles = []
char_count = 0
for fn in glob.glob(data_dir + 'livros/Machado_de_Assis__*.txt'):
    _, book = os.path.basename(fn).split('__')
    txt = open(fn, encoding='utf-8').read()
    txt = clean_text(txt)
    book_texts.append(txt)
    book_titles.append(book[:-4])
    print('{:7d}  {}'.format(len(txt), book[:-4]))
    char_count += len(txt)
print('{:7d}'.format(char_count))

nb_books = len(book_texts)


# In[5]:

all_text = ''
for txt in book_texts:
    all_text += txt
    
chars = sorted(list(set(all_text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

nb_chars = len(chars)

print('\ntotal chars:', nb_chars)
print(chars)


# In[6]:

# print(book_texts[1][:400])


# In[7]:

seq_len = 40
step = 3
sentences = []
next_chars = []
indexes = []
for k, text in enumerate(book_texts):
    for i in range(0, len(text) - seq_len - 1, step):
        sentences.append(text[i: i + seq_len])
        next_chars.append(text[i + seq_len])
        indexes.append([k, i])

print('nb sequences:', len(sentences))


# In[8]:

print('Vectorization ...')
nb_samples = len(sentences)

X = np.zeros((nb_samples, seq_len, nb_chars), dtype=np.bool)
y = np.zeros((nb_samples, nb_chars), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1
    
print('X.shape:', X.shape)
print('y.shape:', y.shape)


# In[9]:

model_name = '../../models/machado_1'
# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(seq_len, len(chars))))
model.add(BatchNormalization())
model.add(Dense(len(chars)))
model.add(Activation('softmax'))
model.summary()

X_tra, X_val, y_tra, y_val = train_test_split(X, y, test_size=0.5)
print(X_tra.shape, y_tra.shape, X_val.shape, y_val.shape)

fit_params = {
           'opt': Adam(),
    'batch_size': 128, 
       'nepochs': 60,
      'patience': 15,
         'ploss': 0.0,
         'reset': True,
}

N = 300000
Xtra, ytra = X_tra[:N], y_tra[:N]

train_network(model, model_name, Xtra, ytra, **fit_params);

model, histo = load_model_and_history(model_name)

start_index = random.randint(0, len(all_text) - seq_len - 1)
generated = ''
sentence = all_text[start_index: start_index + seq_len]
generated += sentence
print(sentence)
print('-' * len(sentence))

for i in range(400):
    x = np.zeros((1, seq_len, nb_chars))
    for t, char in enumerate(sentence):
        x[0, t, char_indices[char]] = 1.

    preds = model.predict(x, verbose=0)[0]
    index = np.argmax(preds)
    next_char = indices_char[index]
    
    generated += next_char
    sentence = sentence[1:] + next_char

    if i % 100 == 0:
        print(generated)
        print('-' * 60)
