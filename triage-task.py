#!/usr/bin/env python
# coding: utf-8

# ## Triage Task

# DIR = '/home/a75377/biocreative-vi-track4/'

# ### Import pre-trained Word2Vec vector space

import gensim
from gensim.models import KeyedVectors

# wv_from_bin = gensim.models.KeyedVectors.load_word2vec_format('wikipedia-pubmed-and-PMC-w2v.bin', binary = True)
wv_from_bin = gensim.models.KeyedVectors.load_word2vec_format('PubMed-and-PMC-w2v.bin',limit=100000, binary = True)
print('Pubmed w2v model loaded.')


# ### Import training set file

import json
import numpy as np 
import pandas as pd 
from pandas.io.json import json_normalize

with open('PMtask_Triage_TrainingSet.json') as json_file:
    data = json.load(json_file)
  
documents = json_normalize(data['documents'])

# ### Unpack the passages column into a standalone dataframe

passages = json_normalize(data = data['documents'], record_path = 'passages', record_prefix = 'passage.', meta = 'id')

types = json_normalize(passages['passage.infons'], meta = 'id')


# ### Merge

documents = documents.merge(passages, on = "id", how = "inner")
documents = documents.merge(types, left_index = True, right_index = True)
documents = documents.drop(columns = ['passages', 'relations', 'passage.annotations', 'passage.infons', 'passage.relations', 'passage.sentences'])
documents['infons.relevant'].replace('no', 0, inplace = True)
documents['infons.relevant'].replace('yes', 1, inplace = True)
documents.name = 'training'


# ### Import test set file

with open('PMtask_Triage_TestSet.json') as json_file:
    data_test = json.load(json_file)
    
documents_test = json_normalize(data_test['documents'])
passages_test = json_normalize(data = data_test['documents'], record_path = 'passages', record_prefix = 'passage.', meta = 'id')
types_test = json_normalize(passages_test['passage.infons'], meta = 'id')

documents_test = documents_test.merge(passages_test, on = "id", how = "inner")
documents_test = documents_test.merge(types_test, left_index = True, right_index = True)
documents_test = documents_test.drop(columns = ['passages', 'relations', 'passage.annotations', 'passage.infons', 'passage.relations', 'passage.sentences'])
documents_test['infons.relevant'].replace('no', 0, inplace = True)
documents_test['infons.relevant'].replace('yes', 1, inplace = True)
documents_test.name = 'test'
print(documents_test.head(4))
print(documents_test['passage.text'][0])


# ### Text to word sequence (embedding)


from keras.preprocessing.text import text_to_word_sequence

def vectorize(row, text, embedding_matrix):
    for index, word in enumerate(text):
        try:
            embedding_matrix[row][index] = wv_from_bin.wv.vocab[word].index
        except:
            pass
        
embedding_matrix_train = np.zeros((8162, 3559))
embedding_matrix_test = np.zeros((2854, 3559))

def word_sequence(df):
    df['passage.text'] = df['passage.text'].apply(lambda x: text_to_word_sequence(x, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower = False, split=' '))
    for index, row in df.iterrows():
        if(df.name == 'training'):
            vectorize(index, row['passage.text'], embedding_matrix_train)
        if(df.name == 'test'):
            vectorize(index, row['passage.text'], embedding_matrix_test)


# ### Defining a baseline model

# max(documents.astype('str').applymap(lambda x: len(x)).max())
# max(documents_test.astype('str').applymap(lambda x: len(x)).max())

# Training data
word_sequence(documents)
X_train = embedding_matrix_train
X_train_validation = X_train[8000:]
X_train = X_train[:8000]

y_train = documents['infons.relevant'].values
y_train_validation = y_train[8000:]
y_train = y_train[:8000]

# Test data
word_sequence(documents_test)
X_test = embedding_matrix_test
y_test = documents_test['infons.relevant'].values

# Check matrix
print(X_train.shape)
print(X_train_validation.shape)

# vocab_size = len(np.unique(X_train_batch)) + len(np.unique(X_test_batch))
vocab_size = len(wv_from_bin.vocab)
embedding_dim = len(wv_from_bin['the'])
maxlen = 3559

print('Vocab size: ', vocab_size)
print('Embedding dimensions: ', embedding_dim)
print('Document size ', maxlen)

# # ### Keras embedding layer

from keras.models import Sequential
from keras import layers

 # Getting the embedding layer
w2v_embedding = wv_from_bin.get_keras_embedding()

import random 
random.seed(42)

model = Sequential()
# model.add(layers.Embedding(input_dim = vocab_size, 
#                            output_dim = embedding_dim, 
#                            input_length = maxlen))
model.add(w2v_embedding)
# model.add(layers.Flatten())
model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(10, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))
model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model.summary()

### Model fitting and accuracy

import matplotlib.pyplot as plt
plt.style.use('ggplot')

def plot_history(history):
     acc = history.history['acc']
     val_acc = history.history['val_acc']
     loss = history.history['loss']
     val_loss = history.history['val_loss']
     x = range(1, len(acc) + 1)

     plt.figure(figsize=(12, 5))
     plt.subplot(1, 2, 1)
     plt.plot(x, acc, 'b', label='Training acc')
     plt.plot(x, val_acc, 'r', label='Validation acc')
     plt.title('Training and validation accuracy')
     plt.legend()
     plt.subplot(1, 2, 2)
     plt.plot(x, loss, 'b', label='Training loss')
     plt.plot(x, val_loss, 'r', label='Validation loss')
     plt.title('Training and validation loss')
     plt.legend()
     plt.savefig('result.png')

history = model.fit(X_train, y_train,
                     epochs = 20,
                     verbose = False,
                     validation_data = (X_train_validation, y_train_validation),
                    #  validation_split = 0.1,
                     batch_size = maxlen)

loss, accuracy = model.evaluate(X_train, y_train, verbose = False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose = False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
plot_history(history)

# predict_output_file(model, documents_test):
