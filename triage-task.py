#!/usr/bin/env python
# coding: utf-8

# ## Triage Task
#######################################################
'''
Utilizar o script de teste para validar a rede.
Reduzir o numero máximo de tokens num documento:
    - Retirar palavras mais comuns

'''
#######################################################


# ### Imports

import os, sys

import bioc

import gensim
from gensim.models import KeyedVectors

import numpy as np
import json 
import pandas as pd 
from pandas.io.json import json_normalize

from keras.preprocessing.text import text_to_word_sequence

from keras.models import Model
from keras import layers

import matplotlib.pyplot as plt
plt.style.use('ggplot')

# ### Global variables

# 'local' for own machine or 'cluster' for cluster machine
MODE = 'local'

W2V_FILE = 'PubMed-and-PMC-w2v.bin'
TRAINSET_FILE = 'PMtask_Triage_TrainingSet.xml'
TESTSET_FILE = 'PMtask_Triage_TestSet.xml'
GOLD_STANDARD_FILE = 'PMtask_Triage_TestSet.json'
EVALSET_FILE = 'Predictions.json'
SAVE_FIG_FILE = None
USE_EVALUATE_SCRIPT = True

if MODE == 'local' :
    W2V_LIMIT = 10000
    EPOCHS = 3
    BATCHSIZE = 128
    MODEL_ARC = 'dense'
elif MODE == 'cluster':
    W2V_LIMIT = None
    EPOCHS = 30
    BATCHSIZE = 128
    MODEL_ARC = 'lstm-gpu'

# Manual override
# 'dense' -> Dense
# 'lstm' -> Simple LSTM
# 'lstm-gpu' -> CuDNNLSTM
# MODEL_ARC = 'dense'

if MODE != 'local' and MODE != 'cluster':
    sys.exit('Specify a valid running mode: \'local\' or \'cluster\' ')
if MODEL_ARC != 'dense' and MODEL_ARC != 'lstm' and MODEL_ARC != 'lstm-gpu':
    sys.exit('Specify a valid running model: \'dense\' or \'lstm\' or \'lstm-gpu\' ')

# ### Functions

def load_pretrained_w2v(file, limit_words=None):
    wv_from_bin = gensim.models.KeyedVectors.load_word2vec_format(file,
                                                                  limit=limit_words, 
                                                                  binary = True)
    return wv_from_bin
    
def parse_dataset(filename):
    ids = []
    titles = []
    abstracts = []
    labels = []
    
    with bioc.BioCXMLDocumentReader(filename) as reader:
        collection_info = reader.get_collection_info()
        for document in reader:
            ids.append(document.id)
            relevant = document.infons['relevant']
            labels.append(0 if relevant == 'no' else 1)
            titles.append(document.passages[0].text)
            try:
                abstracts.append(document.passages[1].text)
            except IndexError:
                abstracts.append('')
                
        return ids, titles, abstracts, labels
    
def concat_text(titles, abstracts):
    texts = []
    
    for i in range(0, len(titles)):
        text = titles[i] + abstracts[i]
        texts.append(text)
        
    return texts

def get_max_sequence_length(texts, texts_test):
    max_sequence_training = len(max(texts, key = len))
    max_sequence_testing = len(max(texts_test, key = len))
    maxlen = max_sequence_training if max_sequence_training > max_sequence_testing else max_sequence_testing
    
    return maxlen

# Text to word sequence
def vectorize(row, text, text_sequences):
    for index, word in enumerate(text):
        try:
            text_sequences[row][index] = wv_from_bin.vocab[word].index
        except:
            pass

def word_sequence(texts, shape):
    text_sequences = np.zeros(shape, dtype='int')
    temp = [text_to_word_sequence(x, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower = False, split=' ') for x in texts]
    for index, line in enumerate(temp):
        vectorize(index, line, text_sequences)
    return text_sequences

def data_split(train_sequences, test_sequences, labels, labels_test, validation_size=0, shuffle=False):
    if(shuffle):
        indices = np.arange(len(train_sequences))
        np.random.shuffle(indices)
        X_train = np.array(train_sequences[indices])
        y_train = np.array(labels[indices])
    else:
        # Training data
        X_train = np.array(train_sequences)
        y_train = np.array(labels)
        
        # Test data
        X_test = np.array(test_sequences)
        y_test = np.array(labels_test)
    
    if(validation_size > 0):
        # Training data
        X_validation = X_train[-validation_size:]
        X_train = X_train[:-validation_size]
        
        y_validation = y_train[-validation_size:]
        y_train = y_train[:-validation_size]

        return X_train, X_validation, X_test, y_train, y_validation, y_test
    return X_train, X_test, y_train, y_test

def plot_history(history, file_name=None):
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
    if(file_name != None):
        plt.savefig(file_name)
    
def build_compile_model_Dense(input_shape, w2v_embedding):

    inputs = layers.Input(shape=input_shape)
    embedding = w2v_embedding(inputs)
    flatten = layers.Flatten()(embedding)
    # dense = layers.Dense(10, activation = 'relu')(flatten)
    output = layers.Dense(1, activation = 'sigmoid')(flatten)

    model = Model(inputs=inputs, outputs=output)

    model.compile(optimizer = 'adam',
                  loss = 'binary_crossentropy',
                  metrics = ['accuracy'])
    return model

def build_compile_model_LSTM(input_shape, w2v_embedding):

    inputs = layers.Input(shape=input_shape)
    embedding = w2v_embedding(inputs)
    lstm = layers.LSTM(2)(embedding)
    dense = layers.Dense(10, activation = 'relu')(lstm)
    output = layers.Dense(1, activation = 'sigmoid')(dense)

    model = Model(inputs=inputs, outputs=output)

    model.compile(optimizer = 'adam',
                  loss = 'binary_crossentropy',
                  metrics = ['accuracy'])
    return model
    
def model_lstm_du(input_shape, w2v_embedding):
    inp = layers.Input(shape=input_shape)
    x = w2v_embedding(inp)
    '''
    Here 64 is the size(dim) of the hidden state vector as well as the output vector. Keeping return_sequence we want the output for the entire sequence. So what is the dimension of output for this layer?
        64*70(maxlen)*2(bidirection concat)
    CuDNNLSTM is fast implementation of LSTM layer in Keras which only runs on GPU
    '''
    x = layers.Bidirectional(layers.CuDNNLSTM(4, return_sequences=True))(x)
    avg_pool = layers.GlobalAveragePooling1D()(x)
    max_pool = layers.GlobalMaxPooling1D()(x)
    conc = layers.concatenate([avg_pool, max_pool])
    conc = layers.Dense(4, activation='relu')(conc)
    conc = layers.Dropout(0.1)(conc)
    outp = layers.Dense(1, activation="sigmoid")(conc)
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def fit_model(model, X_train, y_train, X_validation, y_validation, epochs, batch_size, verbose=0):
    history = model.fit(X_train, y_train,
                        epochs = epochs,
                        verbose = verbose,
                        validation_data = (X_validation, y_validation),
                        batch_size = batch_size)
    return history

def save_predictions(model, ids_test, test_sequence):
    data = {}
    documents = []
    # predict_count = 0
    predictions = model.predict(test_sequence)
    for index, id in enumerate(ids_test):
        doc = {}
        infons = {}
        doc['id'] = id
        infons['relevant'] = 'no' if predictions[index] <= 0.5 else 'yes'
        doc['infons'] = infons
        documents.append(doc)

    data['documents'] = documents
    with open(EVALSET_FILE, 'w') as out:
        json.dump(data, out)

def debug_embedding(input_shape, w2v_embedding, test_embedding, X_test):

    inputs = layers.Input(shape=input_shape)
    embedding = w2v_embedding(inputs)

    model = Model(inputs=inputs, outputs=embedding)

    model.compile(optimizer = 'adam',
                  loss = 'binary_crossentropy',
                  metrics = ['accuracy'])
    
    predictions = model.predict(X_test[:1])
    print(predictions)
    print('-----------')
    line = test_embedding[0]
    result = []
    for i in range(5):
        word = line[i]
        word = wv_from_bin.index2entity[word]
        result.append(wv_from_bin.get_vector(word))
    print(result)

# ## Main
if __name__ == "__main__":

    # Load training and testing dataset
    print('-------------------DOC PARSING--------------------')
    ids, titles, abstracts, labels = parse_dataset(TRAINSET_FILE)
    texts = concat_text(titles, abstracts)
    ids_test, titles_test, abstracts_test, labels_test = parse_dataset(TESTSET_FILE)
    texts_test = concat_text(titles_test, abstracts_test)

    MAX_SEQUENCE_LENGTH = get_max_sequence_length(texts, texts_test)
    
    print("Total of training documents: ", len(texts))
    print('Total of testing documents: ',len(texts_test))
    print('Max Sequence Length:', MAX_SEQUENCE_LENGTH)
    print('')

    print('-----------------LOADING EMBEDDING----------------')
    # Load of the pre trained embeddings
    wv_from_bin = load_pretrained_w2v(W2V_FILE, W2V_LIMIT)
    vocab_size = len(wv_from_bin.vocab)
    embedding_dim = len(wv_from_bin['the'])
    print('Vocab size:', vocab_size)
    print('Embedding dimensions:', embedding_dim)
    print('')

    print('-------------------PRE PROCESSING-------------------')
    
    validation_size = int(len(texts)*0.1)
    print("Validation size:",int(validation_size))

    train_word_sequence = word_sequence(texts, (len(texts), MAX_SEQUENCE_LENGTH))
    test_word_sequence = word_sequence(texts_test, (len(texts_test), MAX_SEQUENCE_LENGTH))
    X_train, X_validation, X_test, y_train, y_validation, y_test = data_split(train_word_sequence,
                                                                                test_word_sequence,
                                                                                labels,
                                                                                labels_test,
                                                                                validation_size=validation_size)
    print('Training set size: ', X_train.shape[0])
    print('Training targets set size: ', y_train.shape[0])
    print('Validation set size: ', X_validation.shape[0])
    print('Validation target set size: ', y_validation.shape[0])
    print('Test set size: ', X_test.shape[0])
    print('Test targets set size: ', y_test.shape[0])
    print('')

    print('-------------------TRAINING MODEL-------------------')
    # #### Keras Model
    
    # Getting the embedding layer
    w2v_embedding = wv_from_bin.get_keras_embedding()

    if MODEL_ARC == 'dense':
        model = build_compile_model_Dense((MAX_SEQUENCE_LENGTH,), w2v_embedding)
    elif MODEL_ARC == 'lstm':
        model = build_compile_model_LSTM((MAX_SEQUENCE_LENGTH,), w2v_embedding)
    elif MODEL_ARC == 'lstm-gpu':
        model = model_lstm_du((MAX_SEQUENCE_LENGTH,), w2v_embedding)
    elif MODEL_ARC == 'debug-embedding':
        debug_embedding((MAX_SEQUENCE_LENGTH,), w2v_embedding, test_word_sequence, X_test)
        sys.exit(0)
    
    model.summary()


    # #### Model fitting and accuracy

    history = fit_model(model, X_train, y_train, X_validation, y_validation, epochs=EPOCHS, batch_size=BATCHSIZE, verbose=1)


    print('')

    print('--------------------MODEL EVALUATION-------------------')

    loss, accuracy = model.evaluate(X_train, y_train, verbose = False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    
    loss, accuracy = model.evaluate(X_test, y_test, verbose = False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))
    
    plot_history(history, SAVE_FIG_FILE)
    
    if(USE_EVALUATE_SCRIPT):
        print('---------------------EVALUATION SCRIPT OUTPUT---------------------------')
        save_predictions(model, ids_test, test_word_sequence)
        os.system('python eval_json.py triage ' + GOLD_STANDARD_FILE + ' ' + EVALSET_FILE)

    print('---------------------------END---------------------------------')