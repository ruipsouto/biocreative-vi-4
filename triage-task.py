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
MODE = 'cluster'

W2V_FILE = '~/tmp/PubMed-and-PMC-w2v.bin'
TRAINSET_FILE = 'PMtask_Triage_TrainingSet.json'
TESTSET_FILE = 'PMtask_Triage_TestSet.json'
EVALSET_FILE = 'Predictions.json'
SAVE_FIG_FILE = None
USE_EVALUATE_SCRIPT = True

if MODE == 'local' :
    W2V_LIMIT = 10000
    EPOCHS = 1
    BATCHSIZE = 128
    MODEL_ARC = 'dense'
elif MODE == 'cluster':
    W2V_LIMIT = None
    EPOCHS = 200
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
    
def load_dataset(file):
#     Import file
    with open(file) as json_file:
        data = json.load(json_file)
    documents = json_normalize(data['documents'])
#     Unpack the passages column into a standalone dataframe
    passages = json_normalize(data = data['documents'], record_path = 'passages', record_prefix = 'passage.', meta = 'id')
#     Unpack the passage.infons column into a standalone dataframe
    types = json_normalize(passages['passage.infons'], meta = 'id')
#     Merge
    documents = documents.merge(passages, on = "id", how = "inner")
    documents = documents.merge(types, left_index = True, right_index = True)
    documents = documents.drop(columns = ['passages', 'relations', 'passage.annotations', 'passage.infons', 'passage.relations', 'passage.sentences'])
    documents['infons.relevant'].replace('no', 0, inplace = True)
    documents['infons.relevant'].replace('yes', 1, inplace = True)
    return documents

def concat_text(dataframe):
    d = {
        'id':[],
        'passage.text':[],
        'infons.relevant':[]
    }
    title_count = 0
    abst_count = 0
    last_read_row = None

    for index, row in dataframe.iterrows():
        if row['type'] == 'title':
            if index > 0:
                if(last_read_row['type'] == 'title'):
                    d['id'].append(last_read_row['id'])
                    d['passage.text'].append(last_read_row['passage.text'])
                    d['infons.relevant'].append(last_read_row['infons.relevant'])
            temp = ''
            temp += row['passage.text']
            title_count += 1
            last_read_row = row
        else:
            temp += row['passage.text']
            d['id'].append(row['id'])
            d['passage.text'].append(temp)
            d['infons.relevant'].append(row['infons.relevant'])
            abst_count += 1
            last_read_row = row

    print("Number of titles encountered:",title_count)
    print("Number of abstracts encountered:",abst_count)
    df = pd.DataFrame(data=d)
    return df

# Text to word sequence
def vectorize(row, text, embedding_matrix):
    for index, word in enumerate(text):
        try:
            embedding_matrix[row][index] = wv_from_bin.wv.vocab[word].index
        except:
            pass

def word_sequence(df, shape):
    embedding_matrix = np.ones(shape)
    temp_df = pd.DataFrame()
    temp_df['passage.text'] = df['passage.text'].apply(lambda x: text_to_word_sequence(x, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower = False, split=' '))
    for index, row in temp_df.iterrows():
        vectorize(index, row['passage.text'], embedding_matrix)
    return embedding_matrix

def data_split(train_embedding, test_embedding, corpus, corpus_test, validation_size=0):
    # Training data
    X_train = train_embedding
    y_train = corpus['infons.relevant'].values
    
    # Test data
    X_test = test_embedding
    y_test = corpus_test['infons.relevant'].values
    
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
    lstm = layers.LSTM(64)(embedding)
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
    x = layers.Bidirectional(layers.CuDNNLSTM(64, return_sequences=True))(x)
    avg_pool = layers.GlobalAveragePooling1D()(x)
    max_pool = layers.GlobalMaxPooling1D()(x)
    conc = layers.concatenate([avg_pool, max_pool])
    conc = layers.Dense(64, activation='relu')(conc)
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

def save_predictions(model, corpus_test, test_sequence):
    data = {}
    documents = []
    # predict_count = 0
    predictions = model.predict(test_sequence)
    for index, row in corpus_test.iterrows():
        doc = {}
        infons = {}
        doc['id'] = row['id']
        infons['relevant'] = 'no' if predictions[index] <= 0.5 else 'yes'
        doc['infons'] = infons
        documents.append(doc)

    data['documents'] = documents
    with open(EVALSET_FILE, 'w') as out:
        json.dump(data, out)



# ## Main
if __name__ == "__main__":

    df = load_dataset(TRAINSET_FILE)
    df_test = load_dataset(TESTSET_FILE)
    print("Total of training documents: ", df.shape[0])
    print('Total of testing documents: ',df_test.shape[0])

    corpus = concat_text(df)
    corpus_test = concat_text(df_test)

    print("Total of training documents after concatenation: ", corpus.shape[0])
    print("Total of testing documents after concatenation: ", corpus_test.shape[0])

    print("Max length for training docs:",max(corpus.astype('str').applymap(lambda x: len(x)).max()))
    print("Max length for test docs:",max(corpus_test.astype('str').applymap(lambda x: len(x)).max()))
    maxlen = 3900
    print("Max lenght for all docs after padding: ", maxlen)


    validation_size = int(corpus.shape[0]*0.1)
    print("Validation size:",int(validation_size))
    

    train_word_sequence = word_sequence(corpus, (corpus.shape[0], maxlen))
    test_word_sequence = word_sequence(corpus_test, (corpus_test.shape[0], maxlen))
    X_train, X_validation, X_test, y_train, y_validation, y_test = data_split(train_word_sequence,
                                                                                test_word_sequence,
                                                                                corpus,
                                                                                corpus_test,
                                                                                validation_size=validation_size)
    print('Training set size: ', X_train.shape[0])
    print('Training targets set size: ', y_train.shape[0])
    print('Validation set size: ', X_validation.shape[0])
    print('Validation target set size: ', y_validation.shape[0])
    print('Test set size: ', X_test.shape[0])
    print('Test targets set size: ', y_test.shape[0])

    wv_from_bin = load_pretrained_w2v(W2V_FILE, W2V_LIMIT)
    vocab_size = len(wv_from_bin.vocab)
    embedding_dim = len(wv_from_bin['the'])
    print('Vocab size:', vocab_size)
    print('Embedding dimensions:', embedding_dim)

    # #### Keras Model
    
    # Getting the embedding layer
    w2v_embedding = wv_from_bin.get_keras_embedding()

    if MODEL_ARC == 'dense':
        model = build_compile_model_Dense((maxlen,), w2v_embedding)
    elif MODEL_ARC == 'lstm':
        model = build_compile_model_LSTM((maxlen,), w2v_embedding)
    elif MODEL_ARC == 'lstm-gpu':
        model = model_lstm_du((maxlen,), w2v_embedding)


    # #### Model fitting and accuracy

    history = fit_model(model, X_train, y_train, X_validation, y_validation, epochs=EPOCHS, batch_size=BATCHSIZE, verbose=1)

    loss, accuracy = model.evaluate(X_train, y_train, verbose = False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    
    loss, accuracy = model.evaluate(X_test, y_test, verbose = False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))
    
    plot_history(history, SAVE_FIG_FILE)
    
    if(USE_EVALUATE_SCRIPT):
        print('---------------------EVALUATION SCRIPT OUTPUT---------------------------')
        save_predictions(model, corpus_test, test_word_sequence)
        os.system('python eval_json.py triage ' + TESTSET_FILE + ' ' + EVALSET_FILE)
        print('--------------------------END OF OUTPUT---------------------------------')
