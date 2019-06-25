#!/usr/bin/env python
# coding: utf-8

# ## Triage Task
#######################################################
'''

'''
#######################################################


# ### Imports

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

import os, sys

import bioc

import gensim
from gensim.models import KeyedVectors

import numpy as np
import json 
import pandas as pd 
from pandas.io.json import json_normalize
import tensorflow

from keras.preprocessing.text import text_to_word_sequence
from keras.models import Model, load_model, model_from_json
from keras import layers, callbacks
from keras import backend as K

from capsule_v1 import Capsule
from keras.utils import CustomObjectScope

from functools import partial
from bayes_opt import BayesianOptimization

import matplotlib.pyplot as plt
plt.style.use('ggplot')

# ### Global variables

# 'local' for own machine or 'cluster' for cluster machine
MODE = 'local'

W2V_FILE = 'PubMed-and-PMC-w2v.bin'
TRAINSET_FILE = 'PMtask_Triage_TrainingSet.xml'
TESTSET_FILE = 'PMtask_Triage_TestSet.xml'
GOLD_STANDARD_FILE = 'PMtask_Triage_TestSet.json'
EVALSET_FILE = 'Predictions.json'   #Output file to save the predictions to
SAVE_FIG_FILE = None                # If not None, the script will save the training curves diagram to the given file
USE_EVALUATE_SCRIPT = True          # If true, the evaluation script will be used
USE_BAYES_OPT = False

if MODE == 'local' :
    # Default settings for local machine
    W2V_LIMIT = 10000
    EPOCHS = 3
    BATCHSIZE = 128
    MODEL_ARC = 'dense'
elif MODE == 'cluster':
    # Default settings for cluster
    W2V_LIMIT = None
    EPOCHS = 30
    BATCHSIZE = 128
    MODEL_ARC = 'capsule'

# Manual override
# 'dense' -> Dense
# 'lstm' -> Simple LSTM
# 'capsule' -> BiGRU with capsules
# MODEL_ARC = 'dense'

if MODE != 'local' and MODE != 'cluster':
    sys.exit('Specify a valid running mode: \'local\' or \'cluster\' ')
if MODEL_ARC != 'dense' and MODEL_ARC != 'lstm' and MODEL_ARC != 'capsule':
    sys.exit('Specify a valid running model: \'dense\' or \'lstm\' or \'capsule\' ')

# ### Functions

'''
Function that loads the pre trained word2vec from the binary file
'''
def load_pretrained_w2v(file, limit_words=None):
    wv_from_bin = gensim.models.KeyedVectors.load_word2vec_format(file,
                                                                  limit=limit_words, 
                                                                  binary = True)
    return wv_from_bin

'''
Function where the dataset files are parsed and load into memory
'''
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

'''
Function that concats the titles and abstracts into one document
''' 
def concat_text(titles, abstracts):
    texts = []
    
    for i in range(0, len(titles)):
        text = titles[i] + abstracts[i]
        texts.append(text)
        
    return texts
'''
Function that calculates the maximum number of words weither on training or testing datasets
''' 
def get_max_sequence_length(texts, texts_test=None):
    if(texts_test != None):
        max_sequence_training = len(max(texts, key = len))
        max_sequence_testing = len(max(texts_test, key = len))
        maxlen = max_sequence_training if max_sequence_training > max_sequence_testing else max_sequence_testing
    else:
        maxlen = len(max(texts, key = len))
    
    return maxlen

'''
Function that converts words into its corresponding index in the w2v vocabulary
''' 
# Text to word sequence
def vectorize(row, text, text_sequences):
    for index, word in enumerate(text):
        try:
            text_sequences[row][index] = wv_from_bin.vocab[word].index
        except:
            pass

'''
Function that first transforms raw text into word sequences and then uses vectorize() to get the words indexes.
Shape should be (number_of_documents, max_number_of_words_per_documents)
The output is a matrix where each line is a document and each column is a token.
''' 
def word_sequence(texts, shape):
    text_sequences = np.zeros(shape, dtype='int')
    temp = [text_to_word_sequence(x, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower = False, split=' ') for x in texts]
    for index, line in enumerate(temp):
        vectorize(index, line, text_sequences)
    return text_sequences

'''
The dataset was already divided into training and test, so data division in that manner was not a concern.
This function is responsible for spliting the training set to have a validation set too, in case it is necessary. It also transform the arrays to numpy arrays.
If shuffle is set to true, the training set is shuffled.
'''
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

'''
Function to plot the training vs validation fitting curves.
file_name should be a string in case the plot should be saved to a file.
'''
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

'''
Costum definition of the F-score metric
'''
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

'''
Function that builds a simple model with a 10 node and a 1 node hidden dense layers.
This was the first model used but its mainly purpose was to run the script in the local machine for debuging.
'''
def build_compile_model_Dense(input_shape, w2v_embedding):

    inputs = layers.Input(shape=input_shape)
    embedding = w2v_embedding(inputs)
    flatten = layers.Flatten()(embedding)
    dense = layers.Dense(10, activation = 'relu')(flatten)
    output = layers.Dense(1, activation = 'sigmoid')(dense)

    model = Model(inputs=inputs, outputs=output)

    model.compile(optimizer = 'adam',
                  loss = 'binary_crossentropy',
                  metrics = ['acc',f1])
    return model

'''
Function to experiment an LSTM implementation
'''
def build_compile_model_LSTM(input_shape, w2v_embedding):

    inputs = layers.Input(shape=input_shape)
    embedding = w2v_embedding(inputs)
    lstm = layers.LSTM(2)(embedding)
    dense = layers.Dense(10, activation = 'relu')(lstm)
    output = layers.Dense(1, activation = 'sigmoid')(dense)

    model = Model(inputs=inputs, outputs=output)

    model.compile(optimizer = 'adam',
                  loss = 'binary_crossentropy',
                  metrics = ['accuracy', f1])
    return model

'''
Function that builds the final approach to the challenge.
The model is a BiGRU with capsule networks instead of CNN.
'''
def build_compile_model_Capsule(input_shape, w2v_embedding, learning_rate, gaussian_noise, embedding_dropout, num_gru_nodes, num_capsules, output_capsules, routing_iterations, capsule_dropout, adam_decay):
    inputs = layers.Input(shape = input_shape)
    x = w2v_embedding(inputs)
    x = layers.GaussianNoise(gaussian_noise)(x)
    x = layers.Dropout(embedding_dropout)(x)
    x = layers.Bidirectional(layers.GRU(num_gru_nodes, return_sequences = True))(x)
    x = Capsule(num_capsules, output_capsules, routings = routing_iterations)(x)
    x = layers.Dropout(capsule_dropout)(x)
    x = layers.Flatten()(x)
    # x = layers.Dense(1, activation = "sigmoid", kernel_regularizer = regularizers.l2(0.01))(x)
    x = layers.Dense(1, activation = "sigmoid")(x)

    model = Model(inputs = inputs, outputs = x)
    # adam = Adam(lr = learning_rate, decay = adam_decay)
    model.compile(loss = "binary_crossentropy", optimizer = 'adam', metrics = ['accuracy', f1])
    return model

'''
Funtion used to fit the builded model.
We implemented EarlyStopping and ModelCheckpoint to reduce training time. 
Both callbacks measure improvements on the F-Score metric.
'''
def fit_model(model, X_train, y_train, X_validation, y_validation, epochs, batch_size, verbose=0):
    es = callbacks.EarlyStopping(monitor='val_f1', mode='max', verbose=1, patience=10)
    mc = callbacks.ModelCheckpoint('best_model.h5', monitor='val_f1', mode='max', save_best_only=True, verbose=1)
    cb_list = [es,mc]

    history = model.fit(X_train, y_train,
                        epochs = epochs,
                        verbose = verbose,
                        validation_data = (X_validation, y_validation),
                        batch_size = batch_size,
                        callbacks=cb_list)
    return history

def fit_with_capsule(X_train, y_train, X_validation, y_validation, input_shape, w2v_embedding, verbose, learning_rate, gaussian_noise, embedding_dropout, num_gru_nodes, num_capsules, output_capsules, routing_iterations, capsule_dropout, batch_size, adam_decay):
    
    num_gru_nodes = 128 # max(int(num_gru_nodes * 64), 64)
    num_capsules = 8 # max(int(num_capsules * 8), 8)
    output_capsules = 16 # max(int(output_capsules * 8), 8)
    routing_iterations = 5 # max(int(routing_iterations * 2), 2)
    batch_size = 128 # max(int(batch_size * 64), 64)
    
    blackbox = build_compile_model_Capsule(input_shape = input_shape,
                                        w2v_embedding = w2v_embedding,
                                        learning_rate = learning_rate,
                                        gaussian_noise = gaussian_noise,
                                        embedding_dropout = embedding_dropout,
                                        num_gru_nodes = num_gru_nodes,
                                        num_capsules = num_capsules,
                                        output_capsules = output_capsules,
                                        routing_iterations = routing_iterations,
                                        capsule_dropout = capsule_dropout,
                                        adam_decay = adam_decay)

    es = callbacks.EarlyStopping(monitor = 'val_f1', mode = 'max', verbose = 1, patience = 10)
    mc = callbacks.ModelCheckpoint('best_model.h5', monitor = 'val_f1', mode = 'max', save_best_only = True, verbose = 1)
    cb_list = [es, mc]

    blackbox = model.fit(x = X_train,
                         y = y_train,
                         verbose = 1,
                         epochs = EPOCHS,
                         batch_size = batch_size,
                         validation_data = (X_validation, y_validation),
                         callbacks = cb_list)

    score = model.evaluate(x = X_validation, y = y_validation,  verbose = 1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    return score[1]

'''
Function used to save the predictions made by the model on the test set.
The predictions are saved in json, where each document has the correspondent 'id' and
the infons stating if the documents is relevant.
'''
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

'''
Function used to debug the usage of the embedding layer.
The idea was to compare embeddings given by the layer with manually converted ones.
'''
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

    if(len(sys.argv) == 2):
        print('-------------------USING BEST MODEL-------------------')
        filename = sys.argv[1]
        
        if(MODEL_ARC == 'capsule'):
            with CustomObjectScope({'Capsule': Capsule, 'f1': f1}):
                model = load_model(filename)
                summary = str(model.summary())
                print(summary)
                config = model.get_config()
                model_json = model.to_json()
                with open("model.json", "w") as json_file:
                        json_file.write(model_json)
                print(config)

    else:
        print('-------------------TRAINING MODEL-------------------')
        # #### Keras Model
        
        # Getting the embedding layer
        w2v_embedding = wv_from_bin.get_keras_embedding()

        if MODEL_ARC == 'dense':
            model = build_compile_model_Dense((MAX_SEQUENCE_LENGTH,), w2v_embedding)
        elif MODEL_ARC == 'lstm':
            model = build_compile_model_LSTM((MAX_SEQUENCE_LENGTH,), w2v_embedding)
        elif MODEL_ARC == 'capsule':
            input_shape = (MAX_SEQUENCE_LENGTH, )
            if(USE_BAYES_OPT):
                pbounds = {'learning_rate': (1e-4, 1e-2),
                           'gaussian_noise': (0.0, 0.5),
                           'embedding_dropout': (0.0, 0.5),
                           'num_gru_nodes': (0.9, 3.1),
                           'num_capsules': (0.9, 3.1),
                           'output_capsules': (0.9, 3.1),
                           'routing_iterations': (0.9, 3.1),
                           'capsule_dropout': (0.0, 0.5),
                           'batch_size': (0.9, 3.1),
                           'adam_decay': (1e-6, 1e-2)}

                verbose = 1
                fit_with_partial = partial(fit_with_capsule, X_train, y_train, X_validation, y_validation, input_shape, w2v_embedding, verbose)

                optimizer = BayesianOptimization(
                    f = fit_with_partial,
                    pbounds = pbounds,
                    verbose = 2,
                    random_state = 1
                )

                optimizer.maximize(init_points = 10, n_iter = 10)

                for i, res in enumerate(optimizer.res):
                    print("Iteration {}: \n\t{}".format(i, res))

                print(optimizer.max)
            
            elif(USE_BAYES_OPT == False):
                 # model = build_compile_model_Capsule(input_shape, w2v_embedding, 1e-3, 0.1, 0.3, 128, 8, 16, 5, 0.3, 0.0)
                 model = build_compile_model_Capsule(input_shape = input_shape,
                                                     w2v_embedding = w2v_embedding, 
                                                     learning_rate = 1e-3,
                                                     gaussian_noise = 0.0,
                                                     embedding_dropout = 0.0,
                                                     num_gru_nodes = 128, 
                                                     num_capsules = 8, 
                                                     output_capsules = 16, 
                                                     routing_iterations = 5, 
                                                     capsule_dropout = 0.0, 
                                                     adam_decay = 0.0)
        elif MODEL_ARC == 'debug-embedding':
            debug_embedding((MAX_SEQUENCE_LENGTH,), w2v_embedding, test_word_sequence, X_test)
            sys.exit(0)
        
        model.summary()


        # #### Model fitting and accuracy

        history = fit_model(model, X_train, y_train, X_validation, y_validation, epochs=EPOCHS, batch_size=BATCHSIZE, verbose=1)


        print('')

        print('--------------------MODEL EVALUATION-------------------')

        loss, accuracy, _ = model.evaluate(X_train, y_train, verbose = False)
        print("Training Accuracy: {:.4f}".format(accuracy))
        
        loss, accuracy, f1 = model.evaluate(X_test, y_test, verbose = False)
        print("Testing Accuracy:  {:.4f}".format(accuracy))
        print("Testing f1 score: {:.4f}".format(f1))
        
        plot_history(history, SAVE_FIG_FILE)
        
    if(USE_EVALUATE_SCRIPT):
        print('---------------------EVALUATION SCRIPT OUTPUT---------------------------')
        save_predictions(model, ids_test, test_word_sequence)
        os.system('python eval_json.py triage ' + GOLD_STANDARD_FILE + ' ' + EVALSET_FILE)

    K.clear_session()
    tensorflow.reset_default_graph()
    del model

    print('---------------------------END---------------------------------')