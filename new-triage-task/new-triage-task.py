import sys

# Configuration
MODE = 'cluster' # local' for own machine or 'cluster' for cluster machine
W2V_FILE = '/tmp/PubMed-w2v.bin'
TRAINSET_FILE = 'PMtask_Triage_TrainingSet.xml'
TESTSET_FILE = 'PMtask_Triage_TestSet.xml'
EVALSET_FILE = 'Predictions.json'
SAVE_FIG_FILE = 'results.png'
USE_EVALUATE_SCRIPT = True

# Global Variables
MAX_SEQUENCE_LENGTH = 0
VOCAB_SIZE = 0
EMBEDDING_DIM = 0
VALIDATION_SIZE = 0.1
if MODE == 'local':
    W2V_LIMIT = 10000
    EPOCHS = 1
    BATCHSIZE = 128
    MODEL_ARC = 'dense'
elif MODE == 'cluster':
    W2V_LIMIT = None
    EPOCHS = 30
    BATCHSIZE = 128
    MODEL_ARC = 'lstm-gpu'
    
if MODE != 'local' and MODE != 'cluster':
    sys.exit('Specify a valid running mode: \'local\' or \'cluster\' ')
if MODEL_ARC != 'dense' and MODEL_ARC != 'lstm' and MODEL_ARC != 'lstm-gpu':
    sys.exit('Specify a valid running model: \'dense\' or \'lstm\' or \'lstm-gpu\' ')

# --------------------------------------------------------------------------------------------------------
#                                           Load Data Module
# --------------------------------------------------------------------------------------------------------

import bioc
from bioc import biocjson

def parse_dataset(filename):
    ids = []
    titles = []
    abstracts = []
    labels = []
    
    with bioc.BioCXMLDocumentReader(filename) as reader:
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
    print("Max sequence length: ", maxlen)
    
    return maxlen

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np

def vectorize_text(mode, texts, labels):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    data = pad_sequences(sequences, MAX_SEQUENCE_LENGTH)

    # labels = to_categorical(np.asarray(labels))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)
    
    if(mode == 'training'):
        return data, labels, word_index
    else:
        return data, labels
    
def training_validation_split(texts, labels, texts_shape, labels_test, validation_split):
    indices = np.arange(texts.shape[0])
    np.random.shuffle(indices)
    texts = texts[indices]
    labels = labels[indices]
    nb_validation_samples = int(validation_split * texts.shape[0])

    X_train = texts[:-nb_validation_samples]
    X_val = texts[-nb_validation_samples:]
    
    y_train = labels[:-nb_validation_samples]
    y_val = labels[-nb_validation_samples:]
    
    X_test = np.asarray(texts_shape)
    y_test = labels = to_categorical(np.asarray(labels_test))
    
    print('Training set size: ', X_train.shape[0])
    print('Training targets set size: ', y_train.shape[0])
    print('Validation set size: ', X_val.shape[0])
    print('Validation target set size: ', y_val.shape[0])
    print('Test set size: ', X_test.shape[0])
    print('Test targets set size: ', y_test.shape[0])
    
    return X_train, X_val, y_train, y_val, X_test, y_test

# --------------------------------------------------------------------------------------------------------
#                                        Word Embeddings Module
# --------------------------------------------------------------------------------------------------------

import gensim

def load_pretrained_w2v(file, limit_words = None):
    embedding_space = {}
    embedding_space = gensim.models.KeyedVectors.load_word2vec_format(file,
                                                                      limit = limit_words, 
                                                                      binary = True)
    print('Found %s word vectors.' % len(embedding_space.vocab))
    global VOCAB_SIZE
    VOCAB_SIZE = len(embedding_space.vocab)
    global EMBEDDING_DIM
    EMBEDDING_DIM = len(embedding_space['the'])
    return embedding_space

def compute_embedding_matrix(word_index, embedding_space):
    embedding_matrix = np.zeros((len(embedding_space.vocab) + 1, EMBEDDING_DIM))
    
    for word, i in word_index.items():
        try: 
            embedding_vector = embedding_space[word]
            embedding_matrix[i] = embedding_vector
        except:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = [0] * EMBEDDING_DIM

    print('Shape of embedding matrix: ', embedding_matrix.shape)
    return embedding_matrix

# --------------------------------------------------------------------------------------------------------
#                                               Models Module
# --------------------------------------------------------------------------------------------------------
from keras.models import Model
from keras.layers import Embedding, Input, Flatten, Dense

def get_embedding_layer(embedding_matrix):
    embedding_layer = Embedding(VOCAB_SIZE + 1,
                                EMBEDDING_DIM,
                                weights = [embedding_matrix],
                                input_length = MAX_SEQUENCE_LENGTH,
                                trainable = False)
    
    return embedding_layer

def build_compile_model_Dense(embedding_layer):
    sequence_input = Input(shape = (MAX_SEQUENCE_LENGTH,), dtype = 'int32')
    x = embedding_layer(sequence_input)
    flatten = Flatten()(x)
    # dense = layers.Dense(10, activation = 'relu')(flatten)
    output = Dense(1, activation = 'sigmoid')(flatten)

    model = Model(inputs = sequence_input, outputs = output)

    model.compile(optimizer = 'adam',
                  loss = 'binary_crossentropy',
                  metrics = ['accuracy'])
    return model

from keras.layers import LSTM

def build_compile_model_LSTM(embedding_layer):
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype = 'int32')
    x = embedding_layer(sequence_input)
    lstm = LSTM(64)(x)
    dense = Dense(10, activation = 'relu')(lstm)
    output = Dense(1, activation = 'sigmoid')(dense)

    model = Model(inputs = sequence_input, outputs = output)

    model.compile(optimizer = 'adam',
                  loss = 'binary_crossentropy',
                  metrics = ['accuracy'])
    return model

from keras.layers import Bidirectional, CuDNNLSTM, GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, Dropout

def model_lstm_du(embedding_layer):
    sequence_input = Input(shape = (MAX_SEQUENCE_LENGTH,), dtype = 'int32')
    x = embedding_layer(sequence_input)
    '''
    Here 64 is the size(dim) of the hidden state vector as well as the output vector. Keeping return_sequence we want the output for the entire sequence. So what is the dimension of output for this layer?
        64*70(maxlen)*2(bidirection concat)
    CuDNNLSTM is fast implementation of LSTM layer in Keras which only runs on GPU
    '''
    x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    conc = Dense(64, activation='relu')(conc)
    conc = Dropout(0.1)(conc)
    outp = Dense(1, activation="sigmoid")(conc)
    model = Model(inputs = sequence_input, outputs = outp)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

def fit_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, verbose = 0):
    history = model.fit(X_train, y_train,
                        epochs = epochs,
                        verbose = verbose,
                        validation_data = (X_val, y_val),
                        batch_size = batch_size)
    return history

import matplotlib.pyplot as plt
plt.style.use('ggplot')

def plot_history(history, filename = None):
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
    if(filename != None):
        plt.savefig(filename)
        
import json

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

# --------------------------------------------------------------------------------------------------------
#                                               Main Module
# --------------------------------------------------------------------------------------------------------

import os

if __name__ == "__main__":
    # Load training and testing dataset
    ids, titles, abstracts, labels = parse_dataset('PMtask_Triage_TrainingSet.xml')
    texts = concat_text(titles, abstracts)
    ids_test, titles_test, abstracts_test, labels_test = parse_dataset('PMtask_Triage_TestSet.xml')
    texts_test = concat_text(titles_test, abstracts_test)

    global MAX_SEQUENCE_LENGTH
    MAX_SEQUENCE_LENGTH = get_max_sequence_length(texts, texts_test)

    # Vectorize text
    texts, labels, word_index = vectorize_text('training', texts, labels)
    texts_test, labels_test = vectorize_text('testing', texts_test, labels_test)

    # Training validation split
    X_train, X_val, y_train, y_val, X_test, y_test = training_validation_split(texts, labels, texts_test, labels_test, 0.1)

    # Loading Embedding Space
    embedding_space = load_pretrained_w2v(W2V_FILE, W2V_LIMIT)

    # Compute Embedding Matrix
    embedding_matrix = compute_embedding_matrix(word_index, embedding_space)

    # Keras Model
    # Getting the embedding layer
    embedding_layer = get_embedding_layer(embedding_matrix)

    if MODEL_ARC == 'dense':
        model = build_compile_model_Dense(embedding_layer)
    elif MODEL_ARC == 'lstm':
        model = build_compile_model_LSTM(embedding_layer)
    elif MODEL_ARC == 'lstm-gpu':
        model = model_lstm_du(embedding_layer)


    # Model fitting and accuracy
    history = fit_model(model, X_train, y_train, X_val, y_val, epochs=EPOCHS, batch_size=BATCHSIZE, verbose=1)

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