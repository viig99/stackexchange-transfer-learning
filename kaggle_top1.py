import pandas as pd
from bs4 import BeautifulSoup
# import spacy
import string
import re
import glob
import numpy as np
import os
import numpy as np
np.random.seed(1337)
# from __future__ import print_function
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers import Convolution1D, MaxPooling1D, Embedding, LSTM, GRU, Activation
from keras.layers.advanced_activations import PReLU
from keras.models import Sequential, Model
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import hamming_loss, f1_score
import sys
from sklearn import preprocessing
from sklearn.metrics import label_ranking_average_precision_score, coverage_error, label_ranking_loss
from joblib import Parallel, delayed
import pickle
import csv
nlp = spacy.load('en')
table = string.maketrans("","")

def removePunctuation(s):
    return s.translate(table, string.punctuation)

def htmlToString(s):
    return BeautifulSoup(s, 'html.parser').get_text().encode('utf-8')

def getTranslatedText(s):
    tokens = nlp(s.decode('utf-8'), entity=False)
    tags = []
    list_of_tags = ['CD','FW','JJ','JJR','JJS','NN','NNP','NNPS','NNS','RB','RBR','RBS']
    for token in tokens:
        tags.append((token.lemma_,token.tag_))
    filtered_list = filter(lambda (x,y): y in list_of_tags, tags)
    file_filtered_string = " ".join(map(lambda (x,y): x, filtered_list)).lower()
    return file_filtered_string

def preprocessText(s):
    return getTranslatedText(removePunctuation(htmlToString(s)))

def returnConcatDataFrame(path):
    files = glob.glob(os.path.join(path, "*.csv"))
    dfTemp = (pd.read_csv(f) for f in files)
    df = pd.concat(dfTemp, ignore_index=True)
    return df

def returnSplitToRowDF(df, column, delimiter=' '):
    s = df[column].str.split(delimiter).apply(pd.Series, 1).stack()
    s.index = s.index.droplevel(-1)
    s.name = column
    del df[column]
    return df.join(s)

if __name__ == '__main__':
    # # Load & Save to Disk.
    # training = returnConcatDataFrame('/home/ubuntu/stackexchange/data')
    # training['processed'] = training.apply(lambda x: preprocessText(x['content']), axis=1)
    # training['tagList'] = training.apply(lambda x: x['tags'].split(" "), axis=1)
    # training.to_pickle("/home/ubuntu/stackexchange/pickle/training.pkl")
    # 
    # test = returnConcatDataFrame('/home/ubuntu/stackexchange/test')
    # test['processed'] = test.apply(lambda x: preprocessText(x['content']), axis=1)
    # test.to_pickle("/home/ubuntu/stackexchange/pickle/testing.pkl")

    MAX_NB_WORDS = 100000
    MAX_SEQUENCE_LENGTH = 100
    VALIDATION_SPLIT = 0.10
    EMBEDDING_DIM = 100
    GLOVE_DIR = "/home/ubuntu" # /home/vigi99/FoLink/Kaggle/ /home/ubuntu
    STACKEXCHANGE_DIR = GLOVE_DIR + "/stackexchange"

    dataframe = pd.read_pickle(STACKEXCHANGE_DIR + "/pickle/training_physics.pkl")
    dataframe_test = pd.read_pickle(STACKEXCHANGE_DIR + "/pickle/testing.pkl")

    texts = map(lambda x: str(x.encode('utf8')), dataframe["processed"].tolist())

    texts_test = map(lambda x: str(x.encode('utf8')), dataframe_test["processed"].tolist())
    ids_test = dataframe_test["id"].tolist()

    print("Creating train & test sets.")

    tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)
    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(texts)
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    labeller = preprocessing.MultiLabelBinarizer()
    labels = labeller.fit_transform(dataframe["tagList"])


    nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
    x_train = data[:-nb_validation_samples]
    y_train = labels[:-nb_validation_samples]
    x_val = data[-nb_validation_samples:]
    y_val = labels[-nb_validation_samples:]

    x_test = pad_sequences(tokenizer.texts_to_sequences(texts_test), maxlen=MAX_SEQUENCE_LENGTH)

    print('Indexing word vectors for glove file.')

    embeddings_index = {}
    f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))

    print('Preparing embedding matrix.')

    nb_words = min(MAX_NB_WORDS, len(word_index))
    embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i > MAX_NB_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    embedding_layer = Embedding(nb_words + 1, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False)

    print('Embedding Matrix Created.')

    print('Training model.')

    model = Sequential()
    model.add(embedding_layer)
    model.add(Convolution1D(nb_filter=512,
                            filter_length=7,
                            border_mode='valid',
                            activation='relu',
                            subsample_length=1))
    model.add(MaxPooling1D(pool_length=6))
    model.add(Dropout(0.3))
    model.add(Convolution1D(nb_filter=256,
                            filter_length=6,
                            border_mode='valid',
                            activation='relu',
                            subsample_length=1))
    model.add(MaxPooling1D(pool_length=5))
    model.add(Dropout(0.3))
    model.add(LSTM(100))
    model.add(Dense(300, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(len(labeller.classes_), activation='relu'))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['fmeasure'])
    model.fit(x_train, y_train, validation_data=(x_val, y_val), nb_epoch=30, batch_size=128, verbose=1)
    model.save(os.path.join(GLOVE_DIR, 'model_keras'))

    model = load_model(os.path.join(GLOVE_DIR, 'model_keras'))

    print('Model Trained & Saved.')

    threshold = np.arange(0,0.02,0.00025)
    out = model.predict(x_val)
    out = np.array(out)
    def bestThreshold(y_prob, threshold, i):
        acc = []
        for j in threshold:
            y_pred = np.greater_equal(y_prob, j)*1
            acc.append(matthews_corrcoef(y_val[:,i], y_pred))
        acc = np.array(acc)
        index = np.where(acc==acc.max())
        return threshold[index[0][0]]
    best_threshold = Parallel(n_jobs=4, verbose=1)(delayed(bestThreshold)(out[:,i], threshold, i) for i in range(out.shape[1]))
    y_pred = np.greater_equal(out, np.array(best_threshold)).astype(np.int8)

    hamming_new = hamming_loss(y_pred, y_val)
    hamming_old = pickle.load(open(os.path.join(GLOVE_DIR, 'best_hamming_loss'), "rb"))

    print('Model error\'s, less is better')
    print('New Error: ' + str(hamming_new))
    print('Old Error: ' + str(hamming_old))
    pickle.dump(best_threshold, open(os.path.join(GLOVE_DIR, 'best_threshold'), "wb"))

    best_threshold = pickle.load(open(os.path.join(GLOVE_DIR, 'best_threshold'), "rb"))

    # Save the best_threshold values
    # if hamming_new < hamming_old:
    # pickle.dump(hamming_new, open(os.path.join(GLOVE_DIR, 'best_hamming_loss'), "wb"))
    # model.save(os.path.join(GLOVE_DIR, 'model_keras'))

    del data
    del x_train
    del y_train
    del x_val
    del y_val
    # del labels
    # del labeller
    del dataframe
    del texts
    del out

    print('Prediction Started')
    test_probs = model.predict(x_test, batch_size=256, verbose=1)

    print('Getting vector using best thresholds.')
    y_pred = np.greater_equal(test_probs, np.array(best_threshold))

    print('Get y_tags fast.')
    def getTagFromVector(y):
        return ' '.join(y)
    y_tags_temp = labeller.inverse_transform(y_pred)
    y_tags = Parallel(n_jobs=4, verbose=1)(delayed(getTagFromVector)(y) for y in y_tags_temp)

    print('Saving CSV File.')
    pd.DataFrame.from_records(zip(ids_test, y_tags), columns=["id","tags"]).to_csv(os.path.join(GLOVE_DIR, 'results.csv'), index=False, encoding='utf-8', quoting=csv.QUOTE_ALL)

    print('All Done Bye!')
