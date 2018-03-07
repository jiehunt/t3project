import os
import time
import warnings
import numpy as np
import pandas as pd
import sys, re, csv, codecs
import string
import logging

from scipy.sparse import hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras import backend as K

from keras.layers import Dense, Input, Embedding, Dropout, Activation, Conv1D
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Bidirectional
from keras.layers import BatchNormalization
from keras.layers import GlobalMaxPool1D, MaxPooling1D, Add, Flatten
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D

from keras.engine import InputSpec, Layer
from keras.models import Model
from keras.models import load_model

from keras.callbacks import Callback
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

from keras.optimizers import Adam, RMSprop

""""""""""""""""""""""""""""""
# system setting
""""""""""""""""""""""""""""""
warnings.filterwarnings('ignore')
start_time = time.time()

np.random.seed(42)
os.environ["OMP_NUM_THREADS"] = "4"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


""""""""""""""""""""""""""""""
# Evaluation Callback Function
""""""""""""""""""""""""""""""
class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: {:d} - score: {:.6f}".format(epoch+1, score))


""""""""""""""""""""""""""""""
# Get Data
""""""""""""""""""""""""""""""

""""""""""""""""""""""""""""""
# Feature
""""""""""""""""""""""""""""""
def f_get_coefs(word,*arr):
  return word, np.asarray(arr, dtype='float32')


def f_get_pretraind_features(f_train_text, f_test_text, f_embed_size, f_embedding_path, max_features = 100000, max_len = 150):

    tk = Tokenizer(num_words = max_features, lower = True)
    tk.fit_on_texts(f_train_text)
    f_train = tk.texts_to_sequences(f_train_text)
    f_test  = tk.texts_to_sequences(f_test_text)

    f_train = pad_sequences(f_train, maxlen = max_len)
    f_test  = pad_sequences(f_test, maxlen = max_len)

    f_word_index = tk.word_index
    f_embedding_index = dict(f_get_coefs(*o.strip().split(" ")) for o in open(f_embedding_path))
    nb_words = min(max_features, len(f_word_index))
    f_embedding_matrix = np.zeros((nb_words, f_embed_size))
    for word, i in f_word_index.items():
        if i >= max_features:
          continue
        f_embedding_vector = f_embedding_index.get(word)
        if f_embedding_vector is not None:
            f_embedding_matrix[i] = f_embedding_vector
    return f_train, f_test, f_embedding_matrix


re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')


def f_tokenize(s):
    return re_tok.sub(r' \1 ', s).split()


def f_get_tfidf_features(f_train_text, f_test_text, f_max_features=10000, f_type='word'):
    try:

        print(f_type)
        f_all_text = pd.concat([f_train_text, f_test_text])
        if f_type == 'word':
            print(f_type)
            word_vectorizer = TfidfVectorizer(
                sublinear_tf=True,
                strip_accents='unicode',
                analyzer=f_type,
                token_pattern=r'\w{1,}',
                stop_words='english',
                ngram_range=(1, 1),
                max_features=f_max_features,
            )
        elif f_type == 'char':
            print(f_type)
            word_vectorizer = TfidfVectorizer(
                sublinear_tf=True,
                strip_accents='unicode',
                analyzer=f_type,
                stop_words='english',
                ngram_range=(2, 6),
                max_features=f_max_features,
            )
        elif f_type == 'shortchar':
            print(f_type)
            word_vectorizer = TfidfVectorizer(
                sublinear_tf=True,
                strip_accents='unicode',

                tokenizer=f_tokenize,

                ngram_range=(1, 2),
                min_df=3,
                max_df=0.9,
                use_idf=1,
                smooth_idf=1,
            )
    except:
        print("exception in f_get_tfidf_features")
    else:
        word_vectorizer.fit(f_all_text)
        train_word_features = word_vectorizer.transform(f_train_text)
        test_word_features = word_vectorizer.transform(f_test_text)
        return train_word_features, test_word_features


# f_get_tfidf_features(train['comment_text'], test['comment_text'], f_max_features = 10000, f_type = 'word')
# print("test word over")
# f_get_tfidf_features(train['comment_text'], test['comment_text'], f_max_features = 50000, f_type = 'char')
# print("test char over")
# f_get_tfidf_features(train['comment_text'], test['comment_text'], f_max_features=50000, f_type='shortchar')
# print("test char over")



""""""""""""""""""""""""""""""
# Model
""""""""""""""""""""""""""""""
def m_gru_model(m_max_len, m_max_features, m_embed_size, m_embedding_matrix,
                X_valid, Y_valid, X_train, Y_train, m_file_path,
                m_trainable = False,lr = 0.0, lr_d = 0.0, units = 0, dr = 0.0,
                m_batch_size = 128, m_epochs = 4, m_verbose = 1, ):
    check_point = ModelCheckpoint(m_file_path, monitor = "val_loss", verbose = 1,
                                  save_best_only = True, mode = "min")
    ra_val = RocAucEvaluation(validation_data=(X_valid, Y_valid), interval = 1)
    early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 5)

    inp = Input(shape = (m_max_len,))
    if m_trainable == True:
        x = Embedding(m_max_features, m_embed_size)(inp)
    else:
        x = Embedding(m_max_features, m_embed_size, weights = [m_embedding_matrix], trainable = m_trainable)(inp)
    x = SpatialDropout1D(dr)(x)

    x = Bidirectional(GRU(units, return_sequences = True))(x)
    x = Conv1D(64, kernel_size = 2, padding = "valid", kernel_initializer = "he_uniform")(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    x = concatenate([avg_pool, max_pool])

    x = Dense(6, activation = "sigmoid")(x)
    model = Model(inputs = inp, outputs = x)
    model.compile(loss = "binary_crossentropy",
        optimizer = Adam(lr = lr, decay = lr_d),
        metrics = ["accuracy"])
    history = model.fit(X_train, Y_train, batch_size = m_batch_size, epochs = m_epochs, validation_data = (X_valid, Y_valid),
                        verbose = m_verbose, callbacks = [ra_val, check_point, early_stop])
    model = load_model(m_file_path)
    return model

def m_lstm_model(m_max_len, m_max_features, m_embed_size, m_embedding_matrix,
                X_valid, Y_valid, X_train, Y_train, m_file_path,
                m_trainable = False,lr = 0.0, lr_d = 0.0, units = 0, dr = 0.0,
                m_batch_size = 128, m_epochs = 4, m_verbose = 1, ):
    check_point = ModelCheckpoint(m_file_path, monitor = "val_loss", verbose = 1,
                                  save_best_only = True, mode = "min")
    ra_val = RocAucEvaluation(validation_data=(X_valid, Y_valid), interval = 1)
    early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 5)

    inp = Input(shape = (m_max_len,))
    if m_trainable == True:
        x = Embedding(m_max_features, m_embed_size)(inp)
    else:
        x = Embedding(m_max_features, m_embed_size, weights = [m_embedding_matrix], trainable = m_trainable)(inp)

    # x = SpatialDropout1D(dr)(x)
    # x = Bidirectional(LTSM(units, return_sequences = True))(x)
    # x = Conv1D(64, kernel_size = 2, padding = "valid", kernel_initializer = "he_uniform")(x)
    # avg_pool = GlobalAveragePooling1D()(x)
    # max_pool = GlobalMaxPooling1D()(x)
    # x = concatenate([avg_pool, max_pool])

    x = Bidirectional(LSTM(units, return_sequences=True))(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(dr)(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(dr)(x)

    x = Dense(6, activation = "sigmoid")(x)
    model = Model(inputs = inp, outputs = x)
    model.compile(loss = "binary_crossentropy",
        optimizer = Adam(lr = lr, decay = lr_d),
        metrics = ["accuracy"])
    history = model.fit(X_train, Y_train, batch_size = m_batch_size, epochs = m_epochs, validation_data = (X_valid, Y_valid),
                        verbose = m_verbose, callbacks = [ra_val, check_point, early_stop])
    model = load_model(m_file_path)
    return model



""""""""""""""""""""""""""""""
# Train
""""""""""""""""""""""""""""""
train = pd.read_csv('./input/train.csv').fillna(' ')
test = pd.read_csv('./input/test.csv').fillna(' ')

glove_embedding_path = "./input/glove.840B.300d.txt"
fasttext_embedding_path = './input/crawl-300d-2M.vec'

train["comment_text"].fillna("no comment")
test["comment_text"].fillna("no comment")

def app_gru (train, test):

    train_text = train["comment_text"]
    test_text = test["comment_text"]
    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    y = train[list_classes].values
    max_features = 100000
    max_len = 150
    embed_size = 300

    train,test, embedding_matrix = f_get_pretraind_features(train_text, test_text, embed_size, fasttext_embedding_path,max_features, max_len)

    X_train, X_valid, Y_train, Y_valid = train_test_split(train, y, test_size = 0.1)

    file_path = "./model/best_model_gru_fast.hdf5"
    model = m_gru_model(max_len, max_features, embed_size, embedding_matrix,
                        X_valid, Y_valid, X_train,  Y_train, file_path,
                        m_trainable=False, lr = 1e-3, lr_d = 0, units = 128, dr = 0.2,
                        m_batch_size = 32, m_epochs = 4, m_verbose = 1 )
    pred = model.predict(test, batch_size = 32, verbose = 1)

    submission = pd.read_csv("./input/sample_submission.csv")
    submission[list_classes] = (pred)
    submission.to_csv("./res/submission_fast_gru.csv", index = False)
    print("[{}] Completed!".format(time.time() - start_time))

    train,test, embedding_matrix = f_get_pretraind_features(train_text, test_text, embed_size, glove_embedding_path,max_features, max_len)

    X_train, X_valid, Y_train, Y_valid = train_test_split(train, y, test_size = 0.1)

    file_path = "./model/best_model_gru_glove.hdf5"
    model = m_gru_model(max_len, max_features, embed_size, embedding_matrix,
                        X_valid, Y_valid, X_train,  Y_train, file_path,
                        m_trainable=False, lr = 1e-3, lr_d = 0, units = 128, dr = 0.2,
                        m_batch_size = 32, m_epochs = 4, m_verbose = 1 )
    pred = model.predict(test, batch_size = 32, verbose = 1)

    submission = pd.read_csv("./input/sample_submission.csv")
    submission[list_classes] = (pred)
    submission.to_csv("./res/submission_glove_gru.csv", index = False)
    print("[{}] Completed!".format(time.time() - start_time))

    max_features = 20000
    train, test = f_get_tfidf_features(train_text, test_text, f_max_features=max_features, f_type='word')

    X_train, X_valid, Y_train, Y_valid = train_test_split(train, y, test_size = 0.1)

    file_path = "./model/best_model_gru_word.hdf5"
    model = m_gru_model(max_len, max_features, embed_size, embedding_matrix,
                        X_valid, Y_valid, X_train,  Y_train, file_path,
                        m_trainable=True, lr = 1e-3, lr_d = 0, units = 128, dr = 0.2,
                        m_batch_size = 32, m_epochs = 4, m_verbose = 1 )
    pred = model.predict(test, batch_size = 32, verbose = 1)

    submission = pd.read_csv("./input/sample_submission.csv")
    submission[list_classes] = (pred)
    submission.to_csv("./res/submission_word_gru.csv", index = False)
    print("[{}] Completed!".format(time.time() - start_time))

    train,test = f_get_tfidf_features(train_text, test_text, f_max_features=max_features, f_type='char')

    X_train, X_valid, Y_train, Y_valid = train_test_split(train, y, test_size = 0.1)

    file_path = "./model/best_model_gru_char.hdf5"
    model = m_gru_model(max_len, max_features, embed_size, embedding_matrix,
                        X_valid, Y_valid, X_train,  Y_train, file_path,
                        m_trainable=True, lr = 1e-3, lr_d = 0, units = 128, dr = 0.2,
                        m_batch_size = 32, m_epochs = 4, m_verbose = 1 )
    pred = model.predict(test, batch_size = 32, verbose = 1)

    submission = pd.read_csv("./input/sample_submission.csv")
    submission[list_classes] = (pred)
    submission.to_csv("./res/submission_char_gru.csv", index = False)
    print("[{}] Completed!".format(time.time() - start_time))

    train,test = f_get_tfidf_features(train_text, test_text, f_max_features=max_features, f_type='shortchar')

    X_train, X_valid, Y_train, Y_valid = train_test_split(train, y, test_size = 0.1)

    file_path = "./model/best_model_gru_shortchar.hdf5"
    model = m_gru_model(max_len, max_features, embed_size, embedding_matrix,
                        X_valid, Y_valid, X_train,  Y_train, file_path,
                        m_trainable=True, lr = 1e-3, lr_d = 0, units = 128, dr = 0.2,
                        m_batch_size = 32, m_epochs = 4, m_verbose = 1 )
    pred = model.predict(test, batch_size = 32, verbose = 1)

    submission = pd.read_csv("./input/sample_submission.csv")
    submission[list_classes] = (pred)
    submission.to_csv("./res/submission_shortchar_gru.csv", index = False)
    print("[{}] Completed!".format(time.time() - start_time))

    return pred

def app_lstm (train, test):

    train_text = train["comment_text"]
    test_text = test["comment_text"]
    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    y = train[list_classes].values
    max_features = 20000
    max_len = 150
    embed_size = 300

    train,test, embedding_matrix = f_get_pretraind_features(train_text, test_text, embed_size, fasttext_embedding_path,max_features, max_len)

    X_train, X_valid, Y_train, Y_valid = train_test_split(train, y, test_size = 0.1)

    file_path = "./model/best_model_lstm_fast.hdf5"
    model = m_lstm_model(max_len, max_features, embed_size, embedding_matrix,
                        X_valid, Y_valid, X_train,  Y_train, file_path,
                        m_trainable=False, lr = 1e-3, lr_d = 0, units = 128, dr = 0.1,
                        m_batch_size = 128, m_epochs = 2, m_verbose = 1 )
    pred = model.predict(test, batch_size = 1024, verbose = 1)

    submission = pd.read_csv("./input/sample_submission.csv")
    submission[list_classes] = (pred)
    submission.to_csv("./res/submission_fast_lstm.csv", index = False)
    print("[{}] Completed!".format(time.time() - start_time))

    train,test, embedding_matrix = f_get_pretraind_features(train_text, test_text, embed_size, glove_embedding_path,max_features, max_len)

    X_train, X_valid, Y_train, Y_valid = train_test_split(train, y, test_size = 0.1)

    file_path = "./model/best_model_lstm_glove.hdf5"
    model = m_lstm_model(max_len, max_features, embed_size, embedding_matrix,
                        X_valid, Y_valid, X_train,  Y_train, file_path,
                        m_trainable=False, lr = 1e-3, lr_d = 0, units = 128, dr = 0.1,
                        m_batch_size = 128, m_epochs = 2, m_verbose = 1 )
    pred = model.predict(test, batch_size = 1024, verbose = 1)

    submission = pd.read_csv("./input/sample_submission.csv")
    submission[list_classes] = (pred)
    submission.to_csv("./res/submission_glove_lstm.csv", index = False)
    print("[{}] Completed!".format(time.time() - start_time))


    max_features = 20000
    train, test = f_get_tfidf_features(train_text, test_text, f_max_features=max_features, f_type='word')

    X_train, X_valid, Y_train, Y_valid = train_test_split(train, y, test_size = 0.1)

    file_path = "./model/best_model_lstm_word.hdf5"
    model = m_lstm_model(max_len, max_features, embed_size, embedding_matrix,
                        X_valid, Y_valid, X_train,  Y_train, file_path,
                        m_trainable=True, lr = 1e-3, lr_d = 0, units = 128, dr = 0.2,
                        m_batch_size = 32, m_epochs = 4, m_verbose = 1 )
    pred = model.predict(test, batch_size = 32, verbose = 1)

    submission = pd.read_csv("./input/sample_submission.csv")
    submission[list_classes] = (pred)
    submission.to_csv("./res/submission_word_gru.csv", index = False)
    print("[{}] Completed!".format(time.time() - start_time))

    train,test = f_get_tfidf_features(train_text, test_text, f_max_features=max_features, f_type='char')

    X_train, X_valid, Y_train, Y_valid = train_test_split(train, y, test_size = 0.1)

    file_path = "./model/best_model_lstm_char.hdf5"
    model = m_lstm_model(max_len, max_features, embed_size, embedding_matrix,
                        X_valid, Y_valid, X_train,  Y_train, file_path,
                        m_trainable=True, lr = 1e-3, lr_d = 0, units = 128, dr = 0.2,
                        m_batch_size = 32, m_epochs = 4, m_verbose = 1 )
    pred = model.predict(test, batch_size = 32, verbose = 1)

    submission = pd.read_csv("./input/sample_submission.csv")
    submission[list_classes] = (pred)
    submission.to_csv("./res/submission_char_lstm.csv", index = False)
    print("[{}] Completed!".format(time.time() - start_time))

    train,test = f_get_tfidf_features(train_text, test_text, f_max_features=max_features, f_type='shortchar')

    X_train, X_valid, Y_train, Y_valid = train_test_split(train, y, test_size = 0.1)

    file_path = "./model/best_model_gru_shortchar.hdf5"
    model = m_lstm_model(max_len, max_features, embed_size, embedding_matrix,
                        X_valid, Y_valid, X_train,  Y_train, file_path,
                        m_trainable=True, lr = 1e-3, lr_d = 0, units = 128, dr = 0.2,
                        m_batch_size = 32, m_epochs = 4, m_verbose = 1 )
    pred = model.predict(test, batch_size = 32, verbose = 1)

    submission = pd.read_csv("./input/sample_submission.csv")
    submission[list_classes] = (pred)
    submission.to_csv("./res/submission_shortchar_lstm.csv", index = False)
    print("[{}] Completed!".format(time.time() - start_time))


    return pred



print ("goto app_gru")
pred = app_gru(train, test)



print ("goto app_lstm")
app_lstm(train, test)


""""""""""""""""""""""""""""""
# Stacking
""""""""""""""""""""""""""""""

""""""""""""""""""""""""""""""
# Ganerate Result
""""""""""""""""""""""""""""""
