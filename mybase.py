import os
import time
import warnings
import numpy as np
import pandas as pd
import sys, re, csv, codecs
import string
import requests, re, sys
import logging
import psutil
import glob

from scipy.sparse import hstack
from scipy.sparse import csr_matrix

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict
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

from contextlib import contextmanager

from collections import defaultdict

import lightgbm as lgb
import xgboost as xgb
from xgboost.sklearn import XGBClassifier

""""""""""""""""""""""""""""""
# system setting
""""""""""""""""""""""""""""""
warnings.filterwarnings('ignore')
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
# Help Function
""""""""""""""""""""""""""""""
def h_get_train_test_list():
   oof_files= glob.glob("oof/*")
   train_list = []
   test_list = []

   for f in oof_files:
       train_list.append(f)
       # oof_path = str(f).split('\\')[0]
       # oof_file = str(f).split('\\')[1]
       oof_path = str(f).split('/')[0]
       oof_file = str(f).split('/')[1]
       oof_test_pre = str(oof_file).split('oof')[0]
       test_file = str(oof_path) + '_test/'+str(oof_test_pre) + 'test_oof.csv'
       test_list.append(test_file)

   return train_list, test_list

def h_prepare_data_train(file_list):
    class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    class_names_oof = []
    for c in class_names:
        class_names_oof.append(c+'_oof')

    df = pd.read_csv('input/train.csv')
    df = df[class_names]
    for (n, f) in enumerate(file_list):
        one_file = pd.read_csv(f)
        one_file_n = one_file[class_names_oof]
        n_class_name = []
        for c in class_names_oof:
            n_class_name.append(c+str(n))

        one_file_n.columns = n_class_name
        df = pd.concat([df, one_file_n], axis=1)

    return df

def h_prepare_data_test(file_list):
    class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    class_names_oof = []
    for c in class_names:
        class_names_oof.append(c+'_oof')

    df = pd.DataFrame()
    for (n, f) in enumerate(file_list):
        one_file = pd.read_csv(f)
        one_file_n = one_file[class_names]
        n_class_name = []
        for c in class_names_oof:
            n_class_name.append(c+str(n))
        one_file_n.columns = n_class_name
        df = pd.concat([df, one_file_n], axis=1)

    return df

def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])

# Contraction replacement patterns
cont_patterns = [
    (b'(W|w)on\'t', b'will not'),
    (b'(C|c)an\'t', b'can not'),
    (b'(I|i)\'m', b'i am'),
    (b'(A|a)in\'t', b'is not'),
    (b'(\w+)\'ll', b'\g<1> will'),
    (b'(\w+)n\'t', b'\g<1> not'),
    (b'(\w+)\'ve', b'\g<1> have'),
    (b'(\w+)\'s', b'\g<1> is'),
    (b'(\w+)\'re', b'\g<1> are'),
    (b'(\w+)\'d', b'\g<1> would'),
]
patterns = [(re.compile(regex), repl) for (regex, repl) in cont_patterns]

@contextmanager
def timer(name):
    """
    Taken from Konstantin Lopuhin https://www.kaggle.com/lopuhin
    in script named : Mercari Golf: 0.3875 CV in 75 LOC, 1900 s
    https://www.kaggle.com/lopuhin/mercari-golf-0-3875-cv-in-75-loc-1900-s
    """
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


def prepare_for_char_n_gram(text):
    """ Simple text clean up process"""
    # 1. Go to lower case (only good for english)
    # Go to bytes_strings as I had issues removing all \n in r""
    clean = bytes(text.lower(), encoding="utf-8")
    # 2. Drop \n and  \t
    clean = clean.replace(b"\n", b" ")
    clean = clean.replace(b"\t", b" ")
    clean = clean.replace(b"\b", b" ")
    clean = clean.replace(b"\r", b" ")
    # 3. Replace english contractions
    for (pattern, repl) in patterns:
        clean = re.sub(pattern, repl, clean)
    # 4. Drop puntuation
    # I could have used regex package with regex.sub(b"\p{P}", " ")
    exclude = re.compile(b'[%s]' % re.escape(bytes(string.punctuation, encoding='utf-8')))
    clean = b" ".join([exclude.sub(b'', token) for token in clean.split()])
    # 5. Drop numbers - as a scientist I don't think numbers are toxic ;-)
    clean = re.sub(b"\d+", b" ", clean)
    # 6. Remove extra spaces - At the end of previous operations we multiplied space accurences
    clean = re.sub(b'\s+', b' ', clean)
    # Remove ending space if any
    clean = re.sub(b'\s+$', b'', clean)
    # 7. Now replace words by words surrounded by # signs
    # e.g. my name is bond would become #my# #name# #is# #bond#
    # clean = re.sub(b"([a-z]+)", b"#\g<1>#", clean)
    clean = re.sub(b" ", b"# #", clean)  # Replace space
    clean = b"#" + clean + b"#"  # add leading and trailing #

    return str(clean, 'utf-8')

def count_regexp_occ(regexp="", text=None):
    """ Simple way to get the number of occurence of a regex"""
    return len(re.findall(regexp, text))

def get_indicators_and_clean_comments(df):
    """
    Check all sorts of content as it may help find toxic comment
    Though I'm not sure all of them improve scores
    """
    # Count number of \n
    df["ant_slash_n"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\n", x))
    # Get length in words and characters
    df["raw_word_len"] = df["comment_text"].apply(lambda x: len(x.split()))
    df["raw_char_len"] = df["comment_text"].apply(lambda x: len(x))
    # Check number of upper case, if you're angry you may write in upper case
    df["nb_upper"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"[A-Z]", x))
    # Number of F words - f..k contains folk, fork,
    df["nb_fk"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"[Ff]\S{2}[Kk]", x))
    # Number of S word
    df["nb_sk"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"[Ss]\S{2}[Kk]", x))
    # Number of D words
    df["nb_dk"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"[dD]ick", x))
    # Number of occurence of You, insulting someone usually needs someone called : you
    df["nb_you"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\W[Yy]ou\W", x))
    # Just to check you really refered to my mother ;-)
    df["nb_mother"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\Wmother\W", x))
    # Just checking for toxic 19th century vocabulary
    df["nb_ng"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\Wnigger\W", x))
    # Some Sentences start with a <:> so it may help
    df["start_with_columns"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"^\:+", x))
    # Check for time stamp
    df["has_timestamp"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\d{2}|:\d{2}", x))
    # Check for dates 18:44, 8 December 2010
    df["has_date_long"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\D\d{2}:\d{2}, \d{1,2} \w+ \d{4}", x))
    # Check for date short 8 December 2010
    df["has_date_short"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\D\d{1,2} \w+ \d{4}", x))
    # Check for http links
    df["has_http"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"http[s]{0,1}://\S+", x))
    # check for mail
    df["has_mail"] = df["comment_text"].apply(
        lambda x: count_regexp_occ(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', x)
    )
    # Looking for words surrounded by == word == or """" word """"
    df["has_emphasize_equal"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\={2}.+\={2}", x))
    df["has_emphasize_quotes"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\"{4}\S+\"{4}", x))

    # Now clean comments
    df["clean_comment"] = df["comment_text"].apply(lambda x: prepare_for_char_n_gram(x))

    # Get the new length in words and characters
    df["clean_word_len"] = df["clean_comment"].apply(lambda x: len(x.split()))
    df["clean_char_len"] = df["clean_comment"].apply(lambda x: len(x))
    # Number of different characters used in a comment
    # Using the f word only will reduce the number of letters required in the comment
    df["clean_chars"] = df["clean_comment"].apply(lambda x: len(set(x)))
    df["clean_chars_ratio"] = df["clean_comment"].apply(lambda x: len(set(x))) / df["clean_comment"].apply(lambda x: 1 + min(99, len(x)))

def char_analyzer(text):
    """
    This is used to split strings in small lots
    I saw this in an article (I can't find the link anymore)
    so <talk> and <talking> would have <Tal> <alk> in common
    """
    tokens = text.split()
    return [token[i: i + 3] for token in tokens for i in range(len(token) - 2)]


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
    f_embedding_index = dict(f_get_coefs(*o.strip().split(" ")) for o in open(f_embedding_path, encoding="utf-8_sig"))
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

        f_all_text = pd.concat([f_train_text, f_test_text])
        if f_type == 'word':
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
            word_vectorizer = TfidfVectorizer(
                sublinear_tf=True,
                strip_accents='unicode',
                analyzer=f_type,
                stop_words='english',
                ngram_range=(2, 6),
                max_features=f_max_features,
            )
        elif f_type == 'shortchar':
            word_vectorizer = TfidfVectorizer(
                sublinear_tf=True,
                strip_accents='unicode',

                tokenizer=f_tokenize,

                ngram_range=(1, 2),
                min_df=3,
                max_df=0.9,
                use_idf=1,
                smooth_idf=1,
                max_features=f_max_features,
            )
        elif f_type == 'tchar':
            word_vectorizer = TfidfVectorizer(
                sublinear_tf=True,
                strip_accents='unicode',

                tokenizer=char_analyzer,
                ngram_range=(1, 1),
                max_features=f_max_features,
            )
    except:
        print("exception in f_get_tfidf_features")
    else:
        word_vectorizer.fit(f_all_text)
        train_word_features = word_vectorizer.transform(f_train_text)
        test_word_features = word_vectorizer.transform(f_test_text)
        return train_word_features, test_word_features

def f_gen_tfidf_features(train, test):
    import gc
    class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


    with timer("Creating numerical features"):
        num_features = [f_ for f_ in train.columns
                        if f_ not in ["comment_text", "clean_comment", "id", "remaining_chars",
                                      'has_ip_address'] + class_names]
        skl = MinMaxScaler()
        train_num_features = csr_matrix(skl.fit_transform(train[num_features]))
        test_num_features = csr_matrix(skl.fit_transform(test[num_features]))

    # Get TF-IDF features
    train_text = train['clean_comment']
    test_text = test['clean_comment']
    all_text = pd.concat([train_text, test_text])

    with timer("get word features: "):
        train_word_features, test_word_features = f_get_tfidf_features(train_text, test_text,
        f_max_features=10000, f_type='word')

    #with timer("get char features: "):
    #    train_char_features, test_char_features = f_get_tfidf_features(train_text, test_text,
    #    f_max_features=20000, f_type='char')

    #with timer("get shortchar features: "):
    #    train_shortchar_features, test_shortchar_features = f_get_tfidf_features(train_text, test_text,
    #    f_max_features=50000, f_type='shortchar')

    with timer("get tchar features: "):
        train_tchar_features, test_tchar_features = f_get_tfidf_features(train_text, test_text,
        f_max_features=50000, f_type='tchar')

    del train_text
    del test_text
    gc.collect()

    # Now stack TF IDF matrices
    with timer("Staking matrices"):
        csr_trn = hstack(
            [
                # train_char_features,
                train_word_features,
                # train_shortchar_features,
                train_tchar_features,
                train_num_features
            ]
        ).tocsr()
        del train_word_features
        del train_num_features
        # del train_char_features
        del train_tchar_features
        # del train_shortchar_features
        gc.collect()

        csr_sub = hstack(
            [
                # test_char_features,
                test_word_features,
                # test_shortchar_features,
                test_tchar_features,
                test_num_features
            ]
        ).tocsr()
        del test_word_features
        del test_num_features
        # del test_char_features
        del test_tchar_features
        # del test_shortchar_features
        gc.collect()

    return csr_trn, csr_sub


""""""""""""""""""""""""""""""
# Model
""""""""""""""""""""""""""""""
def squash(x, axis=-1):
    # s_squared_norm is really small
    # s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    # scale = K.sqrt(s_squared_norm)/ (0.5 + s_squared_norm)
    # return scale * x
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)
    scale = K.sqrt(s_squared_norm + K.epsilon())
    return x / scale


# A Capsule Implement with Pure Keras
class Capsule(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, kernel_size=(9, 1), share_weights=True,
                 activation='default', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size
        self.share_weights = share_weights
        if activation == 'default':
            self.activation = squash
        else:
            self.activation = Activation(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     # shape=self.kernel_size,
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
        # final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]

        b = K.zeros_like(u_hat_vecs[:, :, :, 0])  # shape = [None, num_capsule, input_num_capsule]
        for i in range(self.routings):
            b = K.permute_dimensions(b, (0, 2, 1))  # shape = [None, input_num_capsule, num_capsule]
            c = K.softmax(b)
            c = K.permute_dimensions(c, (0, 2, 1))
            b = K.permute_dimensions(b, (0, 2, 1))
            outputs = self.activation(K.batch_dot(c, u_hat_vecs, [2, 2]))
            if i < self.routings - 1:
                b = K.batch_dot(outputs, u_hat_vecs, [2, 3])

        return outputs

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)

def m_capsule_gru_model(max_len, max_features, embed_size, embedding_matrix,
                X_valid, Y_valid, X_train, Y_train, m_file_path,
                m_trainable = False,lr = 0.0, lr_d = 0.0, units = 0, dr = 0.0,
                m_batch_size = 128, m_epochs = 4, m_verbose = 1,
                m_num_capsule = 10, m_dim_capsule = 16,m_routings = 5,
                ):

    input1 = Input(shape=(max_len,))
    embed_layer = Embedding(max_features,
                            embed_size,
                            input_length=max_len,
                            weights=[embedding_matrix],
                            trainable=m_trainable)(input1)
    embed_layer = SpatialDropout1D(dr)(embed_layer)

    x = Bidirectional(
        GRU(units, activation='relu', dropout=dr, recurrent_dropout=dr, return_sequences=True))(
        embed_layer)
    capsule = Capsule(num_capsule=m_num_capsule, dim_capsule=m_dim_capsule, routings=m_routings,
                      share_weights=True)(x)
    # output_capsule = Lambda(lambda x: K.sqrt(K.sum(K.square(x), 2)))(capsule)
    capsule = Flatten()(capsule)
    capsule = Dropout(dr)(capsule)
    output = Dense(6, activation='sigmoid')(capsule)
    model = Model(inputs=input1, outputs=output)
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    model.summary()

    check_point = ModelCheckpoint(m_file_path, monitor = "val_loss", verbose = 1,
                                  save_best_only = True, mode = "min")
    ra_val = RocAucEvaluation(validation_data=(X_valid, Y_valid), interval = 1)
    early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 5)

    history = model.fit(X_train, Y_train, batch_size = m_batch_size, epochs = m_epochs, validation_data = (X_valid, Y_valid),
                        verbose = m_verbose, callbacks = [ra_val, check_point, early_stop])

    model = load_model(m_file_path)
    return model


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

def m_lgb_model(csr_trn, csr_sub, train, test, feature_type):

    class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    # Set LGBM parameters
    params = {
        "objective": "binary",
        'metric': {'auc'},
        "boosting_type": "gbdt",
        "verbosity": -1,
        "num_threads": 4,
        "bagging_fraction": 0.8,
        "feature_fraction": 0.8,
        "learning_rate": 0.1,
        "num_leaves": 31,
        "verbose": -1,
        "min_split_gain": .1,
        "reg_alpha": .1,
        "device": "gpu",
        "gpu_platform_id": 0,
        "gpu_device_id": 0,
        "max_bin": 63
    }
    # print (type(train)) frame.DataFrame

    # Now go through folds
    # I use K-Fold for reasons described here :
    # https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/49964
    with timer("Scoring Light GBM"):
        scores = []
        folds = StratifiedKFold(n_splits=4, shuffle=True, random_state=1)
        lgb_round_dict = defaultdict(int)
        trn_lgbset = lgb.Dataset(csr_trn, free_raw_data=False)
        # del csr_trn
        # gc.collect()

        for class_name in class_names:
            print("Class %s scores : " % class_name)
            class_pred = np.zeros(len(train))
            train_target = train[class_name]
            trn_lgbset.set_label(train_target.values)

            lgb_rounds = 500

            for n_fold, (trn_idx, val_idx) in enumerate(folds.split(train, train_target)):
                watchlist = [
                    trn_lgbset.subset(trn_idx),
                    trn_lgbset.subset(val_idx)
                ]
                # Train lgb l1
                model = lgb.train(
                    params=params,
                    train_set=watchlist[0],
                    num_boost_round=lgb_rounds,
                    valid_sets=watchlist,
                    early_stopping_rounds=50,
                    verbose_eval=0
                )
                class_pred[val_idx] = model.predict(trn_lgbset.data[val_idx], num_iteration=model.best_iteration)
                score = roc_auc_score(train_target.values[val_idx], class_pred[val_idx])

                # Compute mean rounds over folds for each class
                # So that it can be re-used for test predictions
                lgb_round_dict[class_name] += model.best_iteration
                print("\t Fold %d : %.6f in %3d rounds" % (n_fold + 1, score, model.best_iteration))

            print("full score : %.6f" % roc_auc_score(train_target, class_pred))
            scores.append(roc_auc_score(train_target, class_pred))
            train[class_name + "_oof"] = class_pred

        # Save OOF predictions - may be interesting for stacking...
        file_name = 'oof/'+ 'lgb_'+str(feature_type) + '_oof.csv'
        train[["id"] + class_names + [f + "_oof" for f in class_names]].to_csv(file_name,
                                                                               index=False,
                                                                               float_format="%.8f")

        print('Total CV score is {}'.format(np.mean(scores)))

        pred = np.zeros( shape=(len(test), len(class_names)) )
        pred =pd.DataFrame(pred)
        pred.columns = class_names
        with timer("Predicting probabilities"):
            # Go through all classes and reuse computed number of rounds for each class
            for class_name in class_names:
                with timer("Predicting probabilities for %s" % class_name):
                    train_target = train[class_name]
                    trn_lgbset.set_label(train_target.values)
                    # Train lgb
                    model = lgb.train(
                        params=params,
                        train_set=trn_lgbset,
                        num_boost_round=int(lgb_round_dict[class_name] / folds.n_splits)
                    )
                    pred[class_name] = model.predict(csr_sub, num_iteration=model.best_iteration)

        return pred

def m_pr(x, y_i, y):
    p = x[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)


def m_nbsvm_model(x, y):
    y = y.values
    r = np.log(m_pr(x, 1,y) / m_pr(x, 0,y))
    m = LogisticRegression(C=5, dual=True)
    x_nb = x.multiply(r)
    return m.fit(x_nb, y), r

class my_nbsvm:

    def __init__(self, **params):
        self.model = LogisticRegression(C=4, dual=True)
        return

    def m_pr(x, y_i, y):
        p = x[y==y_i].sum(0)
        return (p+1) / ((y==y_i).sum()+1)

    def fit(self, X, y):
        y = y.values
        self.r = np.log(m_pr(X, 1,y) / m_pr(X, 0,y))
        x_nb = np.multiply(X, self.r)
        # x_nb = X.multiply(self.r)
        return self.model.fit(x_nb, y)

    def predict(self, X):
        return self.model.predict(np.multiply(X, self.r) )
        # return self.model.predict(X.multiply(self.r))

    def predict_proba(self,X):
        return self.model.predict_proba(np.multiply(X, self.r))
        # return self.model.predict_proba(X.multiply(self.r))

    def get_params(self, deep):
        return self.model.get_params(deep=deep)



""""""""""""""""""""""""""""""
# Stacking
""""""""""""""""""""""""""""""
def app_stack():
    class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    class_names_oof = []
    for c in class_names:
        class_names_oof.append(c+'_oof')

    sub = pd.read_csv('./input/sample_submission.csv')

    train_list, test_list =  h_get_train_test_list()
    num_file = len(train_list)

    train = h_prepare_data_train(train_list)
    test = h_prepare_data_test(test_list)

    # stacker = LogisticRegression()
    # stacker = xgb.XGBClassifier()
    stacker = lgb.LGBMClassifier(max_depth=3, metric="auc", n_estimators=125, num_leaves=10, boosting_type="gbdt",
                                 learning_rate=0.1,  colsample_bytree=0.45,reg_lambda=0.2,)
                                 # feature_fraction=0.45,bagging_fraction=0.8, bagging_freq=5,  verbose=-1)

    train_r = train.drop(class_names,axis=1)
    train_target = train[class_names]
    # X_train, X_valid, Y_train, Y_valid = train_test_split(train_r, train_target, test_size = 0.1, random_state=1982)
    # for class_name in class_names:
    #     y_target = Y_train[class_name]
    #     stacker.fit(X_train, y=y_target)
    #     trn_pred = stacker.predict_proba(X_valid)[:,1]
    #     print ("%s score : %f" % (str(class_name),  roc_auc_score(Y_valid[class_name], trn_pred)))

    #     sub[class_name] = stacker.predict_proba(test)[:,1]

    # Fit and submit
    X_train = train_r
    Y_train = train_target
    scores = []
    for label in class_names:
        print(label)
        score = cross_val_score(stacker, X_train, Y_train[label], cv=5, scoring='roc_auc',verbose=0)
        print("AUC:", score)
        scores.append(np.mean(score))
        stacker.fit(X_train, Y_train[label])
        sub[label] = stacker.predict_proba(test)[:,1]
        # trn_pred = stacker.predict_proba(X_valid)[:,1]
        # print ("%s score : %f" % (str(label),  roc_auc_score(Y_valid[label], trn_pred)))
    print("CV score:", np.mean(scores))

    out_file = 'output/submission_' + str(num_file) +'file.csv'
    sub.to_csv(out_file,index=False)
    return

""""""""""""""""""""""""""""""
# Train
""""""""""""""""""""""""""""""
def m_make_single_submission(m_infile, m_outfile, m_pred):
    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    submission = pd.read_csv(m_infile)
    submission[list_classes] = (m_pred)
    submission.to_csv(m_outfile, index = False)

def app_train_nbsvm(csr_trn, csr_sub,train, test, feature_type):
    class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    n_splits = 4
    model_type = 'nbsvm'
    with timer("Scoring nbsvm"):
        scores = []
        folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)
        nbsvm_round_dict = defaultdict(int)

        for class_name in class_names:
            print("Class %s scores : " % class_name)
            class_pred = np.zeros(len(train))
            train_target = train[class_name]

            model = my_nbsvm()
            for n_fold, (trn_idx, val_idx) in enumerate(folds.split(train, train_target)):
                file_path = './model/'+str(model_type) +'_'+str(feature_type) + str(class_name) + str(n_fold) + '.model'

                # if os.path.exists(file_path):
                #     model.load_model(file_path)
                #     # xgb.Booster.load_model(model,file_path)
                # else:
                with timer("One nbsvm traing"):
                    model.fit(csr_trn[trn_idx], train_target[trn_idx])
                    # model, r = m_nbsvm_model(csr_trn[trn_idx], train_target[trn_idx])


                class_pred[val_idx] = model.predict_proba(csr_trn[val_idx])[:,1]
                # class_pred[val_idx] = model.predict_proba(csr_trn[val_idx].multiply(r))[:,1]
                score = roc_auc_score(train_target[val_idx], class_pred[val_idx])

                # Compute mean rounds over folds for each class
                # So that it can be re-used for test predictions
                print("\t Fold %d : %.6f " % (n_fold + 1, score))

            print("full score : %.6f" % roc_auc_score(train_target, class_pred))
            scores.append(roc_auc_score(train_target, class_pred))
            train[class_name + "_oof"] = class_pred

        # Save OOF predictions - may be interesting for stacking...
        oof_file = './oof/'+str(model_type) +'_'+str(feature_type) + '_oof.csv'
        train[["id"] + class_names + [f + "_oof" for f in class_names]].to_csv(oof_file,
                                                                               index=False,
                                                                               float_format="%.8f")

        print('Total CV score is {}'.format(np.mean(scores)))

        # Use train for test
        pred = np.zeros( shape=(len(test), len(class_names)) )
        pred =pd.DataFrame(pred)
        pred.columns = class_names
#
        with timer("Predicting test probabilities"):
            # Go through all classes and reuse computed number of rounds for each class
            for class_name in class_names:
                with timer("Predicting probabilities for %s" % class_name):
                    train_target = train[class_name]
                    # Train xgb
                    # model, r = m_nbsvm_model(csr_trn, train_target)
                    model = my_nbsvm()
                    model.fit(csr_trn, train_target)
                    file_path = './model/'+str(model_type) +'_'+str(feature_type) + str(class_name) + 'full.model'
                    # model.dump_modle(file_path)
                    # pred[class_name] = model.predict_proba(csr_sub.multiply(r))[:,1]
                    pred[class_name] = model.predict_proba(csr_sub)[:,1]

        return pred

def app_train_rnn(train, test, embedding_path, model_type, feature_type):

    class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    train_target = train[class_names]
    test_text = test["comment_text"]
    train_text = train["comment_text"]

    splits = 3

    max_len = 150
    max_features = 100000
    embed_size = 300

    m_batch_size = 32
    m_epochs = 3
    m_verbose = 1
    lr = 1e-3
    lr_d = 0
    units = 128
    dr = 0.2

    class_pred = np.ndarray(shape=(len(train), len(class_names)))

    with timer("get pretrain features for rnn"):
        train_r,test, embedding_matrix = f_get_pretraind_features(train_text, test_text, embed_size, embedding_path,max_features, max_len)

    with timer("Goto Train RNN Model"):
        folds = KFold(n_splits=splits, shuffle=True, random_state=1)

        for n_fold, (trn_idx, val_idx) in enumerate(folds.split(train_r, train_target)):

            print ("goto %d fold :" % n_fold)
            X_train_n = train_r[trn_idx]
            Y_train_n = train_target.iloc[trn_idx]
            X_valid_n = train_r[val_idx]
            Y_valid_n = train_target.iloc[val_idx]

            if model_type == 'gru': # gru
                file_path = './model/'+str(model_type) +'_'+str(feature_type) + str(n_fold) + '.hdf5'
                if os.path.exists(file_path):
                    model = load_model(file_path)
                else:
                    model = m_gru_model(max_len, max_features, embed_size, embedding_matrix,
                                    X_valid_n, Y_valid_n, X_train_n,  Y_train_n, file_path,
                                    m_trainable=False, lr=lr, lr_d = lr_d, units = units, dr = dr,
                                    m_batch_size= m_batch_size, m_epochs = m_epochs, m_verbose = m_verbose)
            elif model_type == 'lstm': # lstm
                file_path = './model/'+str(model_type) +'_'+str(feature_type) + str(n_fold) + '.hdf5'
                if os.path.exists(file_path):
                    model = load_model(file_path)
                else:
                    model = m_lstm_model(max_len, max_features, embed_size, embedding_matrix,
                                X_valid_n, Y_valid_n, X_train_n,  Y_train_n, file_path,
                                m_trainable=False, lr = lr, lr_d = lr_d, units = units, dr = dr,
                                m_batch_size= m_batch_size, m_epochs = m_epochs, m_verbose = m_verbose)
            elif model_type == 'capgru':
                file_path = './model/'+str(model_type) +'_'+str(feature_type) + str(n_fold) + '.hdf5'
                if os.path.exists(file_path):
                    model = load_model(file_path)
                else:
                    model = m_capsule_gru_model(max_len, max_features, embed_size, embedding_matrix,
                                X_valid_n, Y_valid_n, X_train_n,  Y_train_n, file_path,
                                m_trainable=False, lr = lr, lr_d = lr_d, units = units, dr = dr,
                                m_batch_size= m_batch_size, m_epochs = m_epochs, m_verbose = m_verbose)


            class_pred[val_idx] =pd.DataFrame(model.predict(X_valid_n))

        oof_names = ['toxic_oof', 'severe_toxic_oof', 'obscene_oof', 'threat_oof', 'insult_oof', 'identity_hate_oof']
        class_pred = pd.DataFrame(class_pred)
        class_pred.columns = oof_names
        train_oof = pd.concat([train,class_pred], axis = 1)
        for class_name in class_names:
            print("Class %s scores : " % class_name)
            print("%.6f" % roc_auc_score(train_target[class_name], class_pred[class_name+"_oof"]))

        # Save OOF predictions - may be interesting for stacking...
        file_name = 'oof/'+str(model_type) + '_' + str(feature_type) + '_oof.csv'
        train_oof[["id"] + class_names + [f + "_oof" for f in class_names]].to_csv(file_name,
                                                                               index=False,
                                                                               float_format="%.8f")

        X_train, X_valid, Y_train, Y_valid = train_test_split(train_r, train_target, test_size = 0.1)

        # Use train for test
        if model_type == 'gru': # gru
            file_path = './model/'+str(model_type) + '_'+ str(feature_type) + 'full' + '.hdf5'
            if os.path.exists(file_path):
                model = load_model(file_path)
            else:
                model = m_gru_model(max_len, max_features, embed_size, embedding_matrix,
                            X_valid, Y_valid, X_train,  Y_train, file_path,
                            m_trainable=False, lr=lr, lr_d = lr_d, units = units, dr = dr,
                            m_batch_size= m_batch_size, m_epochs = m_epochs, m_verbose = m_verbose)
        elif model_type == 'lstm': # lstm
            file_path = './model/'+str(model_type) + '_'+ str(feature_type) + 'full' + '.hdf5'
            if os.path.exists(file_path):
                model = load_model(file_path)
            else:
                model = m_lstm_model(max_len, max_features, embed_size, embedding_matrix,
                            X_valid, Y_valid, X_train,  Y_train, file_path,
                            m_trainable=False, lr = lr, lr_d = lr_d, units = units, dr = dr,
                            m_batch_size= m_batch_size, m_epochs = m_epochs, m_verbose = m_verbose)
        elif model_type == 'capgru': # lstm
            file_path = './model/'+str(model_type) + '_'+ str(feature_type) + 'full' + '.hdf5'
            if os.path.exists(file_path):
                model = load_model(file_path)
            else:
                model = m_capsule_gru_model(max_len, max_features, embed_size, embedding_matrix,
                            X_valid, Y_valid, X_train,  Y_train, file_path,
                            m_trainable=False, lr = lr, lr_d = lr_d, units = units, dr = dr,
                            m_batch_size= m_batch_size, m_epochs = m_epochs, m_verbose = m_verbose)


        pred =pd.DataFrame(model.predict(test))
        pred.columns = class_names

    return pred

def h_tuning_xgb(train, train_target,tune_dict, param_test):

    param_dict = {
       'learning_rate' : 0.1,
       'n_estimators'  : 1000,
       'max_depth' : 4,
       'min_child_weight':8,
       'gamma':0,
       'subsample':0.8,
       'colsample_bytree':0.8,
       'reg_alpha':0,
    }
    param_dict = tune_dict

    gsearch = GridSearchCV(estimator=XGBClassifier(learning_rate=param_dict['learning_rate'],
                                                    n_estimators=param_dict['n_estimators'],
                                                    max_depth=param_dict['max_depth'],
                                                    min_child_weight=param_dict['min_child_weight'],
                                                    gamma=param_dict['gamma'],
                                                    subsample=param_dict['subsample'],
                                                    colsample_bytree=param_dict['colsample_bytree'],
                                                    reg_alpha=param_dict['reg_alpha'],
                                                    gpu_id=0,
                                                    max_bin = 16,
                                                    tree_method = 'gpu_hist',
                                                    objective='binary:logistic',
                                                    nthread=4, scale_pos_weight=1,
                                                    seed=27),
                                                    param_grid=param_test, scoring='roc_auc', n_jobs=4, iid=False, cv=5, verbose=2)

    with timer("goto serch max_depth and min_child_wight"):
        gsearch.fit(train, train_target)
        print (gsearch.grid_scores_ )
        print (gsearch.best_params_ )
        print (gsearch.best_score_)
        return gsearch.best_params_


def app_single_lgb(csr_trn, csr_sub, train, test, feature_type):

    class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    train_target = train[class_names]
    train_r = csr_trn
    params = {
        "objective": "binary",
        "metric": {'auc'},
        "boosting_type": "gbdt",
        "verbosity": -1,
        "num_threads": 4,
        "bagging_fraction": 0.8,
        "feature_fraction": 0.8,
        "learning_rate": 0.1,
        "num_leaves": 31,
        "verbose": -1,
        "min_split_gain": .1,
        "reg_alpha": .1,
        "device": "gpu",
        "gpu_platform_id": 0,
        "gpu_device_id": 0,
        "max_bin": 63
    }
    param_test1 = {
        'num_leaves':[ i for i in range(28, 32)],
    }
    param_test = param_test1

    gsearch = GridSearchCV(estimator=lgb.LGBMClassifier(boosting_type="gbdt", objective=params["objective"], metric=params["metric"],
                            verbosity=params["verbosity"],
                            num_threads = params["num_threads"],
                            bagging_fraction= params["bagging_fraction"],
                            feature_fraction = params["feature_fraction"],
                            learning_rate =params["learning_rate"],
                            num_leaves = params["num_leaves"],
                            verbose = params["verbose"],
                            min_split_gain = params["min_split_gain"],
                            reg_alpha=params["reg_alpha"],
                            device = params["device"],
                            gpu_platform_id=params["gpu_platform_id"],
                            gpu_device_id = params["gpu_device_id"],
                            max_bin=params["max_bin"]
                            ) ,
                            param_grid=param_test, scoring='roc_auc', n_jobs=1, iid=False, cv=3, verbose=2)


    with timer("goto tuning lgb_wight"):
        gsearch.fit(train_r, train_target['toxic'])
        print (gsearch.grid_scores_ )
        print (gsearch.best_params_ )
        print (gsearch.best_score_)

    return


def app_single_xgb(csr_trn, csr_sub, train, test, feature_type):

    class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    train_target = train[class_names]
    train_r = csr_trn

    # X_train, X_valid, Y_train, Y_valid = train_test_split(train_r, train_target, test_size = 0.1)

    param_test1 = {
        'max_depth': [4,7, 10],
        'min_child_weight': [4,7, 10]
    }
    param_test2 = {
        'gamma': [i / 10.0 for i in range(0, 5)]
    }
    param_test3 = {
        'subsample': [i / 10.0 for i in range(6, 10)],
        'colsample_bytree': [i / 10.0 for i in range(6, 10)]
    }
    param_test4 = {
        'subsample': [i / 100.0 for i in range(55, 75, 5)],
        'colsample_bytree': [i / 100.0 for i in range(55, 75, 5)]
    }
    param_test5 = {
        'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100]
    }
    param_test6 = {
        'learning_rate': [0.001, 0.01, 0.05, 0.1]
    }
    param_test7 = {
        'n_estimators': [1000,2000,3000,4000,5000]
    }

    param_set = [
        param_test1,
        param_test2,
        param_test3,
        param_test4,
        param_test5,
        param_test6,
        param_test7
    ]

    param_dict = {
        'learning_rate' : 0.1,
        'n_estimators'  : 1000,
        'max_depth' : 4,
        'min_child_weight':8,
        'gamma':0,
        'subsample':0.9,
        'colsample_bytree':0.6,
        'reg_alpha':0,
    }
    for param in param_set:
        with timer("goto serch max_depth and min_child_wight"):
            best_param = h_tuning_xgb(train_r, train_target['toxic'],param_dict, param)

        for key in param_dict:
            for key2 in best_param:
                if key == key2:
                    param_dict[key] = best_param[key2]
                    print ("change %s to %d" % key, best_param[key2])
    return param_dict

def app_train_xgb(csr_trn, csr_sub, train, test, feature_type):
    import gc
    class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    # Set LGBM parameters
    params = {}
    params['objective'] = 'binary:logistic'
    params['learning_rate'] = 0.01
    params['n_estimators'] = 1000
    params['max_depth'] = 4
    params['subsample'] = 0.9
    params['colsample_bytree'] = 0.9
    params['min_child_weight'] = 10
    params['gamma'] = 0
    params['nthread'] = 4
    params['gpu_id'] = 0
    params['max_bin'] = 16
    params['tree_method'] = 'gpu_hist'
    params['scale_pos_weight'] = 1
    params['seed'] = 27

    # Now go through folds
    # I use K-Fold for reasons described here :
    # https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/49964
    with timer("Scoring xgboost"):
        scores = []
        folds = StratifiedKFold(n_splits=4, shuffle=True, random_state=1)
        xgb_round_dict = defaultdict(int)
        # trn_xgbset = xgb.DMatrix(csr_trn)
        # del csr_trn
        # gc.collect()

        for class_name in class_names:
            print("Class %s scores : " % class_name)
            class_pred = np.zeros(len(train))
            train_target = train[class_name]
            # trn_xgbset.set_label(train_target.values)

            xgb_rounds = 500

            for n_fold, (trn_idx, val_idx) in enumerate(folds.split(train, train_target)):
                trn_subset = xgb.DMatrix(csr_trn[trn_idx], train_target[trn_idx])
                val_subset = xgb.DMatrix(csr_trn[val_idx], train_target[val_idx])
                watchlist = [
                    (trn_subset, 'train'),
                    (val_subset, 'eval') ]
                # Train xgb l1
                file_path = './model/'+'xgb_' + str(class_name) + str(n_fold) + '.model'
                # if os.path.exists(file_path):
                #     model = xgb.Booster({'nthread': 4})
                #     model.load_model(file_path)
                #     # xgb.Booster.load_model(model,file_path)
                # else:
                with timer("One XGB traing"):
                    model = xgb.train(
                        params=params,
                        dtrain=trn_subset,
                        num_boost_round=xgb_rounds,
                        evals=watchlist,
                        early_stopping_rounds=50,
                        verbose_eval=False
                    )
                # model.dump_model(file_path)
                xgb.Booster.save_model(model,file_path)

                class_pred[val_idx] = model.predict(val_subset)
                score = roc_auc_score(train_target.values[val_idx], class_pred[val_idx])

                # Compute mean rounds over folds for each class
                # So that it can be re-used for test predictions
                # xgb_round_dict[class_name] += model.best_iteration
                print("\t Fold %d : %.6f " % (n_fold + 1, score))

            print("full score : %.6f" % roc_auc_score(train_target, class_pred))
            scores.append(roc_auc_score(train_target, class_pred))
            train[class_name + "_oof"] = class_pred
            del trn_subset
            del val_subset
            gc.collect()

        # Save OOF predictions - may be interesting for stacking...
        file_name = 'oof/'+ 'xgb_'+str(feature_type) + '_oof.csv'
        train[["id"] + class_names + [f + "_oof" for f in class_names]].to_csv(file_name,
                                                                               index=False,
                                                                               float_format="%.8f")

        print('Total CV score is {}'.format(np.mean(scores)))

        # Use train for test
        pred = np.zeros( shape=(len(test), len(class_names)) )
        pred =pd.DataFrame(pred)
        pred.columns = class_names
#
        with timer("Predicting test probabilities"):
            # Go through all classes and reuse computed number of rounds for each class
            for class_name in class_names:
                with timer("Predicting probabilities for %s" % class_name):
                    train_target = train[class_name]
                    trn_xgbset = xgb.DMatrix(csr_trn, train_target)
                    # Train xgb
                    model = xgb.train(
                        params=params,
                        dtrain=trn_xgbset,
                        # num_boost_round=int(xgb_round_dict[class_name] / folds.n_splits)
                    )
                    file_path = './model/'+'xgb_' + str(class_name) + 'full' + '.model'
                    # model.dump_modle(file_path)
                    xgb.Booster.save_model(model, file_path)
                    pred[class_name] = model.predict(csr_sub)

        return pred


def app_rnn (train, test,embedding_path, feature_type, model_type):

    m_pred = app_train_rnn(train, test, embedding_path, model_type, feature_type)

    m_infile = './input/sample_submission.csv'
    m_outfile = './oof_test/' + str(model_type) + '_' + str(feature_type)+ '_test_oof.csv'
    m_make_single_submission(m_infile, m_outfile, m_pred)
    return

def app_lbg (train, test):
    class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    # test_r = test
    # with timer("Performing basic NLP"):
    #     get_indicators_and_clean_comments(train)
    #     get_indicators_and_clean_comments(test_r)

    # with timer ("gen tfidf features"):
    #     csr_trn, csr_sub =  f_gen_tfidf_features(train, test)

    # print (type(csr_trn))
    # save_sparse_csr('word_tchar_trn.csr',csr_trn)
    # save_sparse_csr('word_tchar_test.csr',csr_sub)

    with timer ("load pretrained feature"):
        csr_trn_1 = load_sparse_csr('./trained_features/word_tchar_char_short_trn.npz')
        csr_sub_1 = load_sparse_csr('./trained_features/word_tchar_char_short_test.npz')

    drop_f = [f_ for f_ in train if f_ not in ["id"] + class_names]
    train.drop(drop_f, axis=1, inplace=True)

    model_type = 'xgboost'
    feature_type = 'wtcs'
    if model_type == 'xgb':
        with timer ("get xgb model"):
            m_pred = app_train_xgb(csr_trn_1, csr_sub_1, train, test, feature_type)
    elif model_type == 'lgb':
        with timer ("get lgb model"):
            m_pred = m_lgb_model(csr_trn_1, csr_sub_1, train, test, feature_type)
    elif model_type == 'nbsvm':
        with timer ("get nbsvm model"):
            m_pred = app_train_nbsvm(csr_trn_1, csr_sub_1, train, test, feature_type)

    m_infile = './input/sample_submission.csv'
    m_outfile = './oof_test/' + str(model_type) + str(feature_type)+ '_test_oof.csv'
    m_make_single_submission(m_infile, m_outfile, m_pred)
    return

def app_tfidf_xbg (train, test):
    class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    # test_r = test
    # with timer("Performing basic NLP"):
    #     get_indicators_and_clean_comments(train)
    #     get_indicators_and_clean_comments(test_r)

    # with timer ("gen tfidf features"):
    #     csr_trn, csr_sub =  f_gen_tfidf_features(train, test)

    # print (type(csr_trn))
    # save_sparse_csr('word_tchar_trn.csr',csr_trn)
    # save_sparse_csr('word_tchar_test.csr',csr_sub)

    with timer ("load pretrained feature"):
        csr_trn_1 = load_sparse_csr('./trained_features/word_tchar_char_short_trn.npz')
        csr_sub_1 = load_sparse_csr('./trained_features/word_tchar_char_short_test.npz')

    drop_f = [f_ for f_ in train if f_ not in ["id"] + class_names]
    train.drop(drop_f, axis=1, inplace=True)

    print (csr_trn_1.shape)
    param = app_single_xgb(csr_trn_1, csr_sub_1,train, test, feature_type)
    for k in param:
        print ("%s : %.1f" % (k, param[k]))

    return


def app_glove_nbsvm (train, test,embedding_path, feature_type, model_type):
    class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    train_target = train[class_names]
    test_text = test["comment_text"]
    train_text = train["comment_text"]


    splits = 3
    max_len = 150
    max_features = 100000
    embed_size = 300
    m_batch_size = 32
    m_epochs = 3
    m_verbose = 1
    lr = 1e-3
    lr_d = 0
    units = 128
    dr = 0.2

    model_type = 'nbsvm'
    feature_type = 'glove'

    with timer("get pretrain features for rnn"):
        train_r,test_r, embedding_matrix = f_get_pretraind_features(train_text, test_text, embed_size, embedding_path,max_features, max_len)

    pred =  app_train_nbsvm(train_r, test_r,train, test, feature_type)

    m_infile = './input/sample_submission.csv'
    m_outfile = './oof_test/' + str(model_type) + str(feature_type)+ '_test_oof.csv'
    m_make_single_submission(m_infile, m_outfile, pred)

    return

def app_token_rnn(train, test, embedding_path, model_type, feature_type):

    class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    train_target = train[class_names]
    test_text = test["comment_text"].fillna("jiehunt").values
    train_text = train["comment_text"].fillna("jiehunt").values

    splits = 3

    max_len = 150
    max_features = 100000
    embed_size = 300

    m_batch_size = 32
    m_epochs = 3
    m_verbose = 1
    lr = 1e-3
    lr_d = 0
    units = 128
    dr = 0.2
    embedding_matrix = None
    m_trainable = True

    class_pred = np.ndarray(shape=(len(train), len(class_names)))

    # with timer("get pretrain features for rnn"):
    #     train_r,test, embedding_matrix = f_get_pretraind_features(train_text, test_text, embed_size, embedding_path,max_features, max_len)
    # with timer ("load pretrained feature"):
    #     train_r = load_sparse_csr('./trained_features/word_tchar_char_short_trn.npz')
    #     test_r = load_sparse_csr('./trained_features/word_tchar_char_short_test.npz')

    with timer("prepare tokenizer features"):
        tokenizer = text.Tokenizer(num_words=max_features)
        tokenizer.fit_on_texts(list(train_text))
        list_tokenized_train = tokenizer.texts_to_sequences(train_text)
        list_tokenized_test = tokenizer.texts_to_sequences(test_text)
        train_r = sequence.pad_sequences(list_tokenized_train, maxlen=max_len)
        test_r = sequence.pad_sequences(list_tokenized_test, maxlen=max_len)

    with timer("Goto Train RNN Model"):
        folds = KFold(n_splits=splits, shuffle=True, random_state=1)

        for n_fold, (trn_idx, val_idx) in enumerate(folds.split(train_r, train_target)):

            print ("goto %d fold :" % n_fold)
            X_train_n = train_r[trn_idx]
            Y_train_n = train_target.iloc[trn_idx]
            X_valid_n = train_r[val_idx]
            Y_valid_n = train_target.iloc[val_idx]

            if model_type == 'gru': # gru
                file_path = './model/'+str(model_type) +'_'+str(feature_type) + str(n_fold) + '.hdf5'
                if os.path.exists(file_path):
                    model = load_model(file_path)
                else:
                    model = m_gru_model(max_len, max_features, embed_size, embedding_matrix,
                                    X_valid_n, Y_valid_n, X_train_n,  Y_train_n, file_path,
                                    m_trainable=m_trainable, lr=lr, lr_d = lr_d, units = units, dr = dr,
                                    m_batch_size= m_batch_size, m_epochs = m_epochs, m_verbose = m_verbose)
            elif model_type == 'lstm': # lstm
                file_path = './model/'+str(model_type) +'_'+str(feature_type) + str(n_fold) + '.hdf5'
                if os.path.exists(file_path):
                    model = load_model(file_path)
                else:
                    model = m_lstm_model(max_len, max_features, embed_size, embedding_matrix,
                                X_valid_n, Y_valid_n, X_train_n,  Y_train_n, file_path,
                                m_trainable=True, lr = lr, lr_d = lr_d, units = units, dr = dr,
                                m_batch_size= m_batch_size, m_epochs = m_epochs, m_verbose = m_verbose)

            class_pred[val_idx] =pd.DataFrame(model.predict(X_valid_n))

        oof_names = ['toxic_oof', 'severe_toxic_oof', 'obscene_oof', 'threat_oof', 'insult_oof', 'identity_hate_oof']
        class_pred = pd.DataFrame(class_pred)
        class_pred.columns = oof_names
        train_oof = pd.concat([train,class_pred], axis = 1)
        for class_name in class_names:
            print("Class %s scores : " % class_name)
            print("%.6f" % roc_auc_score(train_target[class_name], class_pred[class_name+"_oof"]))

        # Save OOF predictions - may be interesting for stacking...
        file_name = 'oof/'+str(model_type) + '_' + str(feature_type) + '_oof.csv'
        train_oof[["id"] + class_names + [f + "_oof" for f in class_names]].to_csv(file_name,
                                                                               index=False,
                                                                               float_format="%.8f")

        X_train, X_valid, Y_train, Y_valid = train_test_split(train_r, train_target, test_size = 0.1)

        # Use train for test
        if model_type == 'gru': # gru
            file_path = './model/'+str(model_type) + '_'+ str(feature_type) + 'full' + '.hdf5'
            if os.path.exists(file_path):
                model = load_model(file_path)
            else:
                model = m_gru_model(max_len, max_features, embed_size, embedding_matrix,
                            X_valid, Y_valid, X_train,  Y_train, file_path,
                            m_trainable=True, lr=lr, lr_d = lr_d, units = units, dr = dr,
                            m_batch_size= m_batch_size, m_epochs = m_epochs, m_verbose = m_verbose)
        elif model_type == 'lstm': # lstm
            file_path = './model/'+str(model_type) + '_'+ str(feature_type) + 'full' + '.hdf5'
            if os.path.exists(file_path):
                model = load_model(file_path)
            else:
                model = m_lstm_model(max_len, max_features, embed_size, embedding_matrix,
                            X_valid, Y_valid, X_train,  Y_train, file_path,
                            m_trainable=True, lr = lr, lr_d = lr_d, units = units, dr = dr,
                            m_batch_size= m_batch_size, m_epochs = m_epochs, m_verbose = m_verbose)

        pred =pd.DataFrame(model.predict(test_r))
        pred.columns = class_names


        m_infile = './input/sample_submission.csv'
        m_outfile = './oof_test/' + str(model_type) + '_' + str(feature_type)+ '_test_oof.csv'
        m_make_single_submission(m_infile, m_outfile, pred)

    return

def app_token_lgb(train, test, model_type, feature_type):

    class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    train_target = train[class_names]
    test_text = test["comment_text"].fillna("jiehunt").values
    train_text = train["comment_text"].fillna("jiehunt").values

    splits = 3
    max_len = 150
    max_features = 100000

    drop_f = [f_ for f_ in train if f_ not in ["id"] + class_names]
    train.drop(drop_f, axis=1, inplace=True)

    class_pred = np.ndarray(shape=(len(train), len(class_names)))

    with timer("prepare tokenizer features"):
        tokenizer = text.Tokenizer(num_words=max_features)
        tokenizer.fit_on_texts(list(train_text))
        list_tokenized_train = tokenizer.texts_to_sequences(train_text)
        list_tokenized_test = tokenizer.texts_to_sequences(test_text)
        train_r = sequence.pad_sequences(list_tokenized_train, maxlen=max_len)
        test_r = sequence.pad_sequences(list_tokenized_test, maxlen=max_len)

    if model_type == 'xgb':
        with timer ("get xgb model"):
            m_pred = app_train_xgb(train_r, test_r, train, test, feature_type)
    elif model_type == 'lgb':
        with timer ("get lgb model"):
            m_pred = m_lgb_model(train_r, test_r, train, test, feature_type)
    elif model_type == 'nbsvm':
        with timer ("get nbsvm model"):
            m_pred = app_train_nbsvm(train_r, test_r, train, test, feature_type)

    m_infile = './input/sample_submission.csv'
    m_outfile = './oof_test/' + str(model_type) + str(feature_type)+ '_test_oof.csv'
    m_make_single_submission(m_infile, m_outfile, m_pred)
    return

def app_glove_lgb (train, test,embedding_path, feature_type, model_type):
    class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    train_target = train[class_names]
    test_text = test["comment_text"]
    train_text = train["comment_text"]

    splits = 3
    max_len = 150
    max_features = 100000
    embed_size = 300

    model_type = 'xgb'
    feature_type = 'glove'

    drop_f = [f_ for f_ in train if f_ not in ["id"] + class_names]
    train.drop(drop_f, axis=1, inplace=True)

    with timer("get pretrain features for rnn"):
        train_r,test_r, embedding_matrix = f_get_pretraind_features(train_text, test_text, embed_size, embedding_path,max_features, max_len)

    print (type(train_r))
    print (train_r.shape)

    # app_single_lgb(train_r, test_r,train, test, feature_type)
    # m_infile = './input/sample_submission.csv'
    # m_outfile = './oof_test/' + str(model_type) + str(feature_type)+ '_test_oof.csv'
    # m_make_single_submission(m_infile, m_outfile, pred)

    return



if __name__ == '__main__':
    train = pd.read_csv('./input/train.csv').fillna(' ')
    test = pd.read_csv('./input/test.csv').fillna(' ')

    glove_embedding_path = "./input/glove.840B.300d.txt"
    fasttext_embedding_path = './input/crawl-300d-2M.vec'

    train["comment_text"].fillna("no comment")
    test["comment_text"].fillna("no comment")

    # app_stack()

    # print ("goto glove nbsvm")
    # app_glove_nbsvm (train, test,glove_embedding_path, 'glove', 'nbsvm')

    # print ("goto tfidf")
    # app_lbg(train, test)

    # print ("goto rnn")
    model_type = 'capgru' # gru
    feature_type = 'glove'
    app_rnn(train, test, glove_embedding_path, feature_type, model_type)
    # app_rnn(train, test, fasttext_embedding_path, 'fast', model_type)

    # print ("goto tfidf rnn")
    # model_type = 'nbsvm'
    # feature_type = 'token'
    # app_token_rnn(train, test, None, model_type, feature_type)
    # app_token_lgb(train, test, model_type, feature_type)

    # app_glove_lgb (train, test,glove_embedding_path, feature_type, model_type)

    # app_tfidf_xbg (train, test)

""""""""""""""""""""""""""""""
# Ganerate Result
""""""""""""""""""""""""""""""
