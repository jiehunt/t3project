import os
import time
import warnings
import numpy as np
import pandas as pd
import sys, re, csv, codecs
import string
import logging
import psutil

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

def m_lgb_model(csr_trn, csr_sub, train):

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
        # "device": "gpu",
        # "gpu_platform_id": 0,
        # "gpu_device_id": 0
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
        train[["id"] + class_names + [f + "_oof" for f in class_names]].to_csv("lvl0_lgbm_clean_oof.csv",
                                                                               index=False,
                                                                               float_format="%.8f")

        print('Total CV score is {}'.format(np.mean(scores)))
        return model




""""""""""""""""""""""""""""""
# Train
""""""""""""""""""""""""""""""
def m_make_single_submission(m_infile, m_outfile, m_pred):
    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    submission = pd.read_csv(m_infile)
    submission[list_classes] = (m_pred)
    submission.to_csv(m_outfile, index = False)

def app_train_rnn(train, test, embedding_path, model_type):

    class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    train_target = train[class_names]
    test_text = test["comment_text"]
    train_text = train["comment_text"]

    splits = 3

    max_len = 150
    max_features = 100000
    embed_size = 300

    m_batch_size = 32
    m_epochs = 2
    m_verbose = 1
    lr = 1e-3
    lr_d = 0
    units = 128
    dr = 0.2

    class_pred = pd.DataFrame()
    for class_name in class_names:
        class_pred[class_name + "_oof"] = np.zeros(len(train))
        print (class_pred[class_name + "_oof"].shape)


    # train = pd.concat([train,class_pred])
    print (class_pred.shape)

    with timer("get pretrain features for rnn"):
        train_r,test, embedding_matrix = f_get_pretraind_features(train_text, test_text, embed_size, embedding_path,max_features, max_len)

    print (train_r.shape)
    ## ndarray type train
    # X_train_t = train[:X_train.shape[0]]
    # X_valid_t = train[X_train.shape[0]:]

    with timer("Goto Train RNN Model"):
        # scores = []
        folds = KFold(n_splits=splits, shuffle=True, random_state=1)
        # trn_lgbset = lgb.Dataset(csr_trn, free_raw_data=False)
        # class_pred = np.zeros(len(train))
        # del csr_trn
        # gc.collect()

        for n_fold, (trn_idx, val_idx) in enumerate(folds.split(train_r, train_target)):

            print (class_pred.shape)
            print ("goto %d fold :" % n_fold)
            X_train_n = train_r[trn_idx]
            Y_train_n = train_target.iloc[trn_idx]
            X_valid_n = train_r[val_idx]
            Y_valid_n = train_target.iloc[val_idx]
            print (type(X_train_n)) # ndarray
            print (type(Y_train_n)) # ndarray
            print (X_train_n.shape) # ndarray
            print (Y_train_n.shape) # ndarray

            if model_type == 'gru': # gru
              file_path = './model/gru.hdf5'

              model = load_model(file_path)
              # model = m_gru_model(max_len, max_features, embed_size, embedding_matrix,
              #                   X_valid_n, Y_valid_n, X_train_n,  Y_train_n, file_path,
              #                   m_trainable=False, lr=lr, lr_d = lr_d, units = units, dr = dr,
              #                   m_batch_size= m_batch_size, m_epochs = m_epochs, m_verbose = m_verbose)
            elif model_type == 'lstm': # lstm
              file_path = './model/lstm.hdf5'
              model = m_lstm_model(max_len, max_features, embed_size, embedding_matrix,
                                X_valid_n, Y_valid_n, X_train_n,  Y_train_n, file_path,
                                m_trainable=False, lr = lr, lr_d = lr_d, units = units, dr = dr,
                                m_batch_size= m_batch_size, m_epochs = m_epochs, m_verbose = m_verbose)

            class_pred.iloc[val_idx] =pd.DataFrame(model.predict(X_valid_n))
            # class_pred = model.predict(X_valid_n)
            print (type(class_pred))
            print (class_pred.shape)
            # class_pred[val_idx] = model.predict(X_valid_n[val_idx])
            # score = roc_auc_score(train_target.values[val_idx], class_pred[val_idx])

            # # Compute mean rounds over folds for each class
            # # So that it can be re-used for test predictions
            # print("\t Fold %d : %.6f" % (n_fold + 1, score))

        # print("full score : %.6f" % roc_auc_score(train_target, class_pred))
        # scores.append(roc_auc_score(train_target, class_pred))
        # train[class_name + "_oof"] = class_pred
        train = pd.concat([train,class_pred])
        for class_name in class_names:
            print("Class %s scores : " % class_name)
            print("%.6f" % roc_auc_score(train[class_name].values, train[class_name+"_oof"].values))


        # Save OOF predictions - may be interesting for stacking...
        train[["id"] + class_names + [f + "_oof" for f in class_names]].to_csv("lgbm_clean_oof.csv",
                                                                               index=False,
                                                                               float_format="%.8f")

        # print('Total CV score is {}'.format(np.mean(scores)))
#
#     if model_type == 'gru': # gru
#       file_path = './model/gru.hdf5'
#       model = m_gru_model(max_len, max_features, embed_size, embedding_matrix,
#                         X_valid_t, Y_valid, X_train_t,  Y_train, file_path,
#                         m_trainable=False, lr=lr, lr_d = lr_d, units = units, dr = dr,
#                         m_batch_size= m_batch_size, m_epochs = m_epochs, m_verbose = m_verbose)
#     elif model_type == 'lstm': # lstm
#       file_path = './model/lstm.hdf5'
#       model = m_lstm_model(max_len, max_features, embed_size, embedding_matrix,
#                         X_valid_t, Y_valid, X_train_t,  Y_train, file_path,
#                         m_trainable=False, lr = lr, lr_d = lr_d, units = units, dr = dr,
#                         m_batch_size= m_batch_size, m_epochs = m_epochs, m_verbose = m_verbose)
#
#     pred = model.predict(test, m_batch_size, m_verbose)

    return
    # return pred

def app_rnn (train, test,embedding_path):

    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    y = train[list_classes].values

    X_train, X_valid, Y_train, Y_valid = train_test_split(train, y, test_size = 0.1)
    # print (type(X_train))
    model_type = 'gru' # gru
    m_pred = app_train_rnn(train, test, embedding_path, model_type)

    # m_infile = './input/sample_submission.csv'
    # m_outfile = './res/submission_gru.csv'
    # m_make_single_submission(m_infile, m_outfile, m_pred)
    return

def app_lbg (train, test):
    class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    with timer("Performing basic NLP"):
        get_indicators_and_clean_comments(train)
        get_indicators_and_clean_comments(test)

    with timer ("gen tfidf features"):
        csr_trn, csr_sub =  f_gen_tfidf_features(train, test)

    print (type(csr_trn))
    save_sparse_csr('word_tchar_trn.csr',csr_trn)
    save_sparse_csr('word_tchar_test.csr',csr_sub)

    csr_trn_1 = load_sparse_csr('word_tchar_trn.csr')
    csr_sub_1 = load_sparse_csr('word_tchar_trn.csr')
    drop_f = [f_ for f_ in train if f_ not in ["id"] + class_names]
    train.drop(drop_f, axis=1, inplace=True)

    with timer ("get model"):
        model = m_lgb_model(csr_trn_1, csr_sub_1, train)

# print ("goto app_rnn")
# app_rnn(train, test)

if __name__ == '__main__':
    train = pd.read_csv('./input/train.csv').fillna(' ')
    test = pd.read_csv('./input/test.csv').fillna(' ')

    glove_embedding_path = "./input/glove.840B.300d.txt"
    fasttext_embedding_path = './input/crawl-300d-2M.vec'

    train["comment_text"].fillna("no comment")
    test["comment_text"].fillna("no comment")


    print ("goto tfidf")
    app_lbg(train, test)

    print ("goto rnn")
    app_rnn(train, test, glove_embedding_path)

""""""""""""""""""""""""""""""
# Stacking
""""""""""""""""""""""""""""""

""""""""""""""""""""""""""""""
# Ganerate Result
""""""""""""""""""""""""""""""
