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

from mybase import *
""""""""""""""""""""""""""""""
# system setting
""""""""""""""""""""""""""""""
warnings.filterwarnings('ignore')
os.environ["OMP_NUM_THREADS"] = "4"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

if __name__ == '__main__':
    train = pd.read_csv('./input/train.csv').fillna(' ')
    test = pd.read_csv('./input/test.csv').fillna(' ')

    glove_embedding_path = "./input/glove.840B.300d.txt"
    fasttext_embedding_path = './input/crawl-300d-2M.vec'

    train["comment_text"].fillna("no comment")
    test["comment_text"].fillna("no comment")


    # print ("goto tfidf")
    # app_lbg(train, test)

    print ("goto rnn")
    app_rnn(train, test, glove_embedding_path)



""""""""""""""""""""""""""""""
# Stacking
""""""""""""""""""""""""""""""

""""""""""""""""""""""""""""""
# Ganerate Result
""""""""""""""""""""""""""""""
