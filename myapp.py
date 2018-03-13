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

import glob

def h_xgb_tuning():
    param_test1 = {
        'max_depth': [2,3,4,5,6,7],
        'min_child_weight': [3,4,5,6,7]
    }
    param_test3 = {
        'gamma': [i / 10.0 for i in range(0, 5)]
    }
    param_test4 = {
        'subsample': [i / 10.0 for i in range(6, 10)],
        'colsample_bytree': [i / 10.0 for i in range(6, 10)]
    }
    param_test5 = {
        'subsample': [i / 100.0 for i in range(55, 75, 5)],
        'colsample_bytree': [i / 100.0 for i in range(55, 75, 5)]
    }
    param_test6 = {
        'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100]
    }
    param_test7 = {
        'learning_rate': [0.001, 0.01, 0.05, 0.1]
    }
    param_test8 = {
        'n_estimators': [1000,2000,3000,4000,5000]
    }

    param_dict = {
        'learning_rate' : 0.05
        'n_estimators'  : 1000
        'max_depth' : 3
        'min_child_weight':7
        'gamma':0
        'subsample':0.6
        'colsample_bytree':0.6
        'reg_alpha':1
    }

    gsearch1 = GridSearchCV(estimator=XGBClassifier(learning_rate=param_dict['learning_rate'],
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
                                                    objective='binary:logistic', nthread=4, scale_pos_weight=1,
                                                    seed=27),
                                                    param_grid=param_test, scoring='roc_auc', n_jobs=4, iid=False, cv=5, verbose=2)

    # class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    train_r = train.drop(class_names,axis=1)
    train_target = train[class_names]
    with timer("goto serch max_depth and min_child_wight"):
        gsearch1.fit(train_r, train_target['toxic'])
        print (gsearch1.grid_scores_ )
        print (gsearch1.best_params_ )
        print (type(gsearch1.best_params_) )
        print (gsearch1.best_score_)
        print (type(gsearch1.best_score_))

    return

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

    app_stack()
    # print ("goto tfidf")
    # app_lbg(train, test)

    # print ("goto rnn")
    # app_rnn(train, test, glove_embedding_path)



""""""""""""""""""""""""""""""
# Stacking
""""""""""""""""""""""""""""""

""""""""""""""""""""""""""""""
# Ganerate Result
""""""""""""""""""""""""""""""
