#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
モデル性能を評価
"""
from django.db import models

from janome.tokenizer import Tokenizer
from gensim import corpora, matutils
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
import configparser
import pickle
import math
from transformers import BertJapaneseTokenizer
import pandas as pd
import numpy as np

from bayes import NaiveBayesClassifier
from randomforest import extract_tokens
from bert import DocumentClassifier


def evaluate_bayes():
    """
    ナイーブベイズを使った教師あり分類器の評価関数
    """
    config_file = '/code/ML/config.ini'
    config_ini = configparser.ConfigParser()
    config_ini.read(config_file, encoding='utf-8')
    train_file = config_ini['Common']['train_file']
    test_file = config_ini['Common']['test_file']
    categories = eval(config_ini['Common']['categories'])
    nb_model = config_ini['Bayes']['model_file']
    rf_model = config_ini['RandomForest']['model_file']

    category_idx = {}
    for i, con in enumerate(categories):
        category_idx[con]  = i

    nb = NaiveBayesClassifier()
    with open(train_file, 'r') as f:
        lines = f.read().split('\n')
    print('Start training naive bayes')
    for line in tqdm(lines):
        nb.train(line)
    print('Finish training naive bayes')

    with open(test_file, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
    
    #Bayes評価
    all_num = len(lines)
    true, pred = [], []
    for line in tqdm(lines):
        tmp = line.split('\t')
        ans, content = tmp[0], '\t'.join(tmp[1:])
        category = nb.classifier(content, already_tokenize=True)
        pred.append(category_idx[category])
        true.append(category_idx[ans])

    result = classification_report(true, pred)
    print(result)
    mx = confusion_matrix(true, pred)
    print(mx)


def evaluate_randomforest():
    """
    RandomForest分類器の評価
    """
    config_file = '/code/ML/config.ini'
    config_ini = configparser.ConfigParser()
    config_ini.read(config_file, encoding='utf-8')
    train_file = config_ini['Common']['train_file']
    test_file = config_ini['Common']['test_file']
    categories = eval(config_ini['Common']['categories'])
    rf_model = config_ini['RandomForest']['model_file']
    dic_file = config_ini['RandomForest']['dic_file']
    
    category_idx = {}
    for i, con in enumerate(categories):
        category_idx[con]  = i
    
    with open(rf_model, 'rb') as f:
        est = pickle.load(f)
    with open(test_file, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
    dictionary = corpora.Dictionary.load_from_text(dic_file)

    #RandomForest評価
    pred, true = [], []
    for line in tqdm(lines):
        tmp = line.split('\t')
        ans, content = tmp[0], '\t'.join(tmp[1:])
        tmp = dictionary.doc2bow(extract_tokens(content, already_tokenize=True))
        dense = list(matutils.corpus2dense([tmp], num_terms=len(dictionary)).T[0])
        idx_predict = est.predict([dense])
        category = [k for k, v in category_idx.items() if v == idx_predict[0]][0]
        pred.append(category_idx[category])
        true.append(category_idx[ans])

    result = classification_report(true, pred)
    print(result)
    mx = confusion_matrix(true, pred)
    print(mx)


def evaluate_bert():
    """
    BERTベースの分類器の評価
    """
    config_file = '/code/ML/config.ini'
    config_ini = configparser.ConfigParser()
    config_ini.read(config_file, encoding='utf-8')
    test_file = config_ini['Common']['test_file']
    categories = eval(config_ini['Common']['categories'])
    model_file_dir = config_ini['BERT']['model_file_dir']
    model = DocumentClassifier(num_labels=8)
    model.load(model_file_dir)
    
    category_idx = {}
    for i, con in enumerate(categories):
        category_idx[con]  = i
    
    with open(test_file, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
    lines = [line for line in lines if line != '']
    true, pred = [], []
    for line in lines:
        tmp = line.split('\t')
        ans, content = tmp[0].rstrip(), '\t'.join(tmp[1:])
        content = content.replace(' ', '')
        df = pd.DataFrame({'Text': [content]})
        score = model.predict(df)
        idx = np.argmax(score)
        pred.append(idx)
        true.append(category_idx[ans])
    assert len(pred) == len(true) ,\
        'len(pred): {}, len(true): {}'.format(len(pred), len(true))

    result = classification_report(true, pred)
    print(result)
    mx = confusion_matrix(true, pred)
    print(mx)
    

if __name__=='__main__':
    evaluate_bayes()
    evaluate_randomforest()
    evaluate_bert()
