#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
RandomForestをベースとした分類器
"""
from django.db import models

from janome.tokenizer import Tokenizer
from sklearn.ensemble import RandomForestClassifier
from gensim import corpora, matutils
from tqdm import tqdm
import pickle
import configparser


def extract_tokens(text, already_tokenize=True):
    """
    形態素解析をしてトークンのリストを返す
    
    Args:
        *text(str): 入力文
        *already_tokenize(bool): すでに単語分割されているか

    Returns:
        *_(list): 単語分割後の単語リスト
    """
    tokens = []
    if already_tokenize:
        return [token for token in text.split(' ')]
    else:
        t = Tokenizer()
        return [token.surface for token in t.tokenize(text)]


def random_forest_predict(text, already_tokenize=True):
    """
    RandomForestによる分類結果

    Args:
        *text(str): 入力文
        *already_tokenize(bool): すでに単語分割されているか

    Returns:
        *label_predict(str): 分類結果のカテゴリ
    """
    config_file = 'ML/config.ini'
    config_ini = configparser.ConfigParser()
    config_ini.read(config_file, encoding='utf-8')
    dic_file = config_ini['RandomForest']['dic_file']
    train_file = config_ini['Common']['train_file']
    categories = eval(config_ini['Common']['categories'])
    
    category_idx, scores = {}, {}
    for i, con in enumerate(categories):
        category_idx[con] = i
        scores[con] = 0.0

    with open(train_file, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
    
    dictionary = corpora.Dictionary.load_from_text(dic_file)
    with open(config_ini['RandomForest']['model_file'], 'rb') as f:
        est = pickle.load(f)

    #predict
    tmp = dictionary.doc2bow(extract_tokens(text, already_tokenize=True))
    dense = list(matutils.corpus2dense([tmp], num_terms=len(dictionary)).T[0])
    idx_predict = est.predict([dense])
    label_predict = [k for k, v in category_idx.items() if v == idx_predict[0]][0]
    return label_predict

