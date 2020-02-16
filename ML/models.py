#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
内部処理
"""
from django.db import models

import requests
import bs4
import math
import pickle
from janome.tokenizer import Tokenizer

from .bayes import naive_bayes_classifier_predict
from .randomforest import random_forest_predict
from .bert import bert_predict


def show(url):
    """
    対象の記事の分類結果を返す関数
    Args:
        *url: 対象とする記事のURL
    Returns:
        *nb_category(str): ナイーブベイズ分類器出力結果
        *rf_category(str): RandomForest出力結果
        *be_category(str): BERT出力結果
    """
    t = Tokenizer()
    category_num = 8
    probs = [0 for _ in range(category_num)]

    if url == '':
        return '', '', ''
    info = requests.get(url, timeout=5.0)
    if info.status_code == '404':
        return '', '', ''

    obj = bs4.BeautifulSoup(info.text)
    extract_titles = obj.select('title')
    extract_bodys = obj.select('.gtm-click p')
    title_txt, body_txt = '', ''
    for ele in extract_titles:
        title_txt += ele.getText()
    for ele in extract_bodys:
        body_txt += ele.getText()
    title, body = [], []
    for token in t.tokenize(title_txt):
        title.append(token.surface)
    for token in t.tokenize(body_txt):
        body.append(token.surface)
    text = ' '.join(title) + '\t' + ' '.join(body)
    nb_category = naive_bayes_classifier_predict(text, already_tokenize=True)
    rf_category = random_forest_predict(text, already_tokenize=True)
    be_category = bert_predict(text)
    return nb_category, rf_category, be_category
