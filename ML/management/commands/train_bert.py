#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
BERTベースの分類器の訓練
"""
from django.core.management.base import BaseCommand
import os
import time

import configparser
from transformers import BertJapaneseTokenizer, BertForSequenceClassification
import mojimoji
import re
import collections
import torchtext
from torchtext.data import Field, Dataset, Example
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from logzero import logger
import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from tqdm import tqdm
import random
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from ...bert import EarlyStopping, DataFrameDataset
from ...bert import SeriesExample, DocumentClassifier


def make_df(lines, tokenizer, max_length, categories_idx, predict=False):
    """
    ファイルから読み込んだデータからDataFrameを作成
    Args:
        *lines(list): 入力データ
        *tokenizer: 利用する単語分割ツール
        *max_length(int): BERTの最大シーケンス長
        *categories_idx(dict): カテゴリとそのインデックスを変換するための辞書
        *predict: 推論時はdfとその答えを別々に返す
    Returns:
        *df: DataFrame
        *anss: 推論時に返す答えのラベル
    """
    anss, contents = [], []
    for line in lines:
        tmp = line.split('\t')
        try:
            ans = tmp[0].replace('\r', '').rstrip()
            content = ' [SEP] '.join(tmp[1:]).rstrip()
            content = content.replace(' ', '')
        except IndexError:
            ans = line
            content = line
        anss.append(categories_idx[ans])
        contents.append(content)
    if predict:
        df = pd.DataFrame({'Text': contents})
        return df, anss
    else:
        df = pd.DataFrame({'Text': contents, 'Label': anss})
        return df


class Command(BaseCommand):
    help = 'BERTベースの分類器の訓練を行うコマンド'

    def handle(self, *args, **kwargs):
        config_file = 'ML/config.ini'
        config_ini = configparser.ConfigParser()
        config_ini.read(config_file, encoding='utf-8')
        train_file = config_ini['Common']['train_file']
        valid_file = config_ini['Common']['valid_file']
        test_file = config_ini['Common']['test_file']
        categories = eval(config_ini['Common']['categories'])
        model_file_dir = config_ini['BERT']['model_file_dir']
        pretrained_model = config_ini['BERT']['pretrained_model']
        max_length = int(config_ini['BERT']['max_length'])

        categories_idx = {}
        for i, con in enumerate(categories):
            categories_idx[con] = i
        tokenizer = BertJapaneseTokenizer.from_pretrained(pretrained_model)
        with open(train_file, 'r') as f_train:
            train_lines = f_train.read().split('\n')
        with open(valid_file, 'r') as f_valid:
            valid_lines = f_valid.read().split('\n')
        with open(test_file, 'r') as f_test:
            test_lines = f_test.read().split('\n')
        train_lines = [line for line in train_lines if line != '']
        valid_lines = [line for line in valid_lines if line != '']
        test_lines = [line for line in test_lines if line != '']
        train_df = make_df(train_lines, tokenizer, max_length, categories_idx)
        val_df = make_df(valid_lines, tokenizer, max_length, categories_idx)

        model = DocumentClassifier(max_length=max_length,
                                   batch_size=4,
                                   num_labels=8,
                                   num_epochs=10,
                                   )
        model.fit(train_df, val_df, model_file_dir, early_stopping_rounds=3)

        # test
        test_df, true = make_df(test_lines, tokenizer, max_length, categories_idx, predict=True)
        scores = model.predict(test_df)
        preds = np.argmax(scores, axis=1)

        result = classification_report(true, preds)
        print(result)
        mx = confusion_matrix(true, preds)
        print(mx)
