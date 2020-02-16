#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
ナイーブベイズを使った教師あり文書分類器
"""
from django.db import models

import pickle
import math
import configparser
from janome.tokenizer import Tokenizer


def naive_bayes_classifier_predict(text, already_tokenize=True):
    """
    ナイーブベイズを使った教師あり文書分類器を用いて推論を行う関数
    Args:
        *text(str): 入力文
        *already_tokenize (bool): すでに空白を挟んで単語分割されているか
    Returns:
        *category(str): 分類結果
    """
    config_file = 'ML/config.ini'
    config_ini = configparser.ConfigParser()
    config_ini.read(config_file, encoding='utf-8')
    model_file = config_ini['Bayes']['model_file']
    with open(model_file, 'rb') as f:
        nb = pickle.load(f)
    category = nb.classifier(text, already_tokenize=False)
    return category


class NaiveBayesClassifier:
    """
    ナイーブベイズを使った教師あり文書分類器
    """
    def __init__(self):
        self.vocab = set()
        config_file = '/code/ML/config.ini'
        config_ini = configparser.ConfigParser()
        config_ini.read(config_file, encoding='utf-8')
        categories = eval(config_ini['Common']['categories'])
        self.word_count, self.category_count = {}, {}
        for con in categories:
            self.word_count[con] = {}
            self.category_count[con] = 0
    
    def train(self, text):
        """
        分類器の学習
        Args:
            *text(str): 入力文
        """
        t = Tokenizer()
        tmp = text.split('\t')
        category, content = tmp[0].rstrip(), ' [SEP] '.join(tmp[1:]).rstrip()
        content_words = content.split(' ')
        for word in content_words:
            self.__word_count_up(word, category)
        """入力を名詞のみにする場合
        content = content.replace(' ', '')
        for token in t.tokenize(content):
            if token.part_of_speech.split(',')[0]=='名詞':
                self.__word_count_up(token.surface, category)
        """
        self.__category_count_up(category)
    
    def __word_count_up(self, word, category):
        """
        カテゴリごとの単語出現数を数え上げる
        Args:
            *word(str): 入力文中の単語
            *category(str): 入力文のカテゴリラベル
        """
        self.word_count[category].setdefault(word, 0)
        self.word_count[category][word] += 1
        self.vocab.add(word)
    
    def __category_count_up(self, category):
        """
        カテゴリ数を数え上げる
        Args:
            *category: カテゴリラベル
        """
        self.category_count[category] += 1
    
    def classifier(self, text, already_tokenize=True):
        """
        分類器による推論
        Args:
            *text(str): 入力文
            *already_tokenize(bool): すでに単語分割されているか
        Returns:
            *best_category(str): 分類結果のカテゴリ
        """
        best_category = None
        max_prob = -float('inf')
    
        tokenized_tokens = []
        if already_tokenize:
            tokenized_tokens = text.split(' ')
        else:
            t = Tokenizer()
            for token in t.tokenize(text):
                """入力を名詞のみにする場合
                if token.part_of_speech.split(',')[0]=='名詞':
                    tokenized_tokens.append(token.surface)
                """
                tokenized_tokens.append(token.surface)

        #P(Category|Document)が最大のカテゴリを選択
        for category in self.category_count.keys():
            prob = self.__score(tokenized_tokens, category)
            if prob > max_prob:
                best_category = category
                max_prob = prob
        return best_category

    def __score(self, word_list, category):
        """
        ナイーブベイズによる確率を計算する。
        P(C|D)=P(C)*P(D|C)/P(D)
        P(D)を固定して、P(C|D)∝P(C)*P(D|C)
        フロー処理として対数をとり、logP(D)+log(D|C)
        P(D|C)は、そのDocumentのそれぞれの単語の出現確率P(Wn|C)の掛け算で求まる（対数だと足し算）
        Args:
            *word_list(list): 入力文の単語リスト
            *category(str): 入力文に対するカテゴリ
        Returns:
            #score(float): 入力カテゴリに対する確率
        """
        score = math.log(self.__prior_prob(category))
        for word in word_list:
            score += math.log(self.__word_prob(word, category))
        return score

    # P(C)
    def __prior_prob(self, category):
        """
        __score関数内での確率P(C)を求める
        Args: 
            *category(str): カテゴリ

        Returns:
            *_(float): P(C)の値
        """
        prob = float(self.category_count[category] 
                     / sum(self.category_count.values()))
        return prob

    def __word_prob(self, word, category):
        """
        __score関数内での確率P(Wn|C)を求める。
        加算スムージングを利用する。
        Args:
            *word(str): 入力単語
            *category(str): カテゴリ
        Returns:
            *prob(float): P(Wn|C)の値
        """
        prob = (self.__in_category(word, category) + 1.0) 
                / (sum(self.word_count[category].values()) + len(self.vocab) * 1.0)
        return prob
    
    def __in_category(self, word, category):
        """
        単語のカテゴリー内出現回数を返す。
        __word_prob関数内で利用する。
        Args:
            *word(str): 入力単語
            *category(str): カテゴリ
        Returns:
            *_(float): 単語のカテゴリー内出現回数 
        """
        if word in self.word_count[category]:
            return float(self.word_count[category][word])
        return 0.0
