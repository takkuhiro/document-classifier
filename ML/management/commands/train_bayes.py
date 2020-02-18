#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
ナイーブベイズ分類器の訓練
"""
from django.core.management.base import BaseCommand

import pickle
import configparser

from ...bayes import NaiveBayesClassifier


class Command(BaseCommand):
    help = 'ナイーブベイズ分類器の訓練を行うコマンド'

    def handle(self, *args, **kwargs):
        config_file = 'ML/config.ini'
        config_ini = configparser.ConfigParser()
        config_ini.read(config_file, encoding='utf-8')
        train_file = config_ini['Common']['train_file']
        model_file = config_ini['Bayes']['model_file']

        nb = NaiveBayesClassifier()
        with open(train_file, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')
        for line in lines:
            nb.train(line)
        with open(model_file, 'wb') as f:
            pickle.dump(nb, f)
