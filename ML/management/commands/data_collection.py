#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
データ収集
"""
from django.core.management.base import BaseCommand
import os
import sys
import time

import requests
import bs4
import configparser
import random
import numpy as np
from tqdm import tqdm
from janome.tokenizer import Tokenizer


def cleaning(text):
    cleaned_text = text
    cleaned_text = cleaned_text.replace('\u3000', ' ')
    cleaned_text = cleaned_text.replace('\n', '')
    cleaned_text = cleaned_text.replace('\'', '')

    return cleaned_text


class Command(BaseCommand):
    help = '小カテゴリそれぞれに対して5ページ分のデータを収集する。'
    config_file = 'ML/config.ini'
    config_ini = configparser.ConfigParser()
    config_ini.read(config_file, encoding='utf-8')
    target = config_ini['Common']['http']
    categories = eval(config_ini['Common']['categories'])
    article_file = config_ini['Common']['original_article_file']
    format_file = config_ini['Common']['format_file']
    train_file = config_ini['Common']['train_file']
    valid_file = config_ini['Common']['valid_file']
    test_file = config_ini['Common']['test_file']
    
    page_num = 5        #各カテゴリごとにあるページ数
    category_links = [[] for _ in range(len(categories))]
    contents = []
    idx = 0
    exec_flg = True
    
    #1. 収集対象のリンク先をまとめる
    link_cnt = 0
    res = requests.get(target)
    bs4obj = bs4.BeautifulSoup(res.text)
    for i in range(len(categories)):
        links = bs4obj.select('.nav_sub_list_'+str(i+1)+' a')
        for link in links:
            category_links[i].append(link.get('href'))
            link_cnt += 1
    assert  len(categories) == len(category_links), \
        (len(categories), len(category_links))

    for i, cls in enumerate(categories):
        for j, category_link in enumerate(category_links[i]):
            #各カテゴリあたりあるページ数分ループ
            for k in range(1, page_num+1):
                if k == 1:
                    target_page = category_link
                else:
                    target_page = category_link + '?page=' + str(k)
                res = requests.get(target_page)
                bs4obj = bs4.BeautifulSoup(res.text)
                links = bs4obj.select('.list_title a')
                
                #各記事からテキストを抽出
                for l, link in enumerate(tqdm(links)):
                    try:
                        res_each_page = requests.get(link.get('href'))
                        bs4obj2 = bs4.BeautifulSoup(res_each_page.text)
                        title_text = bs4obj2.select('.article_header_title')[0].getText()
                        title_text = cleaning(title_text)
                        body_text = bs4obj2.select('.article')[0].getText()
                        body_text = cleaning(body_text)
                        time.sleep(1.0)
                        
                        contents.append('\t'.join((cls, title_text, body_text)))
                    except:
                        print('強制終了...(i, j, k, l)==({}, {}, {}, {})'.format(i, j, k, l))
                        exec_flg = False
                        break
                if not exec_flg:
                    break
                time.sleep(20.0)
            if not exec_flg:
                break
        if not exec_flg:
            break

    contents = list(set(contents))
    with open(article_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(contents))

    #2. 整形
    t = Tokenizer()

    formatted_lines = []
    for line in tqdm(contents):
        tmp = line.split('\t')
        ans, content = tmp[0], '\t'.join(tmp[1:])

        formatted_line = []
        for token in t.tokenize(content):
            formatted_line.append(token.surface)
        formatted_lines.append(ans + '\t' + ' '.join(formatted_line))

    with open(format_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(formatted_lines))

    #3. train, valid, testにsplit
    random.shuffle(formatted_lines)
    train, valid, test = np.split(formatted_lines, [int(.6 * len(formatted_lines)), int(.8 * len(formatted_lines))])
    
    with open(train_file, 'w', encoding='utf-8') as f_train,\
         open(valid_file, 'w', encoding='utf-8') as f_valid,\
         open(test_file, 'w', encoding='utf-8') as f_test:
        f_train.write('\n'.join(train))
        f_valid.write('\n'.join(valid))
        f_test.write('\n'.join(test))
    


