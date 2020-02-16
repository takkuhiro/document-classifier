#coding: utf-8
from django.core.management.base import BaseCommand

from gensim import corpora, matutils
from sklearn.ensemble import RandomForestClassifier
import pickle
import configparser
from tqdm import tqdm
from janome.tokenizer import Tokenizer

from ...randomforest import extract_tokens


class Command(BaseCommand):
    help = 'RandomForestの訓練を行うコマンド'

    def handle(self, *args, **kwargs):
        #RandomForest分類器の訓練
        config_file = 'ML/config.ini'
        config_ini = configparser.ConfigParser()
        config_ini.read(config_file, encoding='utf-8')
        train_file = config_ini['Common']['train_file']
        model_file = config_ini['RandomForest']['model_file']
        dic_file = config_ini['RandomForest']['dic_file']
        min_valid = int(config_ini['RandomForest']['min_valid'])
        category_idx, scores = {}, {}
        for i, label in enumerate(eval(config_ini['Common']['categories'])):
            category_idx[label] = i
            scores[label] = 0.0
        
        with open(train_file, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')
        
        #辞書作成
        content_words = []
        for line in lines:
            tmp = line.split('\t')
            _, content = tmp[0].rstrip(), ' [SEP] '.join(tmp[1:]).rstrip()
            content_words.append(extract_tokens(content, already_tokenize=True))
        dictionary = corpora.Dictionary(content_words)
        dictionary.filter_extremes(no_below=min_valid)
        dictionary.save_as_text(dic_file)

        #train
        ans, train_text = [], []
        for line in tqdm(lines):
            tmp = line.split('\t')
            category, content = tmp[0].rstrip(), ' [SEP] '.join(tmp[1:]).rstrip()
            ans.append(category_idx[category])
            tmp = dictionary.doc2bow(extract_tokens(content, already_tokenize=True))

            dense = list(matutils.corpus2dense([tmp], num_terms=len(dictionary)).T[0])
            train_text.append(dense)
        assert len(ans)==len(train_text), (len(ans), len(train_text))
        
        estimator = RandomForestClassifier()
        estimator.fit(train_text, ans)
        with open(model_file, 'wb') as f:
            pickle.dump(estimator, f)


