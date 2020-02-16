[![Build Status](https://travis-ci.com/takkuhiro/document-classifier.svg?token=pNQtko6iUq77KkHVx7Jv&branch=master)](https://travis-ci.com/takkuhiro/document-classifier)
# 動作方法  

```
cd document-classifier  
docker-compose build   
docker-compose up -d  
docker exec -it doc_web_1 bash  

# データ収集・整形・データ分割  
python manage.py data_collection  

# モデル訓練：ナイーブベイズ分類器  
python manage.py train_bayes  

# モデル訓練：Random Forest  
python manage.py train_randomforest  

# モデル訓練：BERT(所要時間：1~2時間程度)  
python manage.py train_bert  
```

上記コマンド後[URL](http://127.0.0.1:8000/)にアクセスする。

# モデルの評価
```
python /code/ML/evaluate.py
```
マクロ平均と重み付き平均におけるPrecision, Recall, F1で評価を行う。

# 精度  
| モデル | マクロ平均Precision | マクロ平均Recall | マクロ平均F1 | 重み付き平均Precision | 重み付き平均Recall | 重み付き平均F1 |  
| --- | --- | --- | --- | --- | --- | --- |  
| ナイーブベイズ(名詞のみ) | 0.74 | 0.69 | 0.63 | 0.86 | 0.64 | 0.67 |  
| ナイーブベイズ(全単語) | 0.88 | 0.86 | 0.86 | 0.90 | 0.90 | 0.90 |  
| Random Forest(名詞のみ) | 0.86 | 0.83 | 0.84 | 0.88 | 0.88 | 0.88 |  
| Random Forest(全単語) | 0.86 | 0.83 | 0.84 | 0.89 | 0.89 | 0.88 |  
| BERT | 0.84 | 0.84 | 0.84 | 0.88 | 0.88 | 0.88 |   

※ナイーブベイズとRandomForestにおいては、パラメタ探索を行わないため、検証データも訓練データの一部として利用している。

# 工夫点  
- ナイーブベイズとRandom Forestは、名詞のみの場合と全単語の場合を試して比較した。(BERTは文脈を考慮しているので名詞のみでの実行はしていない。)
- BERTは、訓練時に検証データを用いてLossが一定以上低下しなくなった状態が3epoch続いた場合は訓練を中止した。（EarlyStopping）これにより、最終的に7epoch時のパラメタを利用した。

# 補足
- BERT設定はmax_lengthが128、訓練時バッチサイズが4、それ以外はDevlinらのBERT-BASEに従う。
- BERT記事参照: https://github.com/nekoumei/DocumentClassificationUsingBERT-Japanese  
- BERT事前学習モデル：東北大学　乾・鈴木研究室が公開している事前学習モデルを利用([BERT-base_mecab-ipadic-bpe-32k_whole-word-mask](https://github.com/cl-tohoku/bert-japanese))
- データの収集時期をずらすことで、追加のデータ集めることができる。上記スコアは、収集を1回のみ行って実験したものである。
- 利用した形態素解析ツール  
    - ナイーブベイズ分類器, RandomForest：Janome
    - BERT：Mecab(ライブラリtransformer中のBertJapaneseTokenizer(MeCab+WordPiece, whole word masking))

# 今後の改善点  
- 追加の学習データ収集（可能であれば訓練データ10万件程度まで収集したい。）
- [mecab-ipadic-NEologd辞書](https://github.com/neologd/mecab-ipadic-neologd)の利用（記事中に固有名詞等が多いため。）
- BERTにおける改善
    - DevlinらのBERT-BASE, BERT-LARGEのモデル設定での利用
    - パラメタ探索(正例と負例の更新重み・学習率・エポック)
    - タイトルと本文でSegment Embeddingを変更する。
    - max_lengthを超えた時のtruncated手法として文書中間を削除し、文書先頭より文書後方を重点的に残す。例えば、先頭128単語と後方384単語など。（参照："How to Fine-Tuning BERT for Text Classification" China National Conference on Chinese Computational Linguistics[Chi Sun et al., 2019](http://cips-cl.org/static/anthology/CCL-2019/CCL-19-141.pdf)）
