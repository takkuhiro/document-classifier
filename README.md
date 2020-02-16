# 動作方法  

```docker-compose up -d```
```docker exec -it doc_web_1 bash```  
# データ収集・整形・データ分割（訓練データとテストデータ）  
python manage.py data_collection
# モデル訓練：ナイーブベイズ分類器
python manage.py train_bayes
# モデル訓練：Random Forest
python manage.py train_randomforest
# モデル訓練：BERT
python manage.py train_bert  

[URL](http://127.0.0.1:8000/)にアクセスする。


- データ収集・整形・データ分割（訓練データとテストデータ）  
  ```python manage.py data_collection```  
- モデル訓練  
  - ナイーブベイズ:   
    ```cd /code  
    python manage.py train_bayes  ```  
  - Random Forest:  
    ```cd /code  
    python manage.py train_randomforest```  

# 精度  
| モデル | マクロ平均Precision | マクロ平均Recall | マクロ平均F1 | 重み付き平均Precision | 重み付き平均Recall | 重み付き平均F1 |  
| --- | --- | --- | --- | --- | --- | --- |  
| ナイーブベイズ(名詞のみ) | 0.74 | 0.69 | 0.63 | 0.86 | 0.64 | 0.67 |  
| ナイーブベイズ(全単語) | 0.88 | 0.86 | 0.86 | 0.90 | 0.90 | 0.90 |  
| Random Forest(名詞のみ) | 0.86 | 0.83 | 0.84 | 0.88 | 0.88 | 0.88 |  
| Random Forest(全単語) | 0.86 | 0.83 | 0.84 | 0.89 | 0.89 | 0.88 |  
| BERT | 0.84 | 0.84 | 0.84 | 0.88 | 0.88 | 0.88 |  


# 工夫点  
- ナイーブベイズとRandom Forestは、名詞のみの場合と全単語の場合を試して比較した。(BERTは文脈を考慮しているので名詞のみでの実行はしていない。)
- BERTは、訓練時に検証データを用いてLossが一定以上低下しなくなった状態が3epoch続いた場合は訓練を中止した。（EarlyStopping）最終的には7epoch時のパラメタを利用する。

# 補足
- BERT設定はmax_lengthが128、訓練時バッチサイズが4、それ以外はDevlinらのBERT-BASEに従う。
- BERT記事参照: https://github.com/nekoumei/DocumentClassificationUsingBERT-Japanese  
- BERT事前学習モデル：東北大学　乾・鈴木研究室が公開している事前学習モデルを利用(BERT-base_mecab-ipadic-bpe-32k_whole-word-mask)(https://github.com/cl-tohoku/bert-japanese)
- BERTベースの分類器の訓練では、訓練データ・検証データ・テストデータを利用する。上記分類器の訓練データをもとに、BERT用の訓練データと検証データを作成する。テストデータは他分類器の場合と同じである。
- ナイーブベイズ分類器とRandom Forestでは、今回はパラメタ探索を行わないため、検証データも訓練データに加える。
- データは収集時期をずらすことで多く集めることができる。（今回は収集を行ったのは1回のみ。時間があれば訓練データ8万件程度は集めたい。）
- 形態素解析ツールとして、ナイーブベイズ分類器とRandomForestではJanomeを、BERTではMecabを利用している。(from transformer import BertJapaneseTokenizerによる「MeCab+WordPiece, whole word masking」を利用している。)

# 今後の改善点
- 追加の学習データ収集（可能であれば訓練データ10万件程度まで収集したい。）
- Mecab Neologd辞書を使う。（記事中に固有名詞等が多いため。）
- BERTにおける改善
    - DevlinらのBERT-BASE, BERT-LARGEのモデル設定の利用
    - パラメタ探索(正例と負例の更新重み・学習率・エポック)
    - タイトルと本文でSegment Embeddingを変更する。
    - max_lengthを超えた時のtruncated手法は文書先頭より文書後方を残すようにした方が良さそう。（参照："How to Fine-Tuning BERT for Text Classification" China National Conference on Chinese Computational Linguistics[Chi Sun et al., 2019]）
