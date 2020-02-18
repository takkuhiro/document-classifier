# Docker Hubにあるpythonイメージをベースにする
FROM python:3.6

# 環境変数を設定する
ENV PYTHONUNBUFFERED 1

# コンテナ内にcodeディレクトリを作り、そこをワークディレクトリとする
RUN mkdir /code
WORKDIR /code

# ホストPCにあるrequirements.txtをコンテナ内のcodeディレクトリにコピーする
# コピーしたrequirements.txtを使ってパッケージをインストールする
ADD requirements.txt /code/
RUN pip install -r requirements.txt

# ホストPCの各種ファイルをcodeディレクトリにコピーする
ADD . /code/

RUN apt-get update
RUN apt-get install -y vim
RUN apt-get install -y mecab libmecab-dev mecab-ipadic mecab-ipadic-utf8
