# deep-learning-from-zero

## 環境構築
1. requirements.txtに依存パッケージを書く
2. 以下のコマンドを実行
```
docker build -t dl-zero .
```

## Pythonスクリプトの実行  
1. 以下のコマンドを実行
```     
docker run -it --rm -it -v $PWD:/work dl-zero python foo/bar.py
```
