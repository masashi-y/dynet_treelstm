# TreeLSTM

## ビルド

### Dynetのビルド

__準備(もし入ってなければ)__  
ubuntuなど:
```sh
sudo apt-get install build-essential cmake mercurial
```

macos:
```sh
xcode-select --install
brew install cmake hg  # Using homebrew.
sudo port install cmake mercurial # Using macports.
```

Dynet本体のビルド
```sh
# 適当なところで
$ wget https://cl.naist.jp/~masashi-y/resources/dynet.zip
$ unzip dynet.zip; cd dynet
$ hg clone https://bitbucket.org/eigen/eigen/ -r 2355b22
$ mkdir build; cd build
$ cmake .. -DEIGEN3_INCLUDE_DIR=../eigen
$ make -j 4
```

### haskell-stackのインストール
```sh
$ curl -sSL https://get.haskellstack.org/ | sh
$ (もしくは) brew install haskell-stack
```

### treelstmとdynet.hsのビルド(時間かかります)
```sh
$ git clone https://github.com/masashi-y/dynet_treelstm.git
$ DYNET=(dynetへのパス) EIGEN3_INCLUDE_DIR=(Eigenへのパス) stack build
# (ホームディレクトリに入れた場合) DYNET=$HOME/dynet/ EIGEN3_INCLUDE_DIR=$HOME/dynet/eigen stack build
```
## データ(SNLI)の解凍
```sh
$ cd data
$ unzip snli_1.0.zip
```

## treelstmの実行
```sh
$ find .stack-work -name "libdynetc.so" # (すこし辛い作業)
.stack-work//downloaded/rDD2a0f1la0a/c
# macなら
export DYLD_LIBRARY_PATH=.stack-work//downloaded/rDD2a0f1la0a/c
# linux
export LD_LIBRARY_PATH=.stack-work/downloaded/rDD2a0f1la0a/c
```

モデル: https://cl.naist.jp/~masashi-y/resources/rte.model
```sh
# 学習
$ stack exec train data/snli_1.0/snli_1.0_train.txt data/snli_1.0/snli_1.0_dev.txt
# 予測
$ stack exec predict rte.model data/snli_1.0/snli_1.0_train.txt data/snli_1.0/snli_1.0_dev.txt
```
