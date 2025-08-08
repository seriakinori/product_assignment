# 課題：ライブドアニュース分類モデルの構築（Janome版）
## 1. 最終的なアプローチ
ストーリー仕立てのレポート構成:

まず、ベースラインとしてCNN+GRUモデルを構築・評価する。

次に、CNNの長距離依存関係の学習能力の限界を克服するため、CNNをSelf-Attentionに置き換えた改善モデルを構築する。

両者の結果を比較・考察し、改善の有効性を示す。

## 2. 採用技術
フレームワーク: PyTorch

日本語解析: Janome

RNN層: GRU (LSTMの代替)

## 3. 実装ワークフロー
### 前処理 (preprocess.py):
生のテキストデータを読み込み、提供いただいた**Janomeベースのtokenize_ja関数**で分かち書きを行う。

全データから単語辞書(word_to_id)を作成し、全文章とカテゴリラベルを数値IDに変換する。

重い前処理を毎回繰り返さないよう、数値化済みのデータと辞書を1つの.pklファイルに保存する。

### データ準備 (train.py):
.pklファイルを読み込む。

Pythonリストのまま、scikit-learnのtrain_test_splitを使い、層化サンプリング(stratify)で訓練・検証・テスト用に分割する。

PyTorchのDatasetクラスの__getitem__メソッド内で、データを1件ずつTensor型に変換する。これによりメモリオーバーを防ぐ。

DataLoaderでミニバッチを作成する。

### モデル実装:
Embedding → Conv1d → GRU → Linear という構成でPyTorchのnn.Moduleを継承したクラスを定義する。

### 学習・検証ループ:
model.train()とmodel.eval()を切り替える。

検証ループはwith torch.no_grad():で囲み、計算を高速化する。

検証データの損失が改善した場合のみ、モデルの重み(model.state_dict())を.pthファイルとして保存する。

### テスト・評価:
保存した最良のモデルを読み込む。

テストデータで予測を行い、scikit-learnを使って正解率、適合率・再現率・F1スコア (classification_report)、混同行列 (confusion_matrix)を計算する。

これらの評価結果をテキストファイルや画像ファイルとして保存する。