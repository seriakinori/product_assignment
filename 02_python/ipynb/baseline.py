import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

""" ベースラインモデル """
class CNN_GRU_Model(nn.Module):
    """
    CNNで系列データから局所的な特徴を抽出します。
    GRUでその特徴の時間的な依存関係を学習します。
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        """
        コンストラクタ

        Args:
            vocab_size (int): 語彙数。入力される単語の種類の数。
            embedding_dim (int): 単語埋め込みベクトルの次元数。
            hidden_dim (int): CNNの出力チャネル数。GRUの隠れ状態の次元数。
            num_classes (int): 出力クラス数。分類したいカテゴリの数。
        """

        # 1. Embedding層
        # 単語IDの系列を単語埋め込みに変換
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # 2. 1次元CNN層（Convolutional Neural Network）
        # 埋め込みベクトルの系列から、n-gramのような局所的な特徴を抽出
        # in_channels: 入力チャネル数（埋め込み次元数）
        # out_channels: 出力チャネル数（抽出される特徴マップの数 = hidden_dim）
        # kernel_size=3: ３つの連続する単語ベクトルを一度に見る（3-gramに相当）
        # padding=1: 畳み込み後も系列長を変化させないためにpaddingを追加
        self.cnn = nn.Conv1d(in_channels=embedding_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU() # 活性化関数

        # 3. GRU（Gated Recurrent Unit）
        # CNNが抽出した特徴系列を入力とし、時間的な依存関係を学習
        # input_size: 各タイムステップにおける入力特徴の次元数（＝CNNの出力チャネル数）
        # hidden_size: GRUの隠れ状態ベクトルの次元数
        # batch_size=True: 入力テンソルの形状を（batch_size, seq_len, input_size）として扱う
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, batch_size=True)

        # 4. 全結合層（Fully Connected Layer）
        # GRUから得られた最終的な文脈ベクトルを、指定されたクラス数にマッピングし、分類スコアを出力
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        順伝播

        Args: 
            x (torch.Tensor): 入力データ。単語IDの系列
                            　形状：（batch_size, seq_len）
        Returns:
            torch.Tensor: 各クラスに対する分類スコア
                        　形状：（batch_size, num_clases）
        """
        # 1. Embedding
        # x の形状：（batch_size, seq_len）
        x = self.embedding(x)   # -> (batch_size, seq_len, embedding_len)

        # 2. CNN
        # Conv1dは 入力として(batch, channels, seq_len) の形状を受け付けるため、埋め込みベクトルの次元を入れ替える
        # (batch_size, seq_len, embedding_dim) -> (batch_size, embedding_dim, seq_len)
        #       0         1            2                0            2           1 
        x = x.permute(0, 2, 1)

        # 畳み込み層とのの活性化関数を適用
        x = self.cnn(x)         # -> (batch_size, hidden_dim, seq_len)
        x = self.relu(x)

        # 3. GRU
        # GRUの入力（batch_first=True）は（batch, seq, feature）の形状を受け付けるため、再度次元を入れ替える
        # (batch_size, hidden_size, seq_len) -> (batch_size, seq_len, hidden_dim)
        #       0         1            2             0          2          1 
        x = x.permute(0, 2, 1)

        # GRUは全てのタイムステップの出力（output）と最期のタイムステップの隠れ状態（h_n）を返す
        # ここでは文脈全体を返す最後の隠れ状態のみ使用
        _, h_n = self.gru(x)    # h_nの形状：(num_layers, batch_size, hidden_dim) 

        # 4. 全結合層
        # GRuの最後の隠れ状態の次元（num_layer）を解除し、（batch_size, hidden_dim）の形状に整形
        x = h_n.squeeze(0)      # -> (batch_size, hidden_dim)

        # 全結合層に入力し、最終的なクラススコアを得ます
        out = self.fc(x)        # -> (batch_size, num_classes)

        return out


# 学習
# 実行環境（GPU／CPU）の確認と設定
# GPUが利用可能か確認し、利用可能なら "cuda"、そうでなければ "cpu" をdeviceに設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# CNN_GRU_Modelのインスタンスを生成し、.to(device)で指定したデバイスにモデルを転送
# モデルのパラメータ（重み）や計算が、指定したデバイスで実施されるようにする
model = CNN_GRU_Model(vocab_size, embedding_dim, hidden_dim, num_classes)
model.to(device)

# 最適化手法と損失関数の定義
# 最適化アルゴリズムとしてAdamを使用
# model.parameters() を渡すことで、モデルが持つすべてのパラメータを最適化の対象として登録
# lrは学習率（learning rate）で、パラメータ更新のステップ幅を決定
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 損失関数としてクロスエントロピー損失を定義
criterion = nn.CrossEntropyLoss()

# モデルの保存設定と学習ループの準備
# 才能の検証損失を保存する変数を初期化することによって、検証データの損失が改善した場合にのみモデルを保存する
# 早期終了（Early Stopping）に近い戦略
best_valid_loss = np.inf
MODEL_SAVE_PATH = DATA_PATH + 'baseline_model_best_model.pth' # 最良モデルの保存先ファイルパス

n_epoch = 10 # 学習エポック数
for epoch in range(n_epoch):
    # 各エポックの開始時に、訓練データと検証データの損失と正解数をリセット
    loss_train = 0
    loss_valid = 0
    accuracy_train = 0
    accuracy_valid = 0

    # 訓練モード
    # model.train()を呼び出し、モデルを「訓練モード」に切り替える
    # Dropout層やBatchNorm層などが訓練時の挙動となる
    model.train()
    for x,t in train_loader: # DataLoaderからミニバッチを1件ずつ取り出す
        # データをモデルと同じデバイスに転送
        x = x.to(device)
        t = t.to(device)

        # 前のバッチの勾配が累積しないように勾配をリセットする。
        optimizer.zero_grad()

        # 順伝播
        output = model(x)

        # 損失計算：モデルの出力と正解ラベルとを比較し、損失を計算
        loss = criterion(output, t)
        # 逆伝播：損失に基づいて、各パラメータの勾配を計算
        loss.backward()

        # パラメータ更新：計算された勾配を基に、オプティマイザがモデルのパラメータを更新
        optimizer.step()

        # 予測ラベルを計算。出力が最も大きいクラスを予測結果とするる
        pred = output.argmax(dim=1)

        # このバッチでの損失と正解数を、エポック全体の集計に加算
        # .item()はテンソルからPythonのスカラー値を取り出す
        loss_train += loss.item()
        accuracy_train += torch.sum(pred == t.data)
    
    # 検証モード
    # model.eval()を呼び出し、モデルを「検証モード」に切り替える
    model.eval()

    # 検証ではパラメータ更新はおこなわず、メモリ消費量を削減し、計算速度は向上する
    with torch.no_grad():
        for x,t in val_loader:
            # データをデバイスに転送
            x = x.to(device)
            t = t.to(device)

            # 順伝播
            output = model(x)

            # 損失計算
            loss = criterion(output, t)
            
            # 予測ラベルを計算
            pred = output.argmax(dim=1)

            # 損失と正解数を加算
            loss_valid += loss.item()
            # モデルの予測と正解ラベルとが等しいものの個数をカウントして足し合わせる
            accuracy_valid += torch.sum(pred == t.data)
    
    # エポックごとの結果を計算して表示する
    avg_loss_train = loss_train / len(train_loader)
    avg_loss_valid = loss_valid / len(val_loader)
    avg_acc_train = accuracy_train / len(train_dataset)
    avg_acc_valid = accuracy_valid / len(val_dataset)

    # 結果を整形して表示
    print(
        f"| epoch {epoch+1:2d} | train loss {avg_loss_train:.4f}, acc {avg_acc_train:.4f} "
        f"| valid loss {avg_loss_valid:.4f}, acc {avg_acc_valid:.4f}"
    )

    # 最良モデルの保存
    # 検証データの損失が、これまでの最小値を更新した場合
    if avg_loss_valid < best_valid_loss:
        print(f"Validation loss improved ({best_valid_loss:.4f} --> {avg_loss_valid:.4f}). Saving model...")
        best_valid_loss = avg_loss_valid # 最小損失を更新

        # モデルのパラメータ（重み）のみを保存
        torch.ave(model.state_dict(), MODEL_SAVE_PATH)

print(f"\nTraining finished. Best model saved to {MODEL_SAVE_PATH}")
    


import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# モデルの評価と結果の可視化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(f" '{DATA_PATH}' を読み込んでいます...")
with open(DATA_PATH, 'rb') as f:
    data = pickle.load(f)

# pklファイル内のデータから、IDとカテゴリ名の対応辞書を再取得
# 評価レポートやグラフのラベルとして使用
id_to_label = data['id_to_label']

# 評価レポート用に、カテゴリ名のリストもここから作成
category_names = list(id_to_label.values())

# 全ての予測結果と正解ラベルを格納するための空のリストを準備
all_preds = []
all_labels = []

# モデルを「評価モード」に切り替える
model.eval()

#勾配計算を無効にして計算リソースを節約
with torch.no_grad():
    # テストデータ用のDataLoaderからミニバッチを１件ずつ取り出す
    for x,t in test_loader:
        # データをモデルと同じデバイにに転送
        x = x.to(device)
        t = t.to(device)

        # モデルにデータを入力し、順伝播させる
        output = model(x)

        # 最もスコアの高いクラスのインデックスを予測結果とする
        pred = output.argmax(dim=1)

        # 予測結果と正解ラベルをリストに追加
        #　scikit-learnで計算するために、GPUのTensorからCPU上のリストに変換する
        # .cpu()でCPUに転送し、.tolist()でPythonのリストに変換
        all_preds.extend( pred.cpu().tolist() )
        all_labels.extend( t.cpu().tolist() )

# 正解率（Accuracy）の計算
accuracy = accuracy_score(all_labels, all_preds)
print(f"Accuracy: {accuracy:.4f}")

# 適合率（Precision）、再現率（Recall）、F1スコア（F1-Score）を含むレポートを表示
print("\nClassification Report:")
report = classification_report(all_labels, all_preds, target_names=category_names)

# 結果をテキストファイルに保存
with open(DATA_DIR + "CNN_GRU_test_results.txt","w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write("\nClassification Report:\n")
    f.write(report)

# 混合行列（Confusion Matrix）の計算と可視化
cm = confusion_matrix(all_labels, all_preds)

# numpy配列のままだと軸が数字で見づらいため、PandasのDataFrameに変換
# indexとcolumnsにカテゴリ名を設定
cm_df = pd.DataFrame(cm, index=category_names, columns=category_names)

# 可視化
plt.figure(figsize=(10,8)) # グラフのサイズを指定
# seabornのheatmap関数を使って、混合行列をヒートマップとして描画
# annot=True：各セルに数値を表示
# fmt='d': 数値を整数で表示
# cmap='Greens': 色のテーマを指定
sns.heatmap(cm_df, annot=True, cmap='Greens')
plt.title('Confusion Matrix') # グラフのタイトル
plt.ylabel('True Label') # Y軸のラベル
plt.xlabel('Predicted Label') # X軸のラベル

plt.tight_layout() # ラベルがグラフ領域からはみ出ないように自動調整

# 混合行列を画像ファイルとして保存
plt.savefig(DATA_PATH + "CNN_GRU_confusion_matrix.png")

plt.show()


