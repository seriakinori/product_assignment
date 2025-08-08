import torch
import torch.nn as nn

class Attention_GRU_Model(nn.Module):
    """
    Multi-Head Self-Attention機構をGRU前段に組み込んだモデル
    Attention層が入力シーケンス内の単語間の関連性を捉え、
    GRUが時系列に処理することで、より高度な特徴抽出を目指す
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, num_heads=8):
        """
        コンストラクタ

        Args:  
            Vocab_size (int): 語彙数
            embedding_dim (int): 単語埋め込みベクトルの次元数。Attention層の入力次元数でもある
            hidden_dim (int): GRUの隠れ状態の次元数
            num_classes (int): 出力クラス数
            num_heads (int): Multi-Head Attentionのヘッド数
        """
        super(Attention_GRU_Model,self).__init__()

        # 1. Embedding層
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # 2. Attention層（Multi-Head Self-Attention）
        # 入力系列内の各単語が、他のどの単語に注目すべきかを学習します。
        # embeded_dim: 入出力の次元数
        # num_heads: 並列計算のためにAttentionの「ヘッド」をいくつに分割するかを設定
        # batch_first=False: 入力の形式を（seq_len, batch_size, dim）に指定します。
        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            batch_first=False
        )

        # 学習を安定させるためのレイヤ正規化
        # Attention層に適用
        self.norm1 = nn.LayerNorm(embedding_dim)

        # 3. GRU層
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True)

        # 4. 全結合層
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        順伝播

        Args:
            x (torch.Tensro): 入力データ。形状：(batch_size, seq_len)
        
        Returns:
            torch.Tensor: 各クラスに対する分類スコア。形状：(batch_size, num_classes)
        """
        # 1. Embedding
        x = self.embedding(x) # -> (batch_size, seq_len, embedding_dim)

        # 2. Self-Attention
        # MultiheadAttentionの入力形式 (seq_len, batch_size, dim)に合わせるため、次元を入れ替える
        x_permuted = x.permute(1, 0, 2) # -> (seq_len, batch_size, embedding_dim)

        # Self-Atttentionでは、Query, Key, Valueにすべて同じ入力（x_permuted）を用いる
        # MultiheadAttentionの順伝播
        attn_output, _ = self.multi_head_attention(query=x_permuted, key=x_permuted, value=x_permuted)

        # 残差接続（residual Connection）とレイヤ正規化
        # 入力 x_permuted をAttentionの出たた足し合わせることで、勾配消失を防ぐ
        x = self.norm1(x_permuted + attn_output) # -> (seq_len, batch_size, embedding_dim)

        # 3. GRU
        # GRUの入力形式（batch_size, seq_len, dim）に戻すため、再度次元を入れ替える
        x_permuted = x.permute(1, 0, 2) # -> (seq_len, batch_size, embedding_dim)

        # GRU層に入力し、最後の隠れ状態 h_n を取得
        _, h_n = self.gru(x) # h_n の形状：（num_layers, batch_size, hidden_dim)

        # 4. 分類
        x = h_n.squeeze(0) # -> (batch_size, hidden_dim)
        # 全結合層で最終的なクラススコアを出力
        out = self.fc(x) # -> (batch_size, num_classes)

        return out

# 学習
# 実行環境（GPU／CPU）の確認と設定
# GPUが利用可能か確認し、利用可能なら "cuda"、そうでなければ "cpu" をdeviceに設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# CNN_Attention_Modelのインスタンスを生成し、.to(device)で指定したデバイスにモデルを転送
# モデルのパラメータ（重み）や計算が、指定したデバイスで実施されるようにする
model = Attention_GRU_Model(vocab_size, embedding_dim, hidden_dim, num_classes)
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
MODEL_SAVE_PATH = DATA_PATH + 'improved_model_best_model.pth' # 最良モデルの保存先ファイルパス

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

####################################################################################################
# 評価
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
        # 
        all_preds.extend( pred.cpu().tolist() )
        all_labels.extend( t.cpu().tolist() )
        '''
          当初は.numpy()でNumPy配列に変換しようとしていたが、RuntimeError: Numpy is not availableが発生した。
          これは、NumPyライブラリが見つからない、または利用できない」ことを示している。
          プリインストールされていたNumPyが他のライブラリとの兼ね合いなどで不安定になったと思われる。
          ランタイムのリセット等を試したが、改善しなかったので、NumPyを一切経由せず、
          Tensorを直接Pythonの標準的なリストに変換する .tolist() メソッドを使用した。
          scikit-learnの評価関数はNumPy配列だけでなく、Pythonのリストも受け付けるため、この方法で問題なく動作。
        ''' 

# 正解率（Accuracy）の計算
accuracy = accuracy_score(all_labels, all_preds)
print(f"Accuracy: {accuracy:.4f}")

# 適合率（Precision）、再現率（Recall）、F1スコア（F1-Score）を含むレポートを表示
print("\nClassification Report:")
report = classification_report(all_labels, all_preds, target_names=category_names)

# 結果をテキストファイルに保存
with open(DATA_DIR + "Attention_GRU_test_results.txt","w") as f:
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
sns.heatmap(cm_df, annot=True, cmap='Blues')
plt.title('Confusion Matrix') # グラフのタイトル
plt.ylabel('True Label') # Y軸のラベル
plt.xlabel('Predicted Label') # X軸のラベル

plt.tight_layout() # ラベルがグラフ領域からはみ出ないように自動調整

# 混合行列を画像ファイルとして保存
plt.savefig(DATA_PATH + "Attention_GRU_confusion_matrix.png")

plt.show()