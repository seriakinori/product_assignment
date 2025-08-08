import pickle
import torch
import torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# pklファイルから前処理済みデータを読み込む
PKL_FILE_PATH = '/content/drive/MyDrive/deeplearning/processed_data_maxlen512.pkl'
with open(PKL_FILE_PATH, 'rb') as f: # 'rb'：バイナリ読み込みモード
    data = pickle.load(f)

# dataは辞書型オブジェクトとして保存
padded_ids = data['padded_ids'] # padding済みの単語IDシーケンス
labels = data['labels'] 

# データを訓練・検証・テスト用に分割
# ※この時点ではまだPythonのリスト型として扱う

# 1. 全体を「訓練＋検証用」（90%）と「テスト用」（10%）に分割
# test_size=0.1: 全体の10％をテストデータとする
# random_state=42: 乱数を固定し、何度実行しても同じ分割結果とする（再現性の確保）
# stratify=labels: 元のデータのラベル比率を保ったまま分割する
train_val_ids, test_ids, train_val_labels, test_labels = train_test_split(
    padded_ids, labels, test_size=0.1, random_state=42, stratify=labels
)

# 2. 次に「訓練＋検証用」のデータを「訓訓用（80%）」と「検証用（10％）」に分割
# train_valデータは全体の90%なので、そのうちの1/9を検証用にすると、全体から見て10%分相当となる
# 0.9 * 1/9 = 0.1
train_ids, val_ids, train_labels, val_labels = train_test_split(
    train_val_ids, train_val_labels, test_size=(1/9), random_state=42, stratify=train_val_labels
)

# PyTorchのDatasetクラスの定義
# Datasetクラス：データとそのラベルと保持し、インデックスを指定して１つずつ取り出すための器
class LivedoorTensorDataset(Dataset):
    def __init__(self, ids, labels):
        # コンストではｈクタ、IDのリストとラベルのリストをインスタンス変数として保持
        self.ids = ids
        self.labels = labels
    
    def __len__(self):
        # データセット全体のサンプル数を取得
        return len(self.labels)
    
    def __getitem__(self, index):
        # 指定されたindexのデータを取得。DataLoaderから内部的に呼び出される
        # ここでPythonのリスト型からPyTorchが扱えるTensor型へと変換している
        # dtype=torch.long：整数ID（単語IDやラベルIDを扱う際の標準的なデータ型）
        input_ids = torch.tensor(self.ids[index], dtype=torch.long)
        label_id  = torch.tensor(self.labels[index], dtype=torch.long)
        return input_ids, label_id

# 各データ分割に対応するDatasetインスタンスの作成
# 上記で定義したDatasetクラスを使い、訓練・検証・テストそれぞれのデータセットオブジェクトを作成
train_dataset = LivedoorTensorDataset(train_ids, train_labels)
val_dataset   = LivedoorTensorDataset(val_ids,   val_labels)
test_dataset  = LivedoorTensorDataset(test_ids,  test_labels)

# DataLoaderの作成
# DataLoaderはDatasetをラップし、ミニバッチ処理、データのシャッフル、並列読み込みなどを効率的に行う
BATCH_SIZE = 16 # 一度にモデルを投入するデータ数（バッチサイズ）

# 訓練用DataLoader：データをシャッフルすることで、学習の偏りを防ぎ、モデルの汎用性を向上させる
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# 検証用とテスト用DataLoader：評価時はデータの順序を維持するため、シャッフルは不要
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# モデルのハイパーパラメータ
# データから自動的に設定
vocab_size = len(data['word_to_id']) # 語彙数。単語とIDの対応辞書のサイズから取得
num_classes = len(set(labels)) # 分類クラス数。ラベルの種類数をset()で重複を除いてカウント

# 人間が設定
embedding_dim = 256 # 単語をベクトル表現に変換した際のベクトルの次元数
hidden_dim = 256 # CNNやGRUの中間層（隠れ層）の次元数。モデルの表現力に影響

# 設定したパラメータを画面に出力して確認
print(f"語彙数 (vocab_size): {vocab_size}")
print(f"分類クラス数 (num_classes): {num_classes}")
print(f"単語ベクトルの次元数 (embedding_dim): {embedding_dim}")
print(f"隠れ層の次元数 (hidden_dim): {hidden_dim}")