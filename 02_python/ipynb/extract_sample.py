# 保存した両モデルの予測結果をロード
cnn_preds = np.load(PRJ_ROOT + 'cnn_predictions.npy').tolist()
attention_preds = np.load(PRJ_ROOT + 'attention_predictions.npy').tolist()

# pklファイルから前処理済みデータと辞書をロード
with open(DATA_DIR + 'processed_data_maxlen512.pkl','rb') as f:
    data = pickle.load(f)
id_to_label = data['id_to_label']

# テストデータの「原文」と「正解ラベル」を再取得
from sklearn.model_selection import train_test_split
raw_df = pd.read_csv(PRJ_ROOT + 'livedoor_news_corpus.csv') # 元の生データCSV

# livedoor_news_corpus.csv テキストとラベルを取得
X_raw = raw_df['text']
y_raw = raw_df['label'].map( { label: i for i, label in enumerate(raw_df['label'].unique()) } ) # ラベルを数値化

# 分割してテストデータを取得
_, X_test_raw, _, test_labels = train_test_split(
    X_raw, y_raw, test_size=0.1, random_state=42, stratify=y_raw
)

test_texts = X_test_raw.tolist()

# テストデータの原文、正解ラベル、両モデルの予測結果をまとめたDataFrameを作成
results_df = pd.DataFrame({
    'text': test_texts,
    'true_label': test_labels,
    'cnn_pred': cnn_preds,
    'attention_pred': attention_preds
})

# IDをカテゴリ名に変換
results_df['true_label_name'] = results_df['true_label'].map(id_to_label)
results_df['cnn_pred_name'] = results_df['cnn_pred'].map(id_to_label)
results_df['attention_pred_name'] = results_df['attention_pred'].map(id_to_label)

# 抽出条件
# ベースラインモデルでは誤分類したが、改善モデルでは正しく分類したもの
condition1 = results_df['cnn_pred'] != results_df['true_label']
condition2 = results_df['attention_pred'] == results_df['true_label']

improved_examples = results_df[condition1 & condition2]

# 結果の表示
print("【改善事例の抽出結果】")

# 特に 'livedoor-homme' の事例を見てみる
homme_improved = improved_examples[improved_examples['true_label_name'] == 'livedoor-homme']

for index, row in homme_improved.head(3).iterrows():
    print("="*50)
    print(f"【改善事例】 正解カテゴリ: {row['true_label_name']}")
    print(f"  - CNNの予測 (間違い): {row['cnn_pred_name']}")
    print(f"  - Attentionの予測 (正解): {row['attention_pred_name']}")
    print("\n【記事本文 (冒頭部分)】")
    print(row['text'][:200] + "...")
    print("="*50)

