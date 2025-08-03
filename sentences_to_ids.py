import pandas as pd
import pickle
from collections import Counter
# GiNZAなどのTokenizerは準備済みと仮定

# 1. 生データを読み込み、分かち書き
df = pd.read_csv('livedoor_news_corpus.csv')

from janome.tokenizer import Tokenizer

wakati_function = Tokenizer()

df['tokens'] = df['text'].apply(wakati_function) # 分かち書き関数

# 2. 語彙（word_to_id辞書）を構築
word_counter = Counter(word for tokens in df['tokens'] for word in tokens)
word_to_id = {word: i+2 for i, (word, count) in enumerate(word_counter.items())} # 0と1は特殊トークン用
word_to_id['<PAD>'] = 0
word_to_id['<UNK>'] = 1

# 3. テキストとラベルをIDのシーケンスに変換
df['input_ids'] = df['tokens'].apply(lambda tokens: [word_to_id.get(w, word_to_id['<UNK>']) for w in tokens])
# ラベルのID化...

# 4. 必要なデータをファイルに保存
processed_data = {
    'input_ids': df['input_ids'].tolist(),
    'labels': df['label_id'].tolist(),
    'word_to_id': word_to_id,
    # 他にも必要なデータがあれば...
}

with open('processed_data.pkl', 'wb') as f:
    pickle.dump(processed_data, f)

print("前処理とデータの保存が完了しました。")