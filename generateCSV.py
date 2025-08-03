import os
import pandas as pd
from tqdm.auto import tqdm # 進捗バーを表示

# ライブドアニュースコーパスのテキストファイル群が置かれている親ディレクトリ
# 例: /content/text/
data_dir = 'text'

# カテゴリ名（サブディレクトリ名）のリストを取得
# 不要なファイル（例: LICENSE.txt）は除外
categories = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
print("対象カテゴリ:", categories)

# 最終的にDataFrameにするための、行データ（辞書）を格納するリスト
all_data = []

# tqdmを使って進捗を可視化しながらループ
for category in tqdm(categories, desc="カテゴリ処理中"):
    category_path = os.path.join(data_dir, category)
    
    files = os.listdir(category_path)
    for file_name in files:
        # category内のREADME.mdはスキップ
        if file_name.endswith('.txt'):
            file_path = os.path.join(category_path, file_name)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    # 最初の2行はURLとタイムスタンプなので読み飛ばし、3行目以降を本文とする
                    lines = f.readlines()
                    text_body = "".join(lines[2:]).strip()
                    
                    # ラベル（カテゴリ名）とテキスト本文を辞書としてリストに追加
                    all_data.append({'label': category, 'text': text_body})
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

# ループ完了後、リストから一気にDataFrameを作成
df = pd.DataFrame(all_data)

# 作成したDataFrameをCSVとして保存（インデックスは不要なのでFalse）
df.to_csv('livedoor_news_corpus.csv', index=False, encoding='utf-8-sig')

print("\nCSVファイルの作成が完了しました。")
print("データ件数:", len(df))
print("最初の5件:\n", df.head())