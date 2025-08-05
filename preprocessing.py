import pandas as pd
import pickle
from janome.tokenizer import Tokenizer
from tqdm.auto import tqdm

wakati = Tokenizer()

""" 日本語のトークン化 """
def tokenize_ja(sentences_list):
    wakati_list = []
    print("トークン処理を開始します。")
    for sentence in tqdm(sentences_list):
        # tokenizeから返される表層形を分かち書きリストに登録
        wakati_list.append([item.surface for item in wakati.tokenize(sentence)])
    return wakati_list

""" 単語からIDへの辞書を作成 """
def create_word_id_dict(sentences):
    word_to_id = {}  # 単語からIDへの変換辞書
    id_to_word = {}  # IDから単語への逆引き辞書
    # 0はパディング／未知語用に予約
    word_to_id['<PAD>/<UNK>'] = 0
    id_to_word[0] = '<PAD>/<UNK>'

    # すべての文章をループ  
    for sentence in sentences:
        # 文章内の各単語をループ
        for word in sentence:
            # もし単語がまだ辞書に登録されていなければ、新しいIDを割り振る、
            if word not in word_to_id:
                # 新しいIDとして、現在の辞書のサイズ（登録済みの単語数）を使用する
                tmp_id = len(word_to_id)
                word_to_id[word] = tmp_id
                id_to_word[tmp_id] = word

    # (単語をキー、IDをバリューとする辞書, IDをキー、単語をバリューとする辞書)のタプルを返す
    return word_to_id, id_to_word

""" 単語で構成された文章のリストを対応するIDのリストに変換 """
def convert_sentences_to_ids(sentences, word_to_id):
    sentence_id_list = []
    for sentence in sentences:
        # dict.get(key, default)メソッドによって、未知語でもエラーにならずにデフォルトである<UNK>のIDを返す
        sentence_ids = [word_to_id.get(word, 0) for word in sentence]
        sentence_id_list.append( sentence_ids )

    # IDに変換された文章のリストを返す 
    return sentence_ids

""" IDに変換され文章のリスト"に対して、paddingと打ち切り処理を行う """
def padding_and_truncate_sentence(sentences, max_len=512):
    # 処理が行われた後の文章IDを格納するリスト
    processed_sentences = []

    for sentence in sentences:
        # １．打ち切り
        # 文章の長さが max_lenを超える場合、末尾から max_len分だけを取得します。
        # 文章の末尾に重重要な情報含まれる場合が多いため、前から切り捨てます
        # 改善モデルにおけるMultiheadAttention層が内部で行う計算で、
        # 非常に巨大な行列を作成しようとしてGPUのメモリが足りなくなる問題への対応として実施しました。
        sentence = sentence[-max_len:]

        # ２．padding
        # 文章の長さがmax_lenに満たない場合、差分を計算
        padding_size = max_len - len(sentence)

        # 足りない分だけ <PAD>のIDのリストを作成し、文章の前方に連結
        padding = [0] * padding_size
        processed_sentences.append(padding + sentence)

    return processed_sentences

###

# main
# 生のCSVの読み込み
df = pd.read_csv(DATA_PATH + "livedoor_news_corpus.csv")

# 文章カテゴリ（ラベル）をID化
label_to_id = {label: i for i, label in enumerate(df['label'].unique())}
id_to_label = {i: label for i, label in enumerate(df['label'].unique())}

# テキストの分かち書き
ja_sentences = tokenize_ja(df["text"].tolist())

# 単語辞書の作成
word_to_id, id_to_word = create_word_id_dict(ja_sentences)

# 文章をID列に変換
sentence_ids = convert_sentences_to_ids(ja_sentences, word_to_id)

# padding処理
padded_ids = padding_and_truncate_sentence(sentence_ids)

# データの保存
processed_data = {
    'padded_ids': padded_ids,
    'labels': df['label_id'].tolist(),
    'word_to_id': word_to_id,
    'id_to_word': id_to_word,
    'label_to_id': label_to_id,
    'id_to_label': id_to_label,
}

with open('processed_data_maxlen512.pkl', 'wb') as f:
    pickle.dump(processed_data, f)

print(f"保存ファイル: processed_data_maxlen512.pkl")
print(f"語彙数: {len(word_to_id)}")