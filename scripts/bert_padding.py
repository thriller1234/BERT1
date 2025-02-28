import demoji
from transformers import BertJapaneseTokenizer, BertForMaskedLM

demoji.download_codes()  # 絵文字処理のための辞書をダウンロード

model_name = "cl-tohoku/bert-base-japanese-whole-word-masking"
# トークナイザーの準備（unidic を指定）
tokenizer = BertJapaneseTokenizer.from_pretrained(model_name, mecab_kwargs={"mecab_dic": "unidic", "mecab_option": None})
bert_model = BertForMaskedLM.from_pretrained(model_name)

text = "私はお腹が空いたので[MASK]を食べたい"
tokens = tokenizer.tokenize(text)
print(tokens)

encoding = tokenizer(text, max_length=20, padding="max_length", truncation=True, return_tensors="pt")


output = bert_model(**encoding)
print(output[0].shape)
mask_index = encoding["input_ids"][0].tolist().index(4)

max_word = output[0][0][mask_index].argmax().item()
mask_word = tokenizer.convert_ids_to_tokens(max_word)
print(text.replace("[MASK]", mask_word))

top_words = output[0][0][mask_index].topk(5).indices
for word_id in top_words:
  word = tokenizer.convert_ids_to_tokens(word_id.item())
  print(text.replace("[MASK]", word))








