import torch.nn.functional as F
from transformers import BertTokenizer, BertModel

# BERTのトークナイザーとモデルの準備 インストール
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
bert_model = BertModel.from_pretrained(model_name)

print(bert_model.config)
print(bert_model)

text_list = ["I'm reading a book",
             "I'd like to book a room",
             "I'd like to reserve a room",
             "Please reserve this table for us"]


encoding = tokenizer(text_list, max_length=10, padding="max_length", truncation=True, return_tensors="pt")
print(encoding)

#隠れ層も取得
output = bert_model(**encoding, output_hidden_states=True)

len(output[2])
# output[2][層][文][単語]

print(text_list)


#多義語
print("多義語")
print(text_list[0], ",", text_list[1])
layer0_similarity = F.cosine_similarity(output[2][0][0][6].reshape(1, -1), output[2][0][1][6].reshape(1, -1))
layer12_similarity = F.cosine_similarity(output[2][-1][0][6].reshape(1, -1), output[2][-1][1][6].reshape(1, -1))
print(f"0層目: {round(layer0_similarity.item(), 3)}")
print(f"最終層目: {round(layer12_similarity.item(), 3)}")


# 類義語(同じ文脈)
print("類義語(同じ文脈)")
print(text_list[1], ",", text_list[2])
layer0_similarity = F.cosine_similarity(output[2][0][1][6].reshape(1, -1), output[2][0][2][6].reshape(1, -1))
layer12_similarity = F.cosine_similarity(output[2][-1][1][6].reshape(1, -1), output[2][-1][2][6].reshape(1, -1))
print(f"0層目: {round(layer0_similarity.item(), 3)}")
print(f"最終層目: {round(layer12_similarity.item(), 3)}")


# 類義語(違う文脈)
print("類義語(違う文脈)")
print(text_list[1], ",", text_list[3])
layer0_similarity = F.cosine_similarity(output[2][0][1][6].reshape(1, -1), output[2][0][3][2].reshape(1, -1))
layer12_similarity = F.cosine_similarity(output[2][-1][1][6].reshape(1, -1), output[2][-1][3][2].reshape(1, -1))
print(f"0層目: {round(layer0_similarity.item(), 3)}")
print(f"最終層目: {round(layer12_similarity.item(), 3)}")