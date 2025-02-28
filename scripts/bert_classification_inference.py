import glob
import os
import json
import re
import demoji
import torch
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import seaborn as sns
import textwrap
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertJapaneseTokenizer, BertForSequenceClassification

class TextClassificationPipeline:
    def __init__(self, data_path, model_name, checkpoint_path, number, batch_size=8, train_size=7000):
        self.number = number
        self.data_path = data_path
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.batch_size = batch_size
        self.train_size = train_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.font_path = 'C:/Windows/Fonts/msgothic.ttc'
        self.font_prop = fm.FontProperties(fname=self.font_path)
        
        self._check_gpu()
        self.category_list = self._get_categories()
        self._create_annotation_json()
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(self.model_name, mecab_kwargs={"mecab_dic": "unidic"})
        
        self.dataset = LivedoorDataset(self.tokenizer, self.data_path)
        self.train_dataloader, self.val_dataloader = self._create_dataloaders()
        
        self.model = self._load_model()
        
    def _check_gpu(self):
        print("GPU is ready" if torch.cuda.is_available() else "GPU is not available")
    
    def _get_categories(self):
        text_dir = os.listdir(self.data_path)
        categories = [f for f in text_dir if os.path.isdir(os.path.join(self.data_path, f))]
        # print("Categories:", categories)
        return categories
    
    def _create_annotation_json(self):
        id_category_list = [{"id": idx, "category": category} for idx, category in enumerate(self.category_list)]
        annotations_list = []
        
        for item in id_category_list:
            file_list = glob.glob(f'{self.data_path}/{item["category"]}/{item["category"]}*.txt')
            for file in file_list:
                annotations_list.append({
                    "file_name": os.path.basename(file),
                    "label": item["id"],
                    "category_name": item["category"]
                })
        
        json_dict = {"category": id_category_list, "annotations": annotations_list}
        json_save_path = os.path.join(self.data_path, "dataset.json")
        with open(json_save_path, mode="wt", encoding="utf-8") as f:
            json.dump(json_dict, f, indent=4)
        print("Dataset JSON saved.")
    
    def _create_dataloaders(self):
        train_dataset, val_dataset = random_split(self.dataset, [self.train_size, len(self.dataset) - self.train_size])
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True), DataLoader(val_dataset, batch_size=self.batch_size)
    
    def _load_model(self):
        model = BertForSequenceClassification.from_pretrained(self.model_name, num_labels=len(self.category_list))
        model.load_state_dict(torch.load(self.checkpoint_path))
        model.to(self.device)
        return model
    
    def evaluate(self):
        self.model.eval()
        labels_list, outputs_list = [], []
        with torch.no_grad():
            for batch in self.val_dataloader:
                batch = {key: value.to(self.device) for key, value in batch.items()}
                labels_list.extend(batch["labels"].cpu().numpy())
                output = self.model(**batch).logits.argmax(dim=1).cpu().numpy()
                outputs_list.extend(output)
        accuracy = np.mean(np.array(outputs_list) == np.array(labels_list)) * 100
        print(f"Accuracy: {round(accuracy, 1)}% ({sum(np.array(outputs_list) == np.array(labels_list))}/{len(outputs_list)})")
    
    def analyze_attention(self):
        sample_data = next(iter(self.val_dataloader))
        sample_data = {key: value.to(self.device) for key, value in sample_data.items()}
        
        labels = sample_data["labels"]
        output = self.model(**sample_data, output_attentions=True)
        pred = output.logits.argmax(dim=1)
        attentions = output.attentions #全てのtransformerのAttention層の重み行列を取得
        
        print(pred, labels)
        print(type(attentions), len(attentions))
        print(type(attentions[-1]), attentions[-1].shape)
        
        # 最も重要度の高い単語5つを赤、次の5つを青に色付け
        print("最終層のAttention")
        print("-------------------------------------------------------------------------------------")
        for batch_number in range(len([0])):
            # 最終層の[CLS]のAttentionの重みを取得
            all_attens = attentions[-1][batch_number, :, 0, :].sum(dim=0)
            input_ids_index_list = all_attens.topk(30).indices
            text = self.tokenizer.convert_ids_to_tokens(sample_data["input_ids"][batch_number])
            
            for i, input_ids_index in enumerate(input_ids_index_list):
                if i < 15:
                    text[input_ids_index] = f'\033[31m{text[input_ids_index]}\033[0m'  # 赤色
                else:
                    text[input_ids_index] = f'\033[34m{text[input_ids_index]}\033[0m'  # 青色
            
            s_wrap_list = textwrap.wrap(''.join(text), 100)
            print(f"Category: {self.category_list[labels[batch_number]]}")
            print('\n'.join(s_wrap_list), "\n")


    def attention_heatmap(self):
        sample_data = next(iter(self.val_dataloader))
        sample_data = {key: value.to(self.device) for key, value in sample_data.items()}
        
        labels = sample_data["labels"]
        output = self.model(**sample_data, output_attentions=True)
        pred = output.logits.argmax(dim=1)
        attentions = output.attentions  # 全ての Transformer の Attention 層の重み行列を取得
        
        print(pred, labels)
        print(type(attentions), len(attentions))
        print(type(attentions[-1]), attentions[-1].shape)
        
        print("最終層のAttention ヒートマップ")
        print("-------------------------------------------------------------------------------------")
        i=0
        for batch_number in range(len(labels)):
            if self.category_list[labels[batch_number]] in ['it-life-hack', 'kaden-channel', 'movie-enter', 'smax', 'sports-watch', 'topic-news']:
                # 最終層の[CLS]トークン（位置 0）の Attention の重みを取得
                all_attens = attentions[-1][batch_number, :, 0, :].sum(dim=0).detach().cpu().numpy()  # detach() を追加
                
                # トークンのリストを取得
                tokens = self.tokenizer.convert_ids_to_tokens(sample_data["input_ids"][batch_number])
                

                # Attention の高い上位 30 トークンのインデックスを取得
                top_indices = np.argsort(all_attens)[-30:]  # 上位 30 個を取得
                top_tokens = [tokens[i] for i in top_indices]  # 上位 30 トークン


                # ヒートマップの描画
                plt.figure(figsize=(16, 1))  # 高さを小さめに
                sns.heatmap([all_attens], cmap="Reds", annot=False, xticklabels=tokens, yticklabels=['CLS Attention'])
                # plt.title(fontproperties=self.font_prop)
                # plt.xlabel(fontproperties=self.font_prop)
                # plt.ylabel(fontproperties=self.font_prop)
                plt.xticks(fontproperties=self.font_prop)
                plt.yticks(fontproperties=self.font_prop)
                plt.xticks(ticks=top_indices, labels=top_tokens, fontsize=6, rotation=90)  # 上位 30 トークンのみ表示
                # plt.show()
                # 保存
                save_path = os.path.join('../pics', f"attention{self.number}_{i}.png")
                plt.savefig(save_path, bbox_inches="tight", dpi=300)  # PNGで保存
                plt.close()  # メモリを解放
                i += 1
                print(f"Attention ヒートマップを {save_path} に保存しました！")




    def save_attention_as_html(self, filename):
        sample_data = next(iter(self.val_dataloader))
        sample_data = {key: value.to(self.device) for key, value in sample_data.items()}
        
        labels = sample_data["labels"]
        output = self.model(**sample_data, output_attentions=True)
        attentions = output.attentions  # 全ての Transformer の Attention 層の重み行列を取得
        
        html_content = """<html><head><meta charset="utf-8"><style>
        span { font-size: 16px; padding: 2px; }
        .red { background-color: #ff9999; }
        .blue { background-color: #9999ff; }
        .gray { color: gray; }
        </style></head><body>
        """
        
        for batch_number in range(len(labels)):
            if self.category_list[labels[batch_number]] in ['it-life-hack', 'kaden-channel', 'movie-enter', 'smax', 'sports-watch', 'topic-news']:

                # 最終層の [CLS] トークンの Attention 重みを取得
                all_attens = attentions[-1][batch_number, :, 0, :].sum(dim=0).detach().cpu().numpy()
                
                # トークンのリストを取得
                tokens = self.tokenizer.convert_ids_to_tokens(sample_data["input_ids"][batch_number])
                
                # 上位 10 トークンのインデックスを取得（5個を赤、次の5個を青）
                top_indices = np.argsort(all_attens)[-30:]
                red_indices = top_indices[-15:]  # 最も重要な5単語（赤）
                blue_indices = top_indices[:15]  # 次に重要な5単語（青）
                
                # HTML 生成
                html_content += f"<p><strong>Category: {self.category_list[labels[batch_number]]}</strong></p><p>"
                
                for i, token in enumerate(tokens):
                    token_html = token.replace("▁", " ")  # BPEのアンダースコアをスペースに
                    if i in red_indices:
                        html_content += f'<span class="red">{token_html}</span> '
                    elif i in blue_indices:
                        html_content += f'<span class="blue">{token_html}</span> '
                    else:
                        html_content += f'<span class="gray">{token_html}</span> '
                
                html_content += "</p><hr>"
        
        html_content += "</body></html>"
        
        # ファイルに保存
        with open(filename, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        print(f"Attention 出力を {filename} に保存しました！")

















class LivedoorDataset(Dataset):
    def __init__(self, tokenizer, text_dir):
        self.text_dir = text_dir
        self.tokenizer = tokenizer
        self._load_json()
    
    def __len__(self):
        return len(self.annotations_list)
    
    def __getitem__(self, idx):
        text = self._get_text(self.annotations_list[idx]["category_name"], self.annotations_list[idx]["file_name"])
        encoding = self.tokenizer(text, return_tensors="pt", max_length=512, padding="max_length", truncation=True)
        encoding = {key: torch.squeeze(value) for key, value in encoding.items()}
        encoding["labels"] = torch.tensor(self.annotations_list[idx]["label"], dtype=torch.long)
        return encoding
    
    def _load_json(self):
        with open(os.path.join(self.text_dir, 'dataset.json'), encoding='utf-8') as f:
            self.text_json = json.load(f)
        self.annotations_list = self.text_json["annotations"]
    
    def _get_text(self, category_name, file_name):
        file_path = os.path.join(self.text_dir, category_name, file_name)
        with open(file_path, encoding='utf-8') as f:
            lines = f.read().splitlines()
        text = '\n'.join(lines[3:])
        return self._text_preprocess(text)
    
    def _text_preprocess(self, text):
        text = text.translate(str.maketrans({'\n': '', '\t': '', '\r': '', '\u3000': ''}))
        text = re.sub(r'https?://\S+', '', text)
        text = demoji.replace(text, '')
        text = re.sub(r'\d+', '0', text)
        text = text.lower()
        
        for target in ['関連記事', '関連サイト', '関連リンク']:
            idx = text.find(target)
            if idx != -1:
                text = text[:idx]
        return text




if __name__ == "__main__":
    number = 5 # 保存するファイル名につける番号
    pipeline = TextClassificationPipeline("dnn3_ws/text", "cl-tohoku/bert-base-japanese-whole-word-masking", "dnn3_ws/checkpoints/model.pth", number)
    pipeline.evaluate()
    # pipeline.analyze_attention() # Attentionをbashで表示
    pipeline.attention_heatmap() # Attentionをヒートマップで表示
    pipeline.save_attention_as_html(f"dnn3_ws/datas/attention_output{number}.html") # AttentionをHTMLで保存




