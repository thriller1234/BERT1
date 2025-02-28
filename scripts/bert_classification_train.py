import json
import glob
import re
import demoji
import torch
import logging
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertJapaneseTokenizer, BertForSequenceClassification

demoji.download_codes()  # 絵文字処理のための辞書をダウンロード
logging.basicConfig(level=logging.INFO)


class DatasetProcessor:
    """データセットの前処理を行い、JSON ファイルを作成するクラス"""
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.category_list = [d.name for d in self.data_dir.iterdir() if d.is_dir()]
        self.dataset_json = self.data_dir / "dataset.json"

    def create_json(self):
        """カテゴリとアノテーションを JSON に保存"""
        id_category_list = [{"id": idx, "category": cat} for idx, cat in enumerate(self.category_list)]
        annotations_list = []

        for item in id_category_list:
            file_list = glob.glob(str(self.data_dir / item["category"] / f"{item['category']}*.txt"))
            for file in file_list:
                annotations_list.append({
                    "file_name": Path(file).name,
                    "label": item["id"],
                    "category_name": item["category"]
                })

        json_dict = {"category": id_category_list, "annotations": annotations_list}
        with self.dataset_json.open("wt", encoding="utf-8") as f:
            json.dump(json_dict, f, indent=4)

        logging.info(f"JSON file saved: {self.dataset_json}")


class LivedoorDataset(Dataset):
    """データセットを PyTorch の Dataset クラスとして実装"""
    def __init__(self, tokenizer, data_dir):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self._load_json()

    def __len__(self):
        return len(self.annotations_list)

    def __getitem__(self, idx):
        text = self._get_text(self.annotations_list[idx]["category_name"],
                              self.annotations_list[idx]["file_name"])
        encoding = self.tokenizer(text, return_tensors="pt", max_length=512,
                                  padding="max_length", truncation=True)
        encoding = {key: torch.squeeze(value) for key, value in encoding.items()}
        encoding["labels"] = torch.tensor(self.annotations_list[idx]["label"], dtype=torch.long)
        return encoding

    def _load_json(self):
        with open(self.data_dir / "dataset.json", encoding='utf-8') as f:
            self.annotations_list = json.load(f)["annotations"]

    def _get_text(self, category_name, file_name):
        file_path = self.data_dir / category_name / file_name
        with open(file_path, encoding='utf-8') as f:
            text = '\n'.join(f.read().splitlines()[3:])  # 4行目以降を使用
        return self._text_preprocess(text)

    def _text_preprocess(self, text):
        text = text.translate(str.maketrans({'\n': '', '\t': '', '\r': '', '\u3000': ''}))
        text = re.sub(r'https?://\S+', '', text)  # URL削除
        text = demoji.replace(text, '')  # 絵文字削除
        text = re.sub(r'\d+', '0', text)  # 数字を0に置換
        text = text.lower()  # 小文字化
        for target in ['関連記事', '関連サイト', '関連リンク']:
            idx = text.find(target)
            if idx != -1:
                text = text[:idx]
        return text


class Trainer:
    """モデルの学習を管理するクラス"""
    def __init__(self, model_name, dataset, batch_size=8, epochs=10, lr=1e-5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(dataset.category_list)).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.epochs = epochs

        train_size = 7000
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size)

        logging.info(f"Dataset split: Train {len(self.train_dataset)}, Val {len(self.val_dataset)}")

    def train(self):
        """モデルの学習処理"""
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0
            for batch in self.train_loader:
                batch = {key: value.to(self.device) for key, value in batch.items()}
                output = self.model(**batch)
                loss = output.loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            acc = self.evaluate()
            logging.info(f"Epoch {epoch + 1}: Loss={round(train_loss, 2)}, Accuracy={round(acc, 2)}%")

        torch.save(self.model.state_dict(), "dnn3_ws/checkpoints/model.pth")
        logging.info("Model saved.")

    def evaluate(self):
        """モデルの評価"""
        self.model.eval()
        labels_list, outputs_list = [], []
        with torch.no_grad():
            for batch in self.val_loader:
                batch = {key: value.to(self.device) for key, value in batch.items()}
                labels_list.extend(batch["labels"].cpu().numpy())
                outputs_list.extend(self.model(**batch).logits.argmax(dim=1).cpu().numpy())
        return np.mean(np.array(outputs_list) == np.array(labels_list)) * 100


if __name__ == "__main__":
    if torch.cuda.is_available():
        data_path = "dnn3_ws/text"
        processor = DatasetProcessor(data_path)
        processor.create_json()
        tokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking", mecab_kwargs={"mecab_dic": "unidic"})
        dataset = LivedoorDataset(tokenizer, data_path)
        trainer = Trainer("cl-tohoku/bert-base-japanese-whole-word-masking", dataset)
        trainer.train()
    else:
        logging.error("GPU not available.")