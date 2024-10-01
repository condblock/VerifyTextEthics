import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import json
from sklearn.metrics import classification_report
import glob

# 데이터셋 클래스 정의
class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['sentences'][0]['text']
        labels = [0] * 7  # CENSURE, HATE, DISCRIMINATION, SEXUAL, ABUSE, VIOLENCE, CRIME
        types = item['sentences'][0]['types']
        if types != 'IMMORAL_NONE':
            for t in types:
                if t == 'CENSURE':
                    labels[0] = 1
                elif t == 'HATE':
                    labels[1] = 1
                elif t == 'DISCRIMINATION':
                    labels[2] = 1
                elif t == 'SEXUAL':
                    labels[3] = 1
                elif t == 'ABUSE':
                    labels[4] = 1
                elif t == 'VIOLENCE':
                    labels[5] = 1
                elif t == 'CRIME':
                    labels[6] = 1

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.float)
        }

# 데이터 로드 함수
def load_data(file_pattern):
    data = []
    for file_path in glob.glob(file_pattern):
        with open(file_path, 'r', encoding='utf-8') as f:
            data.extend(json.load(f))
    return data

# 데이터셋 및 데이터로더 생성
def create_data_loader(data, tokenizer, max_len, batch_size):
    ds = CustomDataset(data, tokenizer, max_len)
    return DataLoader(ds, batch_size=batch_size, num_workers=4)

# 모델 및 토크나이저 초기화
tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base")
model = AutoModelForSequenceClassification.from_pretrained("beomi/KcELECTRA-base", num_labels=7)

# CUDA 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 데이터 로드
train_data = load_data('dataset/train/talksets-train-*.json')
val_data = load_data('dataset/valid/talksets-train-6.json')

# 데이터로더 생성
train_data_loader = create_data_loader(train_data, tokenizer, max_len=128, batch_size=16)
val_data_loader = create_data_loader(val_data, tokenizer, max_len=128, batch_size=16)

# 트레이너 설정
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    evaluation_strategy="epoch",
    eval_steps=500,
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data_loader.dataset,
    eval_dataset=val_data_loader.dataset
)

# 모델 학습
trainer.train()

# 모델 저장
trainer.save_model('./saved_model')