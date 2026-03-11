import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import re
from collections import Counter
import pandas as pd
import random

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes, num_classes, dropout=0.5):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout2 = nn.Dropout(0.3)

    def forward(self, x):
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        embedded = embedded.permute(0, 2, 1)  # (batch_size, embedding_dim, seq_len)
        conv_outputs = []
        for conv in self.convs:
            conv_out = conv(embedded)  # (batch_size, num_filters, seq_len - kernel_size + 1)
            conv_out = torch.relu(conv_out)
            pooled = torch.max(conv_out, dim=2)[0]  # (batch_size, num_filters)
            conv_outputs.append(pooled)
        # 拼接所有卷积层的输出
        cat_output = torch.cat(conv_outputs, dim=1)  # (batch_size, len(filter_sizes) * num_filters)
        cat_output = self.dropout(cat_output)
        output = self.fc(cat_output)  # (batch_size, 256)
        output = self.dropout2(output)
        output = self.fc2(output)   # (256, num_class)

        return output

class PersonalityDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_length=64):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # 文本转索引序列
        tokens = self.tokenize(text)
        indices = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]

        # 填充或截断
        if len(indices) < self.max_length:
            indices = indices + [self.vocab['<PAD>']] * (self.max_length - len(indices))
        else:
            indices = indices[:self.max_length]

        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)

    def tokenize(self, text):
        # 简单的分词，保留标点符号
        return re.findall(r"\b\w+\b|[^\w\s]", text)


def build_vocab(texts, min_freq=2):
    counter = Counter()
    for text in texts:
        tokens = re.findall(r"\b\w+\b|[^\w\s]", text)
        counter.update(tokens)

    # 构建词汇表
    vocab = {'<PAD>': 0, '<UNK>': 1}
    idx = 2
    for token, count in counter.items():
        if count >= min_freq:
            vocab[token] = idx
            idx += 1

    return vocab

def predict(model, vocab, text, device):
    model.eval()
    dataset = PersonalityDataset([text], [0], vocab)  # 标签设为0，实际不会使用
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            return predicted.item()

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load('../Model/personality_model.pth', map_location=device, weights_only=False)
    vocab = torch.load('../Model/vocab.pth', map_location=device, weights_only=False)

    label_to_id = {'enfj': 0, 'enfp': 1, 'entj': 2, 'entp': 3, 'esfj': 4, 'esfp': 5, 'estj': 6, 'estp': 7,
                   'infj': 8, 'infp': 9, 'intj': 10, 'intp': 11, 'isfj': 12, 'isfp': 13, 'istj': 14,
                   'istp': 15}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    df = pd.read_csv('../Data/data_predict.csv')
    text = df['text'].tolist()
    label = df['label'].tolist()
    random_number = random.randint(0, len(text) - 1)
    input_text = text[random_number]
    true_label = label[random_number]
    result = predict(model, vocab, input_text, device)
    result = id_to_label[result]
    print("Input text:", input_text)
    print("True label:", true_label)
    print("Predicted label:", result)
