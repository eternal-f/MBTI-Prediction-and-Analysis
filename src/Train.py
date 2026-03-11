import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import re
from collections import Counter
import pandas as pd
import csv
import os

torch.manual_seed(5002)
np.random.seed(5002)

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
            # pooled = self.dropout2(pooled)
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


def train_model():
    df = pd.read_csv('../Data/data_train.csv')
    texts = df['text'].tolist()
    labels = df['label'].tolist()

    # 创建标签映射字典
    label_to_id = {'enfj': 0, 'enfp': 1, 'entj': 2, 'entp': 3, 'esfj': 4, 'esfp': 5, 'estj': 6, 'estp': 7,
                   'infj': 8, 'infp': 9, 'intj': 10, 'intp': 11, 'isfj': 12, 'isfp': 13, 'istj': 14, 'istp': 15}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    # 转换标签
    labels = [label_to_id[label] for label in labels]
    # 构建词汇表
    vocab = build_vocab(texts)
    vocab_size = len(vocab)
    # 划分训练集和验证集
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    # 创建数据集和数据加载器
    train_dataset = PersonalityDataset(train_texts, train_labels, vocab)
    val_dataset = PersonalityDataset(val_texts, val_labels, vocab)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 模型参数
    embedding_dim = 100
    num_filters = 100
    filter_sizes = [2, 3, 4]  # 不同尺寸的卷积核
    num_classes = 16
    learning_rate = 0.001
    num_epochs = 60

    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TextCNN(vocab_size, embedding_dim, num_filters, filter_sizes, num_classes).to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    csv_file = '../Output/training_metrics.csv'
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch','Train_Loss', 'Val_Loss', 'Val_Acc'])

    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # 验证
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        train_loss_avg = total_loss / len(train_loader)
        val_loss_avg = val_loss / len(val_loader)
        val_acc = 100. * correct / total

        print(f'Epoch {epoch + 1}/{num_epochs}, '
              f'Train Loss: {train_loss_avg:.4f}, '
              f'Val Loss: {val_loss_avg:.4f}, '
              f'Val Acc: {val_acc:.2f}%')
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, train_loss_avg, val_loss_avg, val_acc])

    # 保存整个模型
    torch.save(model, '../Model/personality_model.pth')
    # 同时保存词汇表，以便后续使用
    torch.save(vocab, '../Model/vocab.pth')

    return model, vocab

# 训练模型
if __name__ == '__main__':
    model, vocab = train_model()
