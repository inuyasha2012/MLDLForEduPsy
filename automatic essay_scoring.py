import csv
import re
import nltk
from torch import nn
from torch.nn.utils.rnn import pack_sequence, PackedSequence
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tokenize(string):
    # 分词
    tokens = nltk.word_tokenize(string)
    for index, token in enumerate(tokens):
        if token == '@' and (index + 1) < len(tokens):
            tokens[index + 1] = '@' + re.sub('[0-9]+.*', '', tokens[index + 1])
            tokens.pop(index)
    return tokens


def create_vocab(essay_list):
    # 创建字典
    word_freqs = {}
    for essay in essay_list:
        for word in essay:
            if word not in word_freqs:
                word_freqs[word] = 1
            else:
                word_freqs[word] += 1
    word_index = {'unk': 0}
    for word, freq in word_freqs.items():
        if freq > 3:
            word_index[word] = len(word_index)
    return word_index, len(word_index)


def read_dataset(file_path):
    # 读取数据
    features, labels = [], []
    with open(file_path, mode='r', encoding='utf-8') as f:
        raw_data = csv.reader(f, delimiter=',')
        essay_list = []
        for i, line in enumerate(raw_data):
            if i == 0:
                continue
            essay_ = line[2].strip()
            labels.append(int(line[6]))
            essay_ = essay_.lower()
            essay = tokenize(essay_)
            essay_list.append(essay)
        word_index, vocab_size = create_vocab(essay_list)
        for essay in essay_list:
            indices = []
            for word in essay:
                if word in word_index:
                    indices.append(word_index[word])
            features.append(torch.tensor(indices, device=device))
    return features, torch.tensor(labels, dtype=torch.float, device=device), vocab_size


class AesNet(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_size, max_score, min_score):
        super(AesNet, self).__init__()
        self.max_score = max_score
        self.min_score = min_score
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, inputs):
        emb = self.embeddings(inputs.data)
        emb = PackedSequence(emb, inputs.batch_sizes)
        output, (h, c) = self.lstm(emb)
        output = self.linear(h)
        # out[out > self.max_score] = self.max_score
        # out[out < self.min_score] = self.min_score
        return output.view(-1)


class AesDataset(Dataset):

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels


def collate(batch):
    batch.sort(key=lambda x: x[0].size()[0], reverse=True)
    features = []
    labels = []
    for feature, label in batch:
        features.append(feature)
        labels.append(label)
    features_batch = pack_sequence(features)
    return features_batch, torch.tensor(labels, device=device)


def kappa(a, b, max_val, min_val):
    a = np.round(a)
    b = np.round(b)
    max_val = int(max_val)
    min_val = int(min_val)
    return _kappa(a, b,max_val, min_val )


def _kappa(a, b, max_val, min_val):
    n = max_val - min_val + 1
    c = np.vstack((a, b)).transpose()
    o = np.zeros((n, n))
    w = np.zeros((n, n))
    e = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            bool_list = np.all(c == [i + min_val, j + min_val], axis=1)
            o[i, j] = len(bool_list[bool_list == True])
            w[i, j] = ((i - j) ** 2) / ((n - 1) ** 2)
            e[i, j] = len(a[a == i + min_val]) * len(b[b == j + min_val]) / len(a)
    val = 1 - np.sum(w * o) / np.sum(w * e)
    return val


if __name__ == '__main__':
    features, labels, vocab_size = read_dataset('data/aes/aes.csv')
    max_score = labels.max().item()
    min_score = labels.min().item()
    train_dataset = AesDataset(features, labels)
    train_data_loader = DataLoader(train_dataset, batch_size=100, shuffle=True, collate_fn=collate)
    model = AesNet(vocab_size, 200, 50, max_score, min_score)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-2)
    for i in range(20):
        for batched in train_data_loader:
            x, y = batched
            output = model(x)
            loss = criterion(output, y)
            print(loss.item())
            print(kappa(output.detach().numpy(), y.detach().numpy(), max_score, min_score))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()