import csv
import re
import nltk
from torch import nn
from torch.nn.utils.rnn import pack_sequence, PackedSequence, pad_packed_sequence, pad_sequence, pack_padded_sequence
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pickle as pk

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
    feature_list, label_list = [], []
    with open(file_path, mode='r', encoding='utf-8') as f:
        raw_data = csv.reader(f, delimiter=',')
        essay_list = []
        for i, line in enumerate(raw_data):
            if i == 0:
                continue
            label_list.append(int(line[6]))
            _essay = line[2].strip().lower()
            essay = tokenize(_essay)
            essay_list.append(essay)
        word_index, vocab_size = create_vocab(essay_list)
        for essay in essay_list:
            indices = []
            for word in essay:
                if word in word_index:
                    indices.append(word_index[word])
            feature_list.append(torch.tensor(indices, device=device))
    return feature_list, torch.tensor(label_list, dtype=torch.float, device=device), vocab_size, word_index


class AesNet(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_size, max_score, min_score, embedding_weight):
        super(AesNet, self).__init__()
        self.max_score = max_score
        self.min_score = min_score
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.embeddings.weight = nn.Parameter(embedding_weight, requires_grad=False)
        self.lstm = nn.LSTM(embedding_dim, hidden_size)
        self.linear = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.cnn = nn.Conv1d(embedding_dim, hidden_size, 2)

    def forward(self, inputs):
        embeddings = self.embeddings(inputs.data)
        output = self.cnn(embeddings)
        output, (h, c) = self.lstm(embeddings)
        output = pad_packed_sequence(output)
        output = output[0].sum(0) / output[1].view(-1, 1)
        output = self.linear(output)
        output = self.sigmoid(output)
        # output[output > self.max_score] = self.max_score
        # output[output < self.min_score] = self.min_score
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
    max_val = int(max_val)
    min_val = int(min_val)
    a = np.round(a * (max_val - min_val) + min_val)
    b = np.round(b * (max_val - min_val) + min_val)
    return _kappa(a, b, max_val, min_val)


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


def _get_embedding_dt():
    embedding_dt = {}
    f = open('data/aes/glove.6B.50d.txt', encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = torch.as_tensor([float(_) for _ in values[1:]])
        embedding_dt[word] = coefs
    f.close()
    return embedding_dt


def get_embedding_weight(vocab_size, word_index):
    embedding_dt = _get_embedding_dt()
    embedding_weight = torch.zeros(vocab_size, 50)
    for word, i in word_index.items():
        embedding_vector = embedding_dt.get(word)
        if embedding_vector is not None:
            embedding_weight[i] = embedding_vector
    return embedding_weight


if __name__ == '__main__':
    # features, labels, vocab_size, word_index = read_dataset('data/aes/aes.csv')
    # with open('data.pkl', 'wb') as f:
    #     pk.dump([features, labels, vocab_size, word_index], f)
    with open('data.pkl', 'rb') as f:
        features, labels, vocab_size, word_index = pk.load(f)
    # embedding_weight = get_embedding_weight(vocab_size, word_index)
    # with open('weight.pkl', 'wb') as f:
    #     pk.dump(embedding_weight, f)
    with open('weight.pkl', 'rb') as f:
        embedding_weight = pk.load(f)
    max_score = labels.max().item()
    min_score = labels.min().item()
    labels = (labels - min_score) / (max_score - min_score)
    train_dataset = AesDataset(features, labels)
    train_data_loader = DataLoader(train_dataset, batch_size=100, shuffle=True, collate_fn=collate)
    model = AesNet(vocab_size, 50, 50, max_score, min_score, embedding_weight)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
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
