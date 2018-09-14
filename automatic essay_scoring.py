import csv
import re
import nltk
from torch import nn
from torch.nn.utils.rnn import pack_sequence
import torch
from torch.utils.data import DataLoader, TensorDataset


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
    word_index = {}
    for essay in essay_list:
        for word in essay:
            if word not in word_freqs:
                word_freqs[word] = 1
                word_index[word] = len(word_index)
            else:
                word_freqs[word] += 1
    return word_index, word_freqs


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
        word_index, word_freqs = create_vocab(essay_list)
        used_word = {}
        for essay in essay_list:
            indices = []
            for word in essay:
                if word in word_index and word_freqs[word] > 3:
                    if word not in used_word:
                        used_word[word] = 1
                    indices.append(word_index[word])
            features.append(torch.tensor(indices))
    return torch.tensor(features), torch.tensor(labels), len(used_word)


class AesNet(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(AesNet, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        lstm_out, _ = self.lstm(embeds)
        linear_out = self.linear(lstm_out)
        return linear_out


if __name__ == '__main__':

    features, labels, vocab_size = read_dataset('data/aes/aes.csv')
    train_dataset = TensorDataset(features, labels)
    train_data_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    model = AesNet(vocab_size, 100, 50)