# 深度知识追踪，http://stanford.edu/~cpiech/bio/papers/deepKnowledgeTracing.pdf

from torch.nn.utils.rnn import pack_sequence
from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn import RNN, Sigmoid, Linear, Dropout, Module
from sklearn.metrics import roc_auc_score
import random
import csv


def read_data(path):
    """
    从文件中读取数据
    :param path: str 文件路径
    :return: int 知识点数量, sequence 原始数据序列
    """
    with open(path, 'r', encoding='UTF-8') as f:
        skill_dt = {}
        seq_dt = {}
        readlines = csv.reader(f)
        for i, readline in enumerate(readlines):
            if i == 0:
                continue
            skill_id = readline[2]
            user_id = readline[0]
            correct = int(readline[1])
            if skill_id == '' or user_id == '' or correct == '':
                continue
            if skill_id not in skill_dt:
                skill_dt[skill_id] = len(skill_dt)
            if user_id in seq_dt:
                seq_dt[user_id].append((skill_dt[skill_id], correct))
            else:
                seq_dt[user_id] = [(skill_dt[skill_id], correct)]
        skill_size = len(skill_dt)
        seq_list = list(seq_dt.values())
        return skill_size, seq_list


def split_dataset(seqs, val_rate=0.2, test_rate=0.2):
    """
    将数据拆分为训练集，验证集和测试集
    :param seqs: sequence 原始数据序列
    :param val_rate: float 验证集占总体数据比
    :param test_rate: float 测试集占总体数据比
    :return:
    """
    random.shuffle(seqs)
    seq_size = len(seqs)
    val_size = int(seq_size * val_rate)
    test_size = int(seq_size * test_rate)
    val_seqs = seqs[:val_size]
    test_seqs = seqs[val_size:val_size + test_size]
    train_seqs = seqs[val_size + test_size:]
    return train_seqs, val_seqs, test_seqs


class QuizDataSet(Dataset):

    def __len__(self):
        return len(self._seq_list)

    def __getitem__(self, index):
        feature_size = self._skill_size * 2
        seq = self._seq_list[index]
        seq_size = len(seq)
        # 包含初始状态x_{0}
        x = torch.zeros(seq_size, feature_size)
        # delta(t + 1)
        y_skill = torch.zeros(seq_size, self._skill_size)
        # a_{t+1}
        y_ans = torch.zeros(seq_size)
        for i, v in enumerate(seq):
            if i < seq_size - 1:
                new_feature_v = torch.zeros(feature_size)
                # skill_id * 2 + correct
                idx = int(v[0] * 2 + v[1])
                new_feature_v[idx] = 1
                x[i + 1] = new_feature_v
            new_skill_v = torch.zeros(self._skill_size)
            new_skill_v[v[0]] = 1
            y_skill[i] = new_skill_v
            y_ans[i] = v[1]
        return x, y_skill, y_ans

    def __init__(self, skill_size, seq_list):
        self._skill_size = skill_size
        self._seq_list = seq_list


def collate(batch):
    # 压紧序列前需要先排序
    batch.sort(key=lambda x: x[0].size()[0], reverse=True)
    y_skill = []
    y_ans = []
    x = []
    for each_x, each_y, each_ans in batch:
        x.append(each_x)
        y_skill.append(each_y)
        y_ans.append(each_ans)
    # 压紧序列数据
    x_batch = pack_sequence(x)
    # delta(t+1)和a_{t+1}不用保留序列结构信息
    y_skill_batch = pack_sequence(y_skill).data
    y_ans_batch = pack_sequence(y_ans).data
    return x_batch, y_skill_batch, y_ans_batch


class DktNet(Module):
    """
    deep knowledge tracing model
    input => rnn => dropout => sigmoid => output
    """

    def __init__(self, skill_size, rnn_h_size, rnn_layer_size, dropout_rate):
        """
        :param skill_size: int 知识点数量
        :param rnn_h_size: int rnn隐藏单元数量
        :param rnn_layer_size: int rnn隐藏层数量
        :param dropout_rate: float
        """
        super(DktNet, self).__init__()
        self.rnn = RNN(skill_size * 2, rnn_h_size, rnn_layer_size)
        self.dropout = Dropout(p=dropout_rate)
        self.linear = Linear(rnn_h_size, skill_size)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        rnn_output, h = self.rnn(x)
        # rnn_output无需保留序列结构数据，直接用rnn_output.data
        dropout_output = self.dropout(rnn_output.data)
        linear_output = self.linear(dropout_output)
        output = self.sigmoid(linear_output)
        return output


def compute_auc(data_loader, model):
    """
    计算验证集和测试集的auc值
    """
    y_pred = torch.tensor([])
    y_ans = torch.tensor([])
    for x_batch, y_skill_batch, y_ans_batch in data_loader:
        skill_pred = model(x_batch)
        _y_pred = (skill_pred.data * y_skill_batch).sum(dim=1)
        y_pred = torch.cat((y_pred, _y_pred))
        y_ans = torch.cat((y_ans, y_ans_batch))
    return roc_auc_score(y_ans, y_pred)

# rnn隐藏单元数量
HIDDEN_SIZE = 200
# rnn隐藏层数量
HIDDEN_LAYER_SIZE = 1
DROPOUT_RATE = 0.5
EPOCHS = 20
BATCH_SIZE = 100

skill_size, seq_list = read_data('data/dkt.csv')
train_seqs, val_seqs, test_seqs = split_dataset(seq_list)
train_dataset = QuizDataSet(skill_size, train_seqs)
val_dataset = QuizDataSet(skill_size, val_seqs)
model = DktNet(skill_size, HIDDEN_SIZE, HIDDEN_LAYER_SIZE, DROPOUT_RATE)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adagrad(model.parameters())
train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate, shuffle=True)
val_data_loader = DataLoader(val_dataset, collate_fn=collate, batch_size=BATCH_SIZE)
max_val_auc = 0
for i in range(EPOCHS):
    train_y_pred = torch.tensor([])
    train_y_ans = torch.tensor([])
    for j, batched in enumerate(train_data_loader):
        x_batch, y_skill_batch, y_ans_batch = batched
        skill_pred = model(x_batch)
        y_pred = (skill_pred * y_skill_batch).sum(dim=1)
        loss = criterion(y_pred, y_ans_batch)
        train_y_pred = torch.cat((train_y_pred, y_pred))
        train_y_ans = torch.cat((train_y_ans, y_ans_batch))
        print('epoch: {0}, step: {1}, loss: {2}'.format(i + 1, j + 1, loss.item()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        # 验证集auc
        val_auc = compute_auc(val_data_loader, model)
        # 训练集auc
        train_auc = roc_auc_score(train_y_ans, train_y_pred.detach().numpy())
        print('epoch: {0}, train_auc: {1}, val_auc: {2}'.format(i + 1, train_auc, val_auc))
        if val_auc > max_val_auc:
            max_val_auc = val_auc
            torch.save(model, 'best_dkt_model.pt')
            print('save new model, epoch: {0}'.format(i + 1))

# 测试集
best_model = torch.load('best_dkt_model.pt')
best_model.eval()
test_dataset = QuizDataSet(skill_size, test_seqs)
test_data_loader = DataLoader(test_dataset, collate_fn=collate, batch_size=BATCH_SIZE)
test_auc = compute_auc(test_data_loader, best_model)
print('test_auc: {0}'.format(test_auc))