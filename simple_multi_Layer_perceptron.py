# -*- coding: utf-8 -*-
import numpy as np


class Layer(object):
    pass


class Dense(Layer):
    """
    全连接层
    """
    def __init__(self, input_size, output_size, act):
        """
        :param input_size: 输入维度数
        :param output_size: 输出维度数
        :param act: 激活单元
        """
        self._input_size = input_size
        self._output_size = output_size
        self._act = act
        self._act_instance = None
        self._parameters = None
        self._x = None
        self._grad = None

    @property
    def parameters(self):
        # 参数
        if self._parameters is None:
            self._parameters = self._init_parameters()
            return self._parameters
        return self._parameters

    def _init_parameters(self):
        """
        随机初始参数
        :return: 权重， 偏置
        """
        w = np.random.randn(self._input_size, self._output_size)
        b = np.random.randn(1, self._output_size)
        return w, b

    def forward(self, x):
        # 网络层前向传播
        w, b = self.parameters
        self._x = x
        self._act_instance = self._act(x.dot(w) + b)
        return self._act_instance.output

    def backward(self, error):
        # 网络层反向传播
        self._grad = error * self._act_instance.der
        return self._grad.dot(self.parameters[0].T)

    def update_parameters(self, learning_rate):
        # 网络层参数更新
        w, b = self.parameters
        grad_w = self._x.T.dot(self._grad)
        grad_b = np.sum(self._grad, axis=0, keepdims=True)
        w -= learning_rate * grad_w
        b -= learning_rate * grad_b


class Dropout(Layer):
    """
    dropout 层
    """
    def __init__(self, p):
        self._p = p
        self._mask = None

    def forward(self, x):
        self._mask = np.random.binomial(1, 1 - self._p, x.shape)
        return self._mask * x

    def backward(self, error):
        return self._mask * error

    def update_parameters(self, learning_rate):
        pass


class Act(object):

    def __init__(self, z):
        # 输出值（激活值）
        self.output = self.output_(z)


class Sigmoid(Act):
    """
    y =  1 / (1 + exp(-z))
    """
    def output_(self, z):
        return 1 / (np.exp(-z) + 1)

    @property
    def der(self):
        # 导数
        return self.output * (1 - self.output)


class Identity(Act):
    """
    y = x
    """
    def output_(self, z):
        return z

    @property
    def der(self):
        return 1


class Relu(Act):
    """
    整流单元
    """
    def output_(self, z):
        return np.maximum(z, 0)

    @property
    def der(self):
        der = np.ones_like(self.output)
        der[self.output == 0] = 0
        return der


def loss_mse(pred, target):
    """
    二次损失（的导数）
    """
    return pred - target


def loss_bce(pred, target):
    """
    二分类交叉熵（的导数）
    """
    return (pred - target) / (pred * (1 - pred) + 1e-6)


def metric_bce(pred, target):
    """
    二分类交叉熵
    """
    return -np.mean(target * np.log(pred + 1e-6) + (1 - target) * np.log(1 - pred + 1e-6))


def metric_mse(pred, target):
    """
    mean square error
    """
    return np.square(pred - target).mean()


class Sequential(object):

    def __init__(self, *layers):
        """
        :param layers: Layer, 网络层
        """
        self._layers = layers

    def _forward(self, x):
        """
        正向输入数据
        :param x: 输入或激活
        :return: 网络输出
        """
        for layer in self._layers:
            x = layer.forward(x)
        return x

    def _backward(self, error):
        """
        反向传播误差
        :param error: 误差
        """
        for layer in self._layers[::-1]:
            error = layer.backward(error)

    def _update_parameters(self, learning_rate):
        """
        更新参数
        :param learning_rate: 学习速率
        """
        for layer in self._layers:
            layer.update_parameters(learning_rate)

    def fit(self, x, y, loss, metric, epochs=50000, learning_rate=1e-3):
        """
        训练
        :param x: 输入特征
        :param y: 标签
        :param loss: 损失函数
        :param metric: 计量损失
        :param epochs: 迭代次数
        :param learning_rate: 学习速率
        """
        for i in range(epochs):
            output = self._forward(x)
            metric_val = metric(output, y)
            print(i, metric_val)
            error = loss(output, y)
            self._backward(error)
            self._update_parameters(learning_rate)


if __name__ == '__main__':
    x = np.random.randn(1000, 100)
    y = np.random.binomial(1, 0.5, (1000, 1))
    model = Sequential(Dense(100, 200, Relu), Dense(200, 50, Relu), Dense(50, 1, Sigmoid))
    model.fit(x, y, loss=loss_bce, metric=metric_bce)
