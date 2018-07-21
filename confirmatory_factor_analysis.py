# 用pytorch的自动求导估计一阶验证性因子分析的最小二乘法参数估计

import torch
import numpy as np


def cfa(s, lam, learning_rate=0.01, max_iter=1000):
    """
    结构方程模型中的测量模型，验证性因子分析参数估计
    设定潜变量的方差为1，即潜变量协方差矩阵为相关系数矩阵
    :param s: 样本协方差矩阵
    :param lam: 因子载荷初值和结构
    :param learning_rate: 学习速率（步长）
    :param max_iter: 最大迭代次数
    :return: 因子载荷，潜变量相关系数，误差协方差估计值
    """
    # 设定误差协方差矩阵初值
    var_e = torch.eye(lam.size(0), requires_grad=True, dtype=torch.float)
    # 设定潜在变量相关矩阵初值
    phi = torch.eye(lam.size(1), requires_grad=True, dtype=torch.float)
    for i in range(max_iter):
        # 计算估计协方差矩阵
        sigma = lam.mm(phi).mm(lam.t()) + var_e
        # 计算损失
        loss = (sigma - s).pow(2).sum()
        print(loss.item())
        loss.backward()
        with torch.no_grad():
            # 参数更新
            lam.grad[lam == 0] = 0
            lam -= learning_rate * lam.grad
            lam.grad.zero_()
            var_e.grad[var_e == 0] = 0
            var_e -= learning_rate * var_e.grad
            var_e.grad.zero_()
            phi.grad[range(lam.size(1)), range(lam.size(1))] = 0
            phi -= learning_rate * phi.grad
            phi.grad.zero_()
    return lam, phi, var_e


data = np.loadtxt('data/cfa.dat', dtype=np.float32)
# 计算样本协方差矩阵
s = torch.from_numpy(np.cov(data, rowvar=False, bias=True)).float()
# 设定因子载荷初值
lam = torch.tensor(
    [
        [1, 0],
        [1, 0],
        [1, 0],
        [0, 1],
        [0, 1],
        [0, 1]
    ],
    dtype=torch.float,
    requires_grad=True)
lam, phi, var_e = cfa(s, lam)
print('因子载荷估计值')
print(lam)
print('潜变量相关矩阵')
print(phi)
print('误差协方差矩阵')
print(var_e)
