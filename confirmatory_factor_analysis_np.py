# 用numpy的手动求导估计一阶验证性因子分析的最小二乘法参数估计

import numpy as np


def cfa(s, lam, learning_rate=0.01, max_iter=5000):
    """
    手动求导CFA梯度下降参数估计
    设定潜变量的方差为1，即潜变量协方差矩阵为相关系数矩阵
    :param s: 样本协方差矩阵
    :param lam: 因子载荷初值和结构
    :param learning_rate: 学习速率（步长）
    :param max_iter: 最大迭代次数
    :return: 因子载荷，潜变量相关系数，误差协方差估计值
    """
    var_e, phi = np.eye(lam.shape[0]), np.eye(lam.shape[1])
    for i in range(max_iter):
        sigma = np.dot(np.dot(lam, phi), lam.transpose()) + var_e
        omega = sigma - s
        # 手动求导
        omega_lam = np.dot(omega, lam)
        dlam, dphi, dvar_e = 2 * np.dot(omega_lam, phi), np.dot(lam.transpose(), omega_lam), omega
        dlam[lam == 0], dphi[range(lam.shape[1]), range(lam.shape[1])], dvar_e[var_e == 0] = 0, 0, 0
        # 参数更新
        lam, phi, var_e = lam - learning_rate * dlam, phi - learning_rate * dphi, var_e - learning_rate * dvar_e
    return lam, phi, var_e


data = np.loadtxt('data/cfa.dat')
s = np.cov(data, rowvar=False, bias=True)
lam = np.array([
    [1, 0],
    [1, 0],
    [1, 0],
    [0, 1],
    [0, 1],
    [0, 1]
])
lam, phi, var_e = cfa(s, lam)
# 因子载荷
print(lam)
# 误差方差
print(np.diag(var_e))
# 潜变量协方差矩阵
print(phi)