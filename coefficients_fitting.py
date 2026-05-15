import pandas as pd
import numpy as np
from numpy.typing import NDArray as ND
import torch
from torch.utils import data
from torch import nn

# 进行拟合系数的函数
def fit(x:pd.DataFrame, y:ND, device,use_net=True,num_epochs=10, batch_size=16, lr=1e-2):
    '''
    系数拟合(只考虑一维情况)
    '''
    # 将pd数据转为np数组
    X = torch.tensor(x.to_numpy(), device=device)
    y = torch.tensor(y, device=device)

    # 检查数据形状，若不是列向量则转为列向量
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)

    # 创造储存结果的数组
    r2_out = [None] * (X.shape[1])
    p_out = [None] * (X.shape[1])
    loss_out = [None] * (X.shape[1])

    # X的每列数据即为y=a*f(x1,x2,..)+b中的f(x1,x2,..)方程数据
    # 对每列数据分别拟合系数
    print("待拟合特征总数:",X.shape[1])
    for i in range(X.shape[1]):
        if i%50 == 0:
            print("当前拟合特征:",i)
        # 解析解
        if not use_net:
            r2, p, loss = analytical_solving(X[:, i], y[:])
        else:
        # 神经网络
            r2, p, loss = net_solving(X[:, i], y[:],num_epochs=num_epochs, batch_size=batch_size, lr=lr)
        
        # 保存结果
        r2_out[i] = r2
        p_out[i] = p
        loss_out[i] = loss

    return r2_out, p_out, loss_out


def get_loss(y_true, y_pred):
    '''
    计算损失函数
    '''
    return float(torch.mean((y_true - y_pred) ** 2) / 2)

def get_r2(y_true, y_pred):
    '''
    计算R2
    '''
    y_true_mean = torch.mean(y_true)
    ss_total = torch.sum((y_true - y_true_mean) ** 2)  # 总平方和
    ss_residual = torch.sum((y_true - y_pred) ** 2)   # 残差平方和
    return float(1 - (ss_residual / ss_total))

def analytical_solving(X:torch.tensor, y:torch.tensor, lambda_reg=1e-6):
    '''
    lambda_reg: 正则化系数（防止 X^T X 不可逆）
    '''
    # 检查数据形状，若不是列向量则转为列向量
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)

    # 增加一列全为1的列，用于计算截距
    ones = torch.ones((X.shape[0], 1), device=X.device)
    X = torch.hstack((X, ones))

    # 计算解析解
    X_T_X = X.T @ X
    X_T_X_reg = X_T_X + lambda_reg * torch.eye(X_T_X.shape[0], dtype=torch.float64, device=X.device)
    X_T_y = X.T @ y
    XTX_inv = torch.linalg.inv(X_T_X_reg)
    p = XTX_inv @ X_T_y

    r2 = get_r2(y, X @ p)
    loss = get_loss(y, X @ p)
    
    # 展平p
    p = p.flatten().tolist()

    return r2, p, loss

def net_solving(X:torch.tensor, y:torch.tensor, num_epochs=40, batch_size=100, lr=0.1):
    def load_array(data_arrays, batch_size, is_train=False):
        """
        构造一个PyTorch数据迭代器。
        """
        dataset = data.TensorDataset(*data_arrays)
        return data.DataLoader(dataset, batch_size, shuffle=is_train)

    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)

    # features = X.float()
    # labels = y.float()

    # ========== 新增：特征归一化 ==========
    # 计算均值和标准差（避免使用全局统计量，防止数据泄露）
    X_mean = torch.mean(X, dim=0, keepdim=True)
    X_std = torch.std(X, dim=0, keepdim=True) + 1e-8  # 防止除0
    X_normalized = (X - X_mean) / X_std  # 标准化
    
    # 标签也建议归一化（可选，视y的范围而定）
    y_mean = torch.mean(y, dim=0, keepdim=True)
    y_std = torch.std(y, dim=0, keepdim=True) + 1e-8
    y_normalized = (y - y_mean) / y_std

    features = X_normalized.float()
    labels = y_normalized.float()  # 用归一化后的标签训练


    # 创建数据迭代器
    data_iter = load_array((features, labels), batch_size)

    # 定义模型
    net = nn.Sequential(nn.Linear(1, 1))

    # 初始化模型参数
    net[0].weight.data.normal_(0, 0.01)
    net[0].bias.data.fill_(0)

    # 定义损失函数和优化器
    loss = nn.MSELoss()
    trainer = torch.optim.SGD(net.parameters(), lr=lr)

    # 训练模型
    for epoch in range(num_epochs):
        for X_iter, y_iter in data_iter:
            l = loss(net(X_iter) ,y_iter)
            trainer.zero_grad()
            l.backward()
            trainer.step()

        # ========== 新增：还原参数到原始尺度 ==========
    # 模型输出是归一化后的结果，需还原参数以匹配原始数据
    # 原始模型：y = w*X + b → 归一化后：y_norm = w*(X_norm) + b
    # 还原为原始尺度：w_original = w * (y_std / X_std), b_original = b*y_std + y_mean - w_original*X_mean
# ===================== 归一化后训练完毕，开始还原权重 =====================
    w_norm = net[0].weight.data.numpy()[0][0]  # 取出归一化后的权重
    b_norm = net[0].bias.data.numpy()[0]       # 取出归一化后的偏置

    # 还原到原始数据尺度（核心公式）
    w_original = w_norm * (y_std.item() / X_std.item())
    b_original = b_norm * y_std.item() + y_mean.item() - w_original * X_mean.item()

    # 构造系数 [w, b]
    p = np.array([w_original, b_original]).reshape(-1, 1)

    X = np.hstack((X, np.ones((X.shape[0], 1))))
    y_pred =torch.tensor( X @ p)
    r2 = get_r2(y, y_pred)

    # 调用官方 MSE 损失
    criterion = nn.MSELoss()
    loss_val = criterion(y,y_pred).item()  # .item() 转成数字

    # 展平系数，返回列表格式
    p = p.flatten().tolist()
    return r2,p,loss_val