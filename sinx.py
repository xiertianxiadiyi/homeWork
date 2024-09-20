import numpy as np
import matplotlib.pyplot as plt

data_count = 100  # 数据点个数
x = np.linspace(0, 2 * np.pi, data_count)

param = [0] * 6

param = [0, 1, 0, -1 / 6, 0, 1 / 120]  #以泰勒张开的五次多项式系数作为初始值


# 定义五次多项式
def function_sin(xx, param):
    res = param[0] + param[1] * xx + param[2] * xx ** 2 + param[3] * xx ** 3 + param[4] * xx ** 4 + param[5] * xx ** 5
    # print(str(xx)+"的拟合值为："+str(res))
    return res


# 定义损失函数
def loss_function(param):
    res = 0
    for i in range(data_count):
        res += (np.sin(x[i]) - function_sin(x[i], param)) ** 2  # 按点计算损失
    res = res / data_count
    return res  # 返回平均损失


# 损失函数偏导数
def partial_derivative(j, param):
    res = 0
    for i in range(data_count):
        res += (np.sin(x[i]) - function_sin(x[i], param)) * x[i] ** j
    res = -2 * res / data_count
    # print("第"+str(j+1)+"个参数的偏导数为："+str(res))
    return res


# 反向传播更新参数
learning_rate = 0.0000001  # 学习率
for epoch in range(9):  # 在不同样本区间单独训练
    if epoch == 0:
        times = 1000
        x = np.linspace(0, 2 * np.pi, data_count)
    elif epoch == 1:
        times = 8000
        x = np.linspace(5, 6, data_count)
    elif epoch == 2:
        times = 500
        x = np.linspace(0, 1, data_count)
    elif epoch == 3:
        times = 5000
        x = np.linspace(3, 4, data_count)
    elif epoch == 4:
        times = 6000
        x = np.linspace(4, 2 * np.pi, data_count)
    elif epoch == 5:
        times = 1000
        x = np.linspace(2, 3, data_count)
    elif epoch == 6:
        times = 3000
        x = np.linspace(4, 2 * np.pi, data_count)
    elif epoch == 7:
        times = 3000
        x = np.linspace(2, 2 * np.pi, data_count)
    elif epoch == 8:
        times = 1000
        x = np.linspace(0, 2 * np.pi, data_count)

    for i in range(times):  # 动态调整学习率
        min_loss = loss_function(param)
        for j in range(6):
            param[j] -= learning_rate * partial_derivative(j, param)
        if loss_function(param) < min_loss:
            param_min = param.copy()
            if min_loss - loss_function(param) < 0.01:
                learning_rate *= 1.1  # 学习率增加
            min_loss = loss_function(param)
        else:
            if loss_function(param) - min_loss > 10:  # 如果发散了，就恢复成之前的最优参数
                param = param_min.copy()
            learning_rate *= 0.85  # 学习率减小

            min_loss = loss_function(param)
        print('第%d次迭代，参数值：%s，损失函数值：%f' % (i + 1, param, loss_function(param)))
    learning_rate = 0.0000001

y = np.sin(x)
# 绘制拟合曲线
plt.plot(x, y, 'o', label='data')
y1 = function_sin(x, param)
plt.plot(x, y1, label='fit')
plt.legend()
plt.show()
