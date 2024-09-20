import matplotlib.pyplot as plt

# 年份数据
year = [2004 + i for i in range(19)]

# 价格数据
prices = [
    4355.94, 5041.21, 6152.17, 8439.06, 8781, 8988, 10615,
    10925.84, 12000.88, 13954, 14739, 14083, 16346, 17685,
    21581.78, 24015, 27112, 30580, 29455
]

# 系数a和b
param = [0.0, 0.0]

# 最小二乘法计算线性回归的参数
n = len(year)
sum_x = sum(year)
sum_y = sum(prices)
sum_xy = sum(year[i] * prices[i] for i in range(n))
sum_x2 = sum(year[i] ** 2 for i in range(n))

# 计算a和b的值
param[0] = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)  # 斜率a
param[1] = (sum_y - param[0] * sum_x) / n  # 截距b

print("系数a和b分别为：" + str(param))


# 定义线性函数
def linear_function(x, param):
    return param[0] * x + param[1]


# 绘图
plt.plot(year, prices, 'o', label='price data')
plt.plot(year, [linear_function(y, param) for y in year], label='fitted line')
plt.xlabel('year')
plt.ylabel('price')
plt.legend()
plt.show()
