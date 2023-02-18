import torch
import matplotlib.pyplot as plt

x = torch.linspace(0, 100, 100).type(torch.FloatTensor)
rand = torch.randn(100) * 10
y = x + rand

# 设置训练集和测试集
x_train = x[:90]
x_test = x[90:]
y_train = y[:90]
y_test = y[90:]

# 预测模型的参数
a = torch.rand(1, requires_grad=True)
b = torch.rand(1, requires_grad=True)

# 学习率
learning_rate = 0.00001

a.cuda()
b.cuda()

for i in range(100000):
    prediction = a.expand_as(x_train) * x_train + b.expand_as(x_train)
    loss = torch.mean((prediction - y_train) ** 2)
    loss.backward()
    a.data.add_(-(learning_rate * a.grad))
    b.data.add_(-(learning_rate * b.grad))
    a.grad.data.zero_()
    b.grad.data.zero_()

x_data = x_train.data.numpy()
plt.figure(figsize=(10, 8))
xplot, = plt.plot(x_train.data.numpy(), y_train.data.numpy(), 'o')
yplot, = plt.plot(x_data, a.data.numpy()*x_data + b.data.numpy())
plt.xlabel("X")
plt.ylabel("Y")
str1 = str(a.data.numpy()[0]) + 'x +' + str(b.data.numpy()[0])
plt.legend([xplot, yplot], ['Data', str1])
plt.show()
