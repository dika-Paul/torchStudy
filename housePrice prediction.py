import torch
import matplotlib.pyplot as plt

x = torch.linspace(0, 100, 100).type(torch.FloatTensor)
rand = torch.randn(100) * 10
y = x + rand

x_train = x[:90]
x_test = x[90:]
y_train = y[:90]
y_test = y[90:]

plt.figure(figsize=(10, 8))
plt.plot(x_train.data.numpy(), y_train.data.numpy(), 'o')
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
