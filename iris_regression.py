from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import random


iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['target'] = iris.target

X = torch.tensor(data['sepal width (cm)'].values, dtype=torch.float32).reshape(-1, 1)  # 将 X 作为列向量
y = torch.tensor(data['sepal length (cm)'].values, dtype=torch.float32)  # y 保持为一维张量
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i:min(i+batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

def linear(w, X, b):
    return torch.matmul(X, w) + b

def Loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

batch_size = 10
w = torch.normal(0, 0.01, size=(1, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

lr = 0.03
num_epochs = 3
loss = Loss         
net = linear

train_l = loss(net(w, X_train, b), y_train)
print(f'epoch 0, loss {float(train_l.mean()):f}')

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, X_train, y_train):
        l = loss(net(w, X, b), y)
        l.sum().backward()  # 计算梯度
        sgd([w, b], lr, batch_size)  # 更新参数
    with torch.no_grad():
        train_l = loss(net(w, X_train, b), y_train)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

plt.scatter(X, y, color='blue', label='truth')
plt.show()
plt.scatter(X_train, y_train, color='blue', label='truth')
plt.plot(X_train.detach().numpy(), linear(w, X_train, b).detach().numpy())
plt.xlabel('sepal width (cm)')
plt.ylabel('sepal length (cm)')
plt.legend()
plt.show()
