import torch
import matplotlib.pyplot as plt
import random

# X = torch.rand(20, 1) # 均匀分布
X = torch.normal(0, 1, size=(100, 3))
true_w = torch.tensor([2.9, 1.3, 0.6])
true_b = -2.8
# y = true_w*X + true_b + torch.randn(20, 1)
y = torch.matmul(X, true_w) + true_b
y += torch.normal(0, 0.1, y.shape)

w = torch.normal(0, 0.01, size=(3, 1), requires_grad=True)    
b = torch.zeros(1, requires_grad=True)

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i:min(i+batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

def linear(w, X, b):
    return torch.matmul(X, w) + b

def MSELoss(y_hat, y):
    # return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
    return ((y_hat - y.reshape(y_hat.shape)) ** 2).mean()

def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

lr = 0.05
batch_size = 10
epochs = 50
Loss = MSELoss
net = linear

print(f'true w = {true_w}\ntrue b = {true_b}')
print(f'Before training, \nw = {w}\nb = {b}')

for epoch in range(epochs):
    for X_batch, y_batch in data_iter(batch_size, X, y):
        l = Loss(net(w, X_batch, b), y_batch)
        l.sum().backward()
        sgd((w, b), lr, batch_size)
    with torch.no_grad():
        loss = Loss(net(w, X, b), y)
        # print(f'epoch {epoch + 1}, loss = {loss.mean():f}\nw = {w}\nb = {b}')
        print(f'epoch {epoch + 1}, loss = {loss.mean():f}')

print(f'After training, \nw = {w}\nb = {b}')
