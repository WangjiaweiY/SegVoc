import torch
import matplotlib.pyplot as plt
import torchvision
from torch.utils import data
from torchvision import transforms
from torch import nn
import time


def get_dataloader_workers():
    return 0 # 工作时线程数，为0表示不适用额外线程

def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="./data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="./data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))

train_iter, test_iter = load_data_fashion_mnist(batch_size=128, resize=224)

class Reshape(torch.nn.Module):
    def forward(self, x):
        return x.view(-1, 1, 28, 28)
    
LeNet = nn.Sequential(
    Reshape(), 
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10)
)

AlexNet = nn.Sequential(
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2), nn.Flatten(),
    nn.Linear(6400, 4096), nn.ReLU(), nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(p=0.5), 
    nn.Linear(4096, 10)
)

lr = 0.01
epochs = 10
net = AlexNet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)
Loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=lr)

# 保存每个轮次的平均损失和准确率
epoch_losses = []
epoch_accuracies = []

start_time = time.time()

for epoch in range(epochs):
    epoch_loss = 0.0
    epoch_accuracy = 0.0
    total_batches = 0
    for i, (X, y) in enumerate(train_iter):
        optimizer.zero_grad()
        X, y = X.to(device), y.to(device)
        y_hat = net(X)
        l = Loss(y_hat, y)
        l.backward()
        optimizer.step()

        # 计算批次精度
        _, predicted = torch.max(y_hat, 1)  # 获取最大预测值的索引
        correct = (predicted == y).sum().item()  # 计算正确预测的数量
        accuracy = correct / y.size(0)  # 计算当前批次的精度
        
        # 累加损失和准确率
        epoch_loss += l.item()
        epoch_accuracy += accuracy
        total_batches += 1

    # 计算当前轮次的平均损失和准确率
    epoch_loss /= total_batches
    epoch_accuracy /= total_batches
    
    # 保存每个轮次的损失和准确率
    epoch_losses.append(epoch_loss)
    epoch_accuracies.append(epoch_accuracy)

    # 输出当前轮次的平均损失和准确率
    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')

# 计算并输出训练时间
end_time = time.time()
print(f'Training Time: {end_time - start_time:.2f} seconds')

# 绘制损失和精度图
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), epoch_losses, label='Loss', color='blue')
plt.plot(range(1, epochs + 1), epoch_accuracies, label='Accuracy', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Loss and Accuracy over Epochs')
plt.legend()
plt.show()