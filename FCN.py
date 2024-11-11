import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import VOCSegmentation
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import time
import os
import torchvision
from utils import *
from torchvision.models import ResNet18_Weights
from U_Net import U_Net
from tqdm import tqdm
import segmentation_models_pytorch as smp


# 检查是否有GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size, crop_size = 32, (320, 480)
train_iter, test_iter = load_data_voc(batch_size, crop_size)
test_images, test_labels = read_voc_images(voc_dir, False)
test_imgs = []

pretrained_net = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
net = nn.Sequential(*list(pretrained_net.children())[:-2])
num_classes = 21
net.add_module('final_conv', nn.Conv2d(512, num_classes, kernel_size=1))
net.add_module('transpose_conv', nn.ConvTranspose2d(num_classes, num_classes,
                                    kernel_size=64, padding=16, stride=32))
W = bilinear_kernel(num_classes, num_classes, 64)
net.transpose_conv.weight.data.copy_(W);

# net = U_Net()

# net = smp.Unet(encoder_name="resnet34",       
#                  encoder_weights="imagenet",     
#                  in_channels=3,                  
#                  classes=21) 

def predict(img):
    X = train_iter.dataset.normalize_image(img).unsqueeze(0)
    pred = net(X.to(devices)).argmax(dim=1)
    return pred.reshape(pred.shape[1], pred.shape[2])

def label2image(pred):
    colormap = torch.tensor(VOC_COLORMAP, device=devices)
    X = pred.long()
    return colormap[X, :]

small_train_iter = []
for i, (X, y) in enumerate(train_iter):
    small_train_iter.append((X, y))
    if i >= 5:
        break
   
def loss(inputs, targets):
    # return F.cross_entropy(inputs, targets, reduction='none').mean(1).mean(1)
    return F.cross_entropy(inputs, targets)

num_epochs, lr, wd, devices = 20, 0.05, 1e-3, device
trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)
# trainer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=wd)

start_time = time.time()
net.to(device)
net.load_state_dict(torch.load("FCNmodel.pth"))
total_loss = []
total_acc = []

crop_rect = (0, 0, 320, 480)
X = torchvision.transforms.functional.crop(test_images[0], *crop_rect)
pred = label2image(predict(X))
test_imgs += [X.permute(1,2,0), pred.cpu(),
            torchvision.transforms.functional.crop(
                test_labels[0], *crop_rect).permute(1,2,0)]

# for epoch in range(num_epochs):
#     epoch_loss = 0
#     correct_pixels = 0
#     total_pixels = 0
#     with tqdm(total=len(train_iter), desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
#         for i, (features, labels) in enumerate(train_iter):
#             features, labels = features.to(device), labels.to(device)
#             net.train()
#             trainer.zero_grad()
#             pred = net(features)
#             l = loss(pred, labels)
#             l.sum().backward()
#             trainer.step()
#             epoch_loss += l.mean().item()
#             pred_labels = torch.argmax(pred, dim=1)
#             correct_pixels += (pred_labels == labels).sum().item()
#             total_pixels += labels.numel()
#             pbar.set_postfix(loss=l.mean().item())
#             pbar.update(1)
#     average_loss = epoch_loss / len(train_iter)
#     accuracy = correct_pixels / total_pixels
#     total_loss.append(average_loss)
#     total_acc.append(accuracy)
#     print(f'time: {time.time() - start_time},epoch: {epoch + 1}, loss: {average_loss}, acc: {accuracy:.4f}')
# pred = label2image(predict(X))
# test_imgs += [X.permute(1,2,0), pred.cpu(),
#             torchvision.transforms.functional.crop(
#                 test_labels[0], *crop_rect).permute(1,2,0)]


# end_time = time.time()
# print(f'training time: {end_time - start_time}')

# show_images(test_imgs)

# torch.save(net.state_dict(), 'U_Netmodel.pth')

# plt.figure(figsize=(10, 6))
# plt.plot(total_acc, c='red', label='Accuracy', linewidth=2, linestyle='-', marker='o', markersize=4)
# plt.plot(total_loss, c='blue', label='Loss', linewidth=2, linestyle='--', marker='x', markersize=4)
# plt.title('Training Accuracy and Loss Over Epochs', fontsize=16)
# plt.xlabel('Epochs', fontsize=14)
# plt.ylabel('Value', fontsize=14)
# plt.legend(loc='upper right', fontsize=12)
# plt.grid(True)
# plt.tight_layout()
# plt.show()


net.eval()

n, imgs = 3, []
for i in range(n):
    crop_rect = (0, 0, 320, 480)
    X = torchvision.transforms.functional.crop(test_images[i], *crop_rect)
    pred = label2image(predict(X))
    imgs += [X.permute(1,2,0), pred.cpu(),
             torchvision.transforms.functional.crop(
                 test_labels[i], *crop_rect).permute(1,2,0)]

show_images(imgs)

test_accuracy = 0
test_batches = 0
test_correct = 0
total = 0
with torch.no_grad():
    for features, labels in test_iter:
        features, labels = features.to(device), labels.to(device)
        outputs = net(features)  # 前向传播
        _, predicted = torch.max(outputs, 1)
        test_correct += (predicted == labels).sum().item()
        total += labels.numel()
    test_accuracy = test_correct / total

print(f"Average Accuracy: {test_accuracy:.4f}")
