import torch
import matplotlib.pyplot as plt
import os
import torchvision

#@save
VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

#@save
VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

voc_dir = './data/VOCdevkit/VOC2012'

def show_images_with_labels(images, labels, num_images=5):
    """
    显示图像及其对应的标签。
    
    参数：
    - images: 图像的列表 (tensor 格式)
    - labels: 标签的列表 (tensor 格式)
    - num_images: 显示的图像数量
    """
    assert len(images) == len(labels), "图像和标签数量不一致"
    
    images = images[:num_images]
    labels = labels[:num_images]
    
    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 3, 6))
    
    for i in range(num_images):
        img = images[i].permute(1, 2, 0)  # 转换为 [height, width, channels]
        label = labels[i].permute(1, 2, 0)  # 转换为 [height, width, channels]
        axes[0, i].imshow(img)
        axes[0, i].axis('off')
        axes[1, i].imshow(label)
        axes[1, i].axis('off')
        
        if i == 0:
            axes[0, i].set_title("Image")
            axes[1, i].set_title("Label")

    plt.tight_layout()
    plt.show()

def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (torch.arange(kernel_size).reshape(-1, 1),
          torch.arange(kernel_size).reshape(1, -1))
    filt = (1 - torch.abs(og[0] - center) / factor) * \
           (1 - torch.abs(og[1] - center) / factor)
    weight = torch.zeros((in_channels, out_channels,
                          kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return weight

#@save
def voc_colormap2label():
    """构建从RGB到VOC类别索引的映射"""
    colormap2label = torch.zeros(256 ** 3, dtype=torch.long)
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[
            (colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
    return colormap2label

#@save
def voc_label_indices(colormap, colormap2label):
    """将VOC标签中的RGB值映射到它们的类别索引"""
    colormap = colormap.permute(1, 2, 0).numpy().astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2])
    return colormap2label[idx]

#@save
def voc_rand_crop(feature, label, height, width):
    """随机裁剪特征和标签图像"""
    rect = torchvision.transforms.RandomCrop.get_params(
        feature, (height, width))
    feature = torchvision.transforms.functional.crop(feature, *rect)
    label = torchvision.transforms.functional.crop(label, *rect)
    return feature, label
#@save
def read_voc_images(voc_dir, is_train=True):
    """读取所有VOC图像并标注"""
    txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation',
                             'train.txt' if is_train else 'val.txt')
    mode = torchvision.io.image.ImageReadMode.RGB
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [], []
    for i, fname in enumerate(images):
        features.append(torchvision.io.read_image(os.path.join(
            voc_dir, 'JPEGImages', f'{fname}.jpg')))
        labels.append(torchvision.io.read_image(os.path.join(
            voc_dir, 'SegmentationClass' ,f'{fname}.png'), mode))
    return features, labels

#@save
class VOCSegDataset(torch.utils.data.Dataset):
    """一个用于加载VOC数据集的自定义数据集"""

    def __init__(self, is_train, crop_size, voc_dir):
        self.transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.crop_size = crop_size
        features, labels = read_voc_images(voc_dir, is_train=is_train)
        self.features = [self.normalize_image(feature)
                         for feature in self.filter(features)]
        self.labels = self.filter(labels)
        self.colormap2label = voc_colormap2label()
        print('read ' + str(len(self.features)) + ' examples')

    def normalize_image(self, img):
        return self.transform(img.float() / 255.)

    def filter(self, imgs):
        return [img for img in imgs if (
            img.shape[1] >= self.crop_size[0] and
            img.shape[2] >= self.crop_size[1])]

    def __getitem__(self, idx):
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx],
                                       *self.crop_size)
        return (feature, voc_label_indices(label, self.colormap2label))

    def __len__(self):
        return len(self.features)
    
#@save
def load_data_voc(batch_size, crop_size):
    """加载VOC语义分割数据集"""
    train_iter = torch.utils.data.DataLoader(
        VOCSegDataset(True, crop_size, voc_dir), batch_size,
        shuffle=True, drop_last=True)
    test_iter = torch.utils.data.DataLoader(
        VOCSegDataset(False, crop_size, voc_dir), batch_size,
        drop_last=True)
    return train_iter, test_iter

def show_images(imgs):
    """展示原图、预测图和标签图。
    
    参数:
    imgs -- 包含每组原图、预测图和标签图的列表，每三张为一组
    """
    num_images = len(imgs) // 3  # 计算要展示的图组数量，每组三张图：原图、预测图、标签图
    
    fig, axes = plt.subplots(num_images, 3, figsize=(10, 3 * num_images))
    if num_images == 1:
        axes = [axes]  # 处理只有一行的情况

    for i in range(num_images):
        # 原图
        axes[i][0].imshow(imgs[3 * i].numpy())
        axes[i][0].set_title("Original Image")
        axes[i][0].axis("off")
        
        # 预测图
        axes[i][1].imshow(imgs[3 * i + 1].numpy())
        axes[i][1].set_title("Predicted Label")
        axes[i][1].axis("off")
        
        # 标签图
        axes[i][2].imshow(imgs[3 * i + 2].numpy())
        axes[i][2].set_title("Ground Truth")
        axes[i][2].axis("off")
    
    plt.tight_layout()
    plt.show()

def calculate_accuracy(pred, labels):
    """
    计算预测的准确率。

    参数:
    pred -- 模型的预测结果，形状为 (batch_size, height, width)
    labels -- 真实标签，形状为 (batch_size, height, width)

    返回:
    accuracy -- 准确率，值在 0 到 1 之间
    """
    # 将预测值转换为标签索引（argmax 输出为单通道类别索引）

    # 比较预测标签和真实标签
    correct = (pred == labels).sum().item()  # 计算正确的像素数
    total = torch.numel(labels)  # 计算所有像素的总数

    # 计算准确率
    accuracy = correct / total
    return accuracy
