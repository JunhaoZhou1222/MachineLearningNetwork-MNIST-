import torch
import torchvision
import torchvision.transforms as transform
from transformers import models


data_transform = {
    'train':
        transform.Compose([
        transform.Resize([128,128]), 
        transform.RandomRotation(45), #随机旋转（-45 - 45度）
        transform.CenterCrop(64), #从中心开始裁剪
        transform.RandomHorizontalFlip(p=0.5), #水平翻转，选择一个概率
        transform.RandomVerticalFlip(p=0.5), #垂直翻转， 选择一个概率
        transform.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1), #亮度， 对比度， 饱和度， 色相
        transform.RandomGrayscale(p=0.025), #概率转换为灰度
        transform.ToTensor(),
        transform.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) #均值， 标准差
    ]),
    'valid':
        transform.Compose([
        transform.Resize([64,64]),
        transform.ToTensor(),
        transform.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

model = models.resnet18
