import torch
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import DataLoader


#用 MNIST 内置数据集
input_size = 28
num_classes = 10
num_epoch = 3
batch_size = 64

device= "cuda" if torch.cuda.is_available() else "cpu"
print(device)

train_datasets = datasets.MNIST(
    root = './data',
    train = True,
    download = True,
    transform = transforms.ToTensor()
)

test_datasets = datasets.MNIST(
    root = './data',
    train = False,
    download = True,
    transform = transforms.ToTensor()
)

train_dataloader = DataLoader(
    dataset=train_datasets,
    batch_size=64,
    shuffle=True
)

test_dataloader = DataLoader(
    dataset=test_datasets,
    batch_size=64,
    shuffle=True
)
