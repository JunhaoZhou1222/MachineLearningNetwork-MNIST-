import torch.nn as nn
import torch.optim as optim
from data_proc import train_dataloader, train_datasets, device, num_epoch,num_classes
import torch
print(torch.__version__)
class CNN(nn.Module):
    def __init__(self,num_classes):
        super(CNN,self).__init__()
        self.conv1 = nn.Sequential(         #input: 1*28*28
            nn.Conv2d(
            in_channels = 1,
            out_channels = 64,
            kernel_size = 3,
            stride = 1,
            padding = 1),                   # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
                                            #output: 64*28*28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)    #output after maxpool: 64*14*14
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64,128,3,1,1),        #input: 64*14*14
            nn.ReLU(),                      #output: 128*14*14
            nn.MaxPool2d(kernel_size=2)    #output after maxpool: 128*7*7
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128,256,3,1,1),       #input: 128*7*7
            nn.ReLU(),                      #output:  256*7*7
            nn.MaxPool2d(kernel_size=2)     #output after maxpool: 256*3*3
        )

        self.flatten = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*3*3, num_classes)
        )

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        return x
    
net = CNN(num_classes)
print(net)

#model = CNN().to(device)

#criterion = nn.CrossEntropyLoss()           #定义Loss
#optimizer = optim.Adam(model.parameters, lr=0.001) #定义优化器

#for epoch in range(num_epoch):
