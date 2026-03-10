import torch.nn as nn
import torch.optim as optim
from data_proc import train_dataloader, train_datasets, device, num_epoch,num_classes, test_dataloader
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
    
#net = CNN(num_classes)
#print(net)

model = CNN(num_classes).to(device)

criterion = nn.CrossEntropyLoss()           #定义Loss
optimizer = optim.Adam(model.parameters(), lr=0.001) #定义优化器

for epoch in range(num_epoch):                                  #训练几轮，epoch=3就是训练全部的数据集3次
    print("开始训练")
    model.train()                                               #训练模式
    train_loss = 0                                              #初始化损失
    for image, label in train_dataloader:                       #每次从数据集中取出一个图片和他的标签
        image, label = image.to(device), label.to(device)       #放进GPU

        optimizer.zero_grad()                                   #清空上次训练的梯度
        output = model(image)                                   #得到这张图片的预测结果
        loss = criterion(output, label)                         #计算预测结果和真实结果的损失
        loss.backward()                                         #根据损失，反向计算每个参数应该怎么调整
        optimizer.step()                                        #按照上一步计算的结果，真正去更新模型的参数
        train_loss += loss.item()                               #把tensor转为数字

    print("开始验证")
    model.eval()
    with torch.no_grad():                                       # 不计算梯度，节省内存，验证时不需要更新参数
        correct = 0                                             # 初始化正确数量为0
        for image, label in test_dataloader:                    # 每次从测试集取出一批图片和标签
            image, label = image.to(device), label.to(device)
            output = model(image)                               # 得到模型预测结果（10个类别的得分
            correct += (output.argmax(1) == label).sum().item() # 取得分最高的类别和真实标签对比，统计正确数量并累加

    print(f"Epoch {epoch+1}/{num_epoch} | "
          f"训练损失: {train_loss/len(train_dataloader):.2f} | "
          f"准确值: {100.*correct/len(test_dataloader.dataset):.2f}%")