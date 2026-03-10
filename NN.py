import torch.nn as nn
import torch.nn.functional as F
from data_proc import num_classes, device, num_epoch, train_dataloader,test_dataloader
import torch.optim as optim

class NN(nn.Module):
    def __init__(self):
        super(NN,self).__init__()
        self.hidden1 = nn.Linear(784, 128)
        self.hidden2 = nn.Linear(128, 256)
        self.hidden3 = nn.Linear(256, 512)
        self.out = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.hidden1(x))
        x = self.dropout(x)
        x = F.relu(self.hidden2(x))
        x = self.dropout(x)
        x = F.relu(self.hidden3(x))
        x = self.dropout(x)
        x = self.out(x)
        return x
 
#net = NN()
#print(net)

model = NN().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epoch):
    print("开始训练")
    model.train()
    train_loss = 0
    for image, label in train_dataloader:
        image, label = image.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    print("开始验证")
    model.eval()
    correct = 0
    for image, label in test_dataloader:
        image, label = image.to(device), label.to(device)
        output = model(image)
        correct += (output.argmax(1) == label).sum().item()

    print(f"训练轮数: {epoch+1}/{num_epoch}")
    print(f"训练损失: {train_loss/len(train_dataloader):.2f}")
    print(f"准确值: {correct/len(test_dataloader.dataset):.2f}")