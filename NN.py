import torch.nn as nn
import torch.nn.functional as F
from data_proc import num_classes


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
 
net = NN()
print(net)