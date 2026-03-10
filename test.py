import torch
from CNN import CNN
from data_proc import device,num_classes
from NN import NN


x = torch.randn(32,1,28,28).to(device)
#model = CNN(num_classes).to(device)
print(x.shape)

model = NN(num_classes).to(device)

output = model(x)
print(output.shape)
