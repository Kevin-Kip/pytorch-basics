import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

def softmax(x):
    return torch.exp(x) / torch.sum(torch.exp(x), dim=1).view(-1,1)

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

trainset = datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,shuffle=True)

dataiter = iter(trainloader)
images, labels = dataiter.next()

inputs = images.view(images.shape[0], -1)

w1 = torch.randn(784,256)
b1 = torch.randn(256)

w2 = torch.randn(256,10)
b2 = torch.randn(10)

h = sigmoid(torch.mm(inputs, w1) + b1)

output = torch.mm(h, w2) + b2

probs = softmax(output)
