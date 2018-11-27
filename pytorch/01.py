import torch

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

torch.manual_seed(7)

features = torch.randn((1,6))

weights = torch.randn_like(features)

bias = torch.randn((1,1))

y = sigmoid(torch.mm(features, weights.view(6,1)) + bias)
print(y)
