import torch

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

torch.manual_seed(7)

features = torch.randn((1,6)) # random tensor(matrix)
                              # with one row and six colums

weights = torch.randn_like(features) # similar to features

bias = torch.randn((1,1))

y = sigmoid(torch.mm(features, weights.view(6,1)) + bias)
print(y)
