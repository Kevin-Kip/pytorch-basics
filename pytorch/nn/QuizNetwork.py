from torch import nn
import torch.nn.functional as F

class QuizNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.hidden1 = nn.Linear(784, 128)
        self.hidden2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 10)

    def forward(self, x):
        x = self.hidden1(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = F.relu(x)
        x = self.output(x)
        x = self.softmax(x, dim=1)

        return x

m = QuizNetwork()
# shows the weights and bias of hidden layer 1
print(m.hidden1.bias)
print(m.hidden1.weight)
