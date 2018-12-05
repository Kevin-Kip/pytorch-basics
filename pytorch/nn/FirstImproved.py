import torch.nn.functional as F
from torch import nn

class FirstImproved(nn.Module):
    def __init__(self):
        super().__init__()

        self.hidden = nn.Linear(784, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        x = F.sigmoid(self.hidden(x))
        x = F.softmax(self.output(x))

        return x
