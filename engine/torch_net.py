import torch.nn as nn
import torch.nn.functional as F
from torch import sigmoid, 
from torch import optim
import sys
sys.path.append("../")

class Engine(nn.Module):
    '''
    Engine - an engine for predicting the probability that white wins a 
    chess game, given a vectorized representation of a board state.
    '''
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(8*8*12, 320)  # 8*8 board, 12 piece types
        self.fc2 = nn.Linear(320, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = sigmoid(self.fc3(x))
        return x



