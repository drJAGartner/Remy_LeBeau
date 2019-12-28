import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import sigmoid, optim
from random import shuffle
import os, engine
from time import time
from uuid import uuid1

class Engine(nn.Module):
    '''
    Engine - an engine for predicting the probability that white wins a 
    chess game, given a vectorized representation of a board state.
    '''
    def __init__(self):
        super(Engine, self).__init__()
        self.fc1 = nn.Linear(8*8*12, 320)  # 8*8 board, 12 piece types
        self.fc2 = nn.Linear(320, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        # TODO: try leaky_relu instead
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = sigmoid(self.fc3(x))
        return x

def games_generator():
    base_path = os.path.dirname(engine.torch_net.__file__)
    games_path = base_path + "../games/"
    l_games = [(x, 0) for x in os.listdir(games_path + "black_wins/")]
    l_games = l_games + [(x, 1) for x in os.listdir(games_path+"white_wins/")]
    l_games = l_games + [(x, .5) for x in os.listdir(games_path+"draws/")]
    shuffle(l_games)
    for game_file in l_games:
        path = games_path + "black_wins/" 
        if game_file[1]==1:
            path = games_path + "white_wins/"
        elif game_file[1]==.5:
            path = games_path + "draws/"
        board_tensor = torch.load(path + game_file[0])
        target_tensor = torch.zeros(board_tensor.shape[0], 1) if game_file[1]==0 else game_file[1]*torch.ones(board_tensor.shape[0], 1)
        yield board_tensor, target_tensor

def train_on_all():
    en = Engine()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(en.parameters(), lr=0.001)
    n_epochs = 2000

    for epoch in range(n_epochs):
        games = games_generator()
        running_loss = 0.0
        i= 0
        start = time()
        for t_in, target in games:
            # in your training loop:
            optimizer.zero_grad()   # zero the gradient buffers
            output = en(t_in)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            # print statistics
            running_loss += loss.item()
            i += 1
        if epoch%100==0:
            print('Epoch %d loss: %.3f' % (epoch, running_loss / i))
            print('Time for this epoch: {}'.format(time()-start))


    torch.save(en.state_dict(), os.getcwd() + "/saved_models/"  + str(uuid1()) + ".pt")

def update_train(en, n_epochs=500, lr=.001):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(en.parameters(), lr=lr)
    
    for epoch in range(n_epochs):
        games = games_generator()
        running_loss = 0.0
        i= 0
        for t_in, target in games:
            # in your training loop:
            optimizer.zero_grad()   # zero the gradient buffers
            output = en(t_in)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            # print statistics
            running_loss += loss.item()
            i += 1
        if epoch%25==0:
            print('Epoch %d loss: %.3f' % (epoch, running_loss / i))

def train_on_one(en, t_in, target, n_epochs=5, lr=.005):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(en.parameters(), lr=lr)
    for epoch in range(n_epochs):
        optimizer.zero_grad()   # zero the gradient buffers
        output = en(t_in)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    train_on_all()