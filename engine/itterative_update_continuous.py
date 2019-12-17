import os, sys, torch
from uuid import uuid1
import numpy as np
sys.path.append("/".join(os.getcwd().split("/")[:-1])+"/" )
from remy import play_game
from torch_net import train_on_one, Engine

def main(n_rounds, b_train_draw=True):
    model = Engine()
    path = os.getcwd() + "/saved_models/"
    model_dicts = [path + x for x in os.listdir(path)]
    model_dicts.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    model.load_state_dict(torch.load(model_dicts[0]))
    model.eval()
    n_moves = []
    n_white, n_black, n_draw = 0,0,0
    for j in range(n_rounds):
        print("\n\nPlay game {}".format(j))
        t_in, wt, n_turns = play_game(model, human_player=False, ret_tensor=True, save_path="../games/")

        if wt == 'white win':
            n_white += 1
            target = torch.ones(t_in.shape[0], 1) 
        elif wt == 'black win':
            n_black += 1
            target = torch.zeros(t_in.shape[0], 1)
        else:
            n_draw += 1
            if b_train_draw is False:
                continue
            target = .5*torch.ones(t_in.shape[0], 1)
        print("*-*-*\nRetrain Network")
        print("Number of wins for white {}, black {}, draws {}".format(n_white, n_black, n_draw))
        n_moves.append(n_turns)
        if j !=0 : print("Average number of moves {0:.3f} +/- {1:.3f}".format(np.mean(n_moves), np.std(n_moves)))
        train_on_one(model, t_in, target)

    torch.save(model.state_dict(), os.getcwd() + "/saved_models/"  + str(uuid1()) + ".pt")

if __name__ == "__main__":
    main(200, b_train_draw=False)
