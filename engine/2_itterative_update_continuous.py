import os, sys, torch
from uuid import uuid1
import numpy as np
from player.remy import play_game
from torch_net import train_on_one, Engine

def main(n_rounds, b_train_draw=True):
    model = Engine()
    base_path = os.path.dirname(engine.torch_net.__file__)
    model_dicts = [base_path + "/saved_models/" + x for x in os.listdir(base_path + "/saved_models/")]
    model_dicts.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    model.load_state_dict(torch.load(model_dicts[0]))
    model.eval()
    n_moves = []
    n_white, n_black, n_draw = 0,0,0
    n_trainings = 0
    for j in range(n_rounds):
        print("\n\nPlay game {}".format(j))
        t_in, wt, n_turns = play_game(model, human_player=False, ret_tensor=True, save_path=base_path + "/../games/")

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
        n_trainings += 1
        print("*-*-*\nRetrain Network")
        print("Number of wins for white {}, black {}, draws {}".format(n_white, n_black, n_draw))
        n_moves.append(n_turns)
        if j !=0 : print("Average number of moves {0:.3f} +/- {1:.3f}".format(np.mean(n_moves), np.std(n_moves)))
        train_on_one(model, t_in, target)
        if n_trainings%300==0:
            print("Saving")
            torch.save(model.state_dict(), base_path + "/saved_models/"  + str(uuid1()) + ".pt")

    torch.save(model.state_dict(), base_path + "/saved_models/"  + str(uuid1()) + ".pt")

if __name__ == "__main__":
    main(10000, b_train_draw=False)
