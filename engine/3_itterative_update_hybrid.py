import os, sys, torch, engine
from uuid import uuid1
import numpy as np
from player.remy import play_game
from torch_net import update_train, train_on_one, Engine

def main(n_rounds, b_train_draw=True):
    model = Engine()
    base_path = os.path.dirname(engine.torch_net.__file__)
    model_dicts = [base_path + "/saved_models/" + x for x in os.listdir(base_path + "/saved_models/")]
    model_dicts.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    model.load_state_dict(torch.load(model_dicts[0]))
    model.eval()
    n_white, n_black, n_draw = [],[],[]
    n_trainings = 0
    for j in range(n_rounds):
        print("\n\nPlay game {}".format(j))
        t_in, wt, n_turns = play_game(model, human_player=False, ret_tensor=True, save_path=base_path + "/../games/")

        if wt == 'white win':
            n_white.append(n_turns)
            target = torch.ones(t_in.shape[0], 1) 
        elif wt == 'black win':
            n_black.append(n_turns)
            target = torch.zeros(t_in.shape[0], 1)
        else:
            n_draw.append(n_turns)
            if len(n_draw) % 200 == 0:
                print("Updating model on all data")
                update_train(model, n_epochs=76)
                print("Saving model")
                torch.save(model.state_dict(), base_path + "/saved_models/"  + str(uuid1()) + ".pt")
                n_trainings = 0
                print("Removing draw files")
                draw_files = os.listdir(base_path + "/../games/draws/")
                for draw_file in draw_files:
                    os.remove(base_path + "/../games/draws/" + draw_file)
            if b_train_draw is False:
                continue
            target = .5*torch.ones(t_in.shape[0], 1)
        n_trainings += 1
        print("*-*-*\nRetrain Network")
        print("Number of wins for white {}, black {}, draws {}".format(len(n_white), len(n_black), len(n_draw)))
        n_moves = n_white+n_black+n_draw
        if j > 50 : 
            print("Average number of moves {0:.3f} +/- {1:.3f}".format(np.mean(n_moves), np.std(n_moves)))
            print("\tfor white wins: {0:.3f} +/- {1:.3f}".format(np.mean(n_white), np.std(n_white)))
            print("\tfor black wins: {0:.3f} +/- {1:.3f}".format(np.mean(n_black), np.std(n_black)))
            print("\tfor draws: {0:.3f} +/- {1:.3f}".format(np.mean(n_draw), np.std(n_draw)))
        train_on_one(model, t_in, target)
        if j%500==0:
            print("Saving")
            torch.save(model.state_dict(), base_path + "/saved_models/"  + str(uuid1()) + ".pt")

    torch.save(model.state_dict(), base_path + "/saved_models/"  + str(uuid1()) + ".pt")

if __name__ == "__main__":
    main(10000, b_train_draw=False)
