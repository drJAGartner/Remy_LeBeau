import os, sys, torch
from uuid import uuid1
sys.path.append("/".join(os.getcwd().split("/")[:-1])+"/" )
from remy import play_game
from torch_net import update_train, Engine

def main():
    model = Engine()
    path = os.getcwd() + "/saved_models/"
    model_dicts = [path + x for x in os.listdir(path)]
    model_dicts.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    model.load_state_dict(torch.load(model_dicts[0]))
    model.eval()
    for j in range(20):
        print("\n\n******")
        print("Round {}".format(j))
        for i in range(100):
            print("Play game {}".format(i))
            play_game(model, computer_white=False, human_player=False, save_path="../games/")
        print("End of play for round ", j)
        print("\n\n*-*-*\nRetrain Network")
        update_train(model, n_epochs=201)
        torch.save(model.state_dict(), os.getcwd() + "/saved_models/"  + str(uuid1()) + ".pt")

if __name__ == "__main__":
    main()
