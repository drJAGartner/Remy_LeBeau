import os, sys, torch, engine
from uuid import uuid1
from player.remy import play_game
from torch_net import update_train, Engine

def main():
    model = Engine()
    base_path = os.path.dirname(engine.torch_net.__file__)
    model_dicts = [base_path + "/saved_models/" + x for x in os.listdir(base_path + "/saved_models/")]
    model_dicts.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    model.load_state_dict(torch.load(model_dicts[0]))
    model.eval()
    for j in range(20):
        print("\n\n******")
        print("Round {}".format(j))
        for i in range(100):
            print("Play game {}".format(i))
            play_game(model, computer_white=False, human_player=False, save_path=base_path + "/../games/")
        print("End of play for round ", j)
        print("\n\n*-*-*\nRetrain Network")
        update_train(model, n_epochs=51)
        torch.save(model.state_dict(), base_path + "/saved_models/"  + str(uuid1()) + ".pt")

if __name__ == "__main__":
    main()
