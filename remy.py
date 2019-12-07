import chess, argparse, os
import numpy as np
import torch
from torch import Tensor
from torch import cat, zeros, ones
from uuid import uuid1
import sys
sys.path.append("./engine")
from torch_net import Engine

class Remy:
    def __init__(self, model=None, depth=0):
        self.board = chess.Board()
        self.piece_to_list = {
            "p":[1] + 5*[0] + 6*[0],
            "n":[0] + [1] + 4*[0] + 6*[0],
            "b":2*[0] + [1] + 3*[0] + 6*[0],
            "r":3*[0] + [1] + 2*[0] + 6*[0],
            "q":4*[0] + [1] + [0] + 6*[0],
            "k":5*[0] + [1] + 6*[0],
            "P":6*[0] + [1] + 5*[0],
            "N":6*[0] + [0] + [1] + 4*[0],
            "B":6*[0] + 2*[0] + [1] + 3*[0],
            "R":6*[0] + 3*[0] + [1] + 2*[0],
            "Q":6*[0] + 4*[0] + [1] + [0],
            "K":6*[0] + 5*[0] + [1],
            ".":12*[0]
        }
        self.model = model
        self.depth = depth

    def __str__(self) -> str:
        return self.board.__str__()

    def board_to_t(self) -> type(Tensor()): 
        '''
        board_to_t = Create a 8*8*12 tensor for the current board state
        '''
        v_board = []
        for pc in str(self.board).split():
            if pc=="\n":
                continue
            else:
                v_board.extend(self.piece_to_list[pc])
        return Tensor(v_board).reshape(1,-1)

    def explore_moves(self, depth, best_n=5):
        if depth == 0:
            move_p = []
            for move in self.board.legal_moves:
                self.board.push_uci(str(move))
                move_p.append((str(move), float(self.model(self.board_to_t())[0][0])))
                self.board.pop()
            return max(move_p)
        else:
            # First explore all legal moves from this position
            potential_moves = []
            for move in self.board.legal_moves:
                self.board.push_uci(str(move))
                potential_moves.append((str(move), float(self.model(self.board_to_t())[0][0])))
                self.board.pop()
            # Once moves are explored, select the best_n moves to 
            # recursively investigate
            move_p = {}
            for move, p in sorted(potential_moves, key=lambda x: x[1], reverse=self.board.turn):
                self.board.push_uci(move)
                move_p[move] = self.explore_move(depth-1)
                self.board.pop()
            return move_p

    def computer_turn(self):
        '''
        computer_turn - the mechanics for a computer turn

        '''
        if self.model is None:
            self.board.push_uci(str(np.random.choice(list(self.board.legal_moves))))
        else:
            # turn is self.board.turn == True if it is whites turn, black otherwise
            move_p = self.explore_moves(self.depth)
                
                
        return self.board_to_t(), self.board.is_game_over()

    def human_turn(self):
        '''
        human_turn - the mechanics for a human turn
        '''
        print(self)

        print("\n\nLegal Moves:")
        print(sorted([str(x) for x in self.board.legal_moves]))

        b_legal = False
        while b_legal is False:
            move = input("Select Move: ")
            if move in [str(x) for x in self.board.legal_moves]:
                b_legal = True
            else:
                print("That's not a legal move")

        self.board.push_uci(move)
        return self.board_to_t(), self.board.is_game_over()

    def end_type(self):
        '''
        end_type - the type of ending to the match

        returns either 'white', 'black', 'draw'
        '''
        if self.board.is_checkmate() is False:
            return 'draw'
        if self.board.turn is True:
            return 'black win'
        return 'white win'


def main(model, computer_white=False, human_player=True):
    # Create board, print initial state
    gambit = Remy(model=model)

    b_over, n_turns = False, 0
    while b_over is False:
        if (gambit.board.turn == computer_white):
            bt, b_over = gambit.computer_turn()
        else: 
            bt, b_over = gambit.human_turn()

        if n_turns == 0:
            board_tensor = bt
        else:
            board_tensor = cat((board_tensor, bt), 0)

        n_turns += 1
        if n_turns%50==0:
            print("Turn ", n_turns)

    print("This game took {} turns".format(n_turns))
    print(board_tensor.shape)

    # Print new board state
    print(gambit)

    wt = gambit.end_type()
    print("Game ends in a {}".format(wt))
    if wt != 'draw':
        if wt == 'white win':
            torch.save(board_tensor, "./games/white_wins/"+str(uuid1())+".pt")
        else:
            torch.save(board_tensor, "./games/black_wins/"+str(uuid1())+".pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-white', type=bool, default=False, help='Make the computer player move the white pieces')
    parser.add_argument('-model', type=str, default='')
    args = parser.parse_args()
    model = None
    if args.model == 'latest':
        model = Engine()
        path = os.getcwd() + "/engine/saved_models/"
        model_dicts = [path + x for x in os.listdir(path)]
        model_dicts.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        model.load_state_dict(torch.load(model_dicts[0]))
        model.eval()

    main(model, computer_white=args.white)