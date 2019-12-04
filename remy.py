import chess, argparse
import numpy as np
import torch
from torch import Tensor
from torch import cat, zeros, ones

from uuid import uuid1

class Remy:
    def __init__(self, model=None):
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

    def computer_turn(self):
        '''
        computer_turn - the mechanics for a computer turn

        params -
        '''
        if self.model is None:
            self.board.push_uci(str(np.random.choice(list(self.board.legal_moves))))
        else:
            for move in self.board.legal_moves:
                self.board.push_uci(str(move))
                # turn is self.board.turn == True if it is whites turn, black otherwise
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


def main(computer_white=False, human_player=True):
    # Create board, print initial state
    gambit = Remy()

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
    args = parser.parse_args()
    
    main(computer_white=args.white)