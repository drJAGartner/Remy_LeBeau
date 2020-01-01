import chess, argparse, os
import numpy as np
import torch
from torch import Tensor
from torch import cat, zeros, ones
from uuid import uuid1
import sys
from engine.torch_net import Engine


class Remy:
    def __init__(self, model=None, depth=0, move_thresh=0.7):
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
        self.current_graph = None
        self.move_thresh = move_thresh

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

    def end_probabilities(self) -> float:
        '''
        end_probabilities = return win probabilities for rules based wins
        three posibilities are white win via checkmate, black win via checkmate, or draw
        '''
        end = self.end_type()
        if end == 'draw':
            return .5
        else:
            if end == 'white win':
                return 1.
            else:
                return 0.
        
    def engine_probabilities(self) -> float:
        '''
        engine_probabilities = return the NN assessment of the win probability of a board state
        '''
        return float(self.model(self.board_to_t())[0][0])
            

    def computer_turn(self):
        '''
        computer_turn - the mechanics for a computer turn

        '''
        if self.model is None:
            self.board.push_uci(str(np.random.choice(list(self.board.legal_moves))))
        else:
            move_ps = []
            # get win probabilitilities for all moves
            for move in self.board.legal_moves:
                # make the move
                self.board.push_uci(str(move))

                # assess if the move ends the game
                if self.board.is_game_over() is True:
                    # if there is a checkmate in favor of the computer, make this move
                    end_p = self.end_probabilities()
                    if bool(end_p) != self.board.turn:
                        return self.board_to_t(), self.board.is_game_over()
                    move_ps.append((str(move), end_p))
                else:
                    move_ps.append((str(move), self.engine_probabilities()))
            
                # undo the move
                self.board.pop()

            if self.board.turn == False:
                move_ps = [(x[0], 1-x[1]) for x in move_ps]
            max_p = max([x[1] for x in move_ps])
            if max_p > 0.:
                move_ps = [x for x in move_ps if x[1]/max_p > self.move_thresh]
                total_p = sum([x[1] for x in move_ps])
                self.board.push_uci(np.random.choice([x[0] for x in move_ps], p=[x[1]/total_p for x in move_ps]))
            else:
                self.board.push_uci(np.random.choice([x[0] for x in move_ps]))
                
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


def play_game(model, move_thresh=0.7, computer_white=False, human_player=True, save_path="../games/", save_game=True, ret_tensor=False, verbose=True):
    '''
    play game - play a game of chess using a version of Remy

    params - 
    model (pytorch neural netowrk, nullable) - a pytorch Engine network
    move_thresh (float) - the win threshold for removing moves, see notes
    computer_white (bool) - for a human v computer game, if computer is moving for white
    human_player (bool) - if a human will be moving one of the sets of pieces
    save_path (str) - base path to save board tensors
    save_game (bool) - if the board tensor should be saved for the match
    ret_tensor - if the board tensor should be returned for the match
    verbose - print for maximal print output while running

    note on move_thresh -
    when calculating all legal moves, the system is not determinisitc, meaning it will choose a moved
    proportionally to the associated win probability of that move.  To move the engine forward, we disallow moves
    by first normalizing with respect to the best moves win probability, and filtering out all moves lower than move_thresh.

    For example if two moves have associated win probabilities for white of .8, .6 and .35 respectively, under the default threshold
    only .35 will be considered.  To have the computer play less randomly, you can raise this threshold.
    '''
    gambit = Remy(model=model)

    b_over, n_turns = False, 0
    while b_over is False:
        if (human_player is True) & (gambit.board.turn != computer_white):
            bt, b_over = gambit.human_turn()
        else: 
            bt, b_over = gambit.computer_turn()

        if n_turns == 0:
            board_tensor = bt
        else:
            board_tensor = cat((board_tensor, bt), 0)

        n_turns += 1
        if (n_turns%50==0) and (verbose is True):
            print("Turn ", n_turns)

    if verbose is True:
        print("This game took {} turns".format(n_turns))
        print(board_tensor.shape)
        print(gambit)

    wt = gambit.end_type()
    print("Game ends in a {}".format(wt))
    if save_game is True:
        if wt == 'draw':
            torch.save(board_tensor, save_path + "draws/"+str(uuid1())+".pt")
        elif wt == 'white win':
            torch.save(board_tensor, save_path + "white_wins/"+str(uuid1())+".pt")
        else:
            torch.save(board_tensor, save_path + "black_wins/"+str(uuid1())+".pt")
    
    if ret_tensor is True:
        return board_tensor, wt, n_turns

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-white', type=bool, default=False, help='Make the computer player move the white pieces')
    parser.add_argument('-model', type=str, default='')
    parser.add_argument('-save_path', type=str, default="../games/")
    parser.add_argument('--human_player', action='store_true')
    args = parser.parse_args()
    model = None
    if args.model == 'latest':
        model = Engine()
        path = os.getcwd() + "/engine/saved_models/"
        model_dicts = [path + x for x in os.listdir(path)]
        model_dicts.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        model.load_state_dict(torch.load(model_dicts[0]))
        model.eval()

    play_game(model, computer_white=args.white, human_player=args.human_player, save_path=args.save_path)