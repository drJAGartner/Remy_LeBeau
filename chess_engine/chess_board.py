import numpy as np
from copy import deepcopy
from chess_square import ChessSquare
from typing import List, Tuple
Square = Tuple[int, int]
MoveList = List[Square]

class ChessBoard(object):
    '''
    Chess Board object.  Used to represent the current game state, as well as 
    compiling the legal moves that can be made for the given board state.
    '''
    def __init__(self, toy_board=False): 
        if toy_board is True:
            self.files = ['A', 'B', 'C', 'D']
            self.board = np.array([[ChessSquare(rank, file) for file in self.files] for rank in range(1, 5)])
            for rank, color in zip([0, 3], ['white', 'black']):
                for file_num in range(4):
                    self.board[rank, file_num].place_piece(color, 'pawn')
        
        else:
            self.files = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
            self.board = np.array([[ChessSquare(rank, file) for file in self.files] for rank in range(1, 9)])
            for rank, color in zip([0, 1, 6, 7], ['white']*2 + ['black']*2):
                for file_num in range(8):
                    if rank == 1 or rank == 6:
                        self.board[rank, file_num].place_piece(color, 'pawn')
                    else:
                        if file_num in [0, 7]:
                            self.board[rank, file_num].place_piece(color, 'rook')
                        if file_num in [1, 6]:
                            self.board[rank, file_num].place_piece(color, 'knight')
                        if file_num in [2, 5]:
                            self.board[rank, file_num].place_piece(color, 'bishop')
                        if file_num == 3:
                            self.board[rank, file_num].place_piece(color, 'queen')
                        if file_num == 4:
                            self.board[rank, file_num].place_piece(color, 'king')

            
    def __str__(self) -> str:
        ret = ""
        for rank in range(len(self.board)-1, -1 , -1):
            ret = ret + "\n" + str(rank+1)
            for file in range(len(self.board)):
                ret = ret + "\t" + str(self.board[rank, file])
        
        ret = ret + "\n\t\t" + "\t\t".join(self.files)
        return ret
    
    @staticmethod
    def what_is_forward(color : str) -> int :
        if color == 'white':
            return 1
        else:
            return -1
    
    def piece_legal_moves(self, source : tuple) -> MoveList:
        moves = list()

        square = self.board[(source)]
        if square.p_type == "":
            return moves

        if square.p_type == 'pawn':
            forward = self.what_is_forward(square.p_color)
            moves.extend(self.pawn_transposition(square.p_color, source, forward))
            moves.extend(self.pawn_attacks(square.p_color, source, forward))
        
        return moves

    def pawn_transposition(self, color: str, source: Square, forward : int) -> MoveList:
        moves = list()
        if self.board[source[0]+forward, source[1]].p_type == "":
            if self.makes_check(source, (source[0]+forward, source[1])) is False :
                moves.append((source[0]+forward, source[1]))
            if self.board[source[0]+2*forward, source[1]].p_type == "" \
                and self.board[source].original_position == True \
                and self.makes_check(source, (source[0]+2*forward, source[1])) is False :
                moves.append((source[0]+2*forward, source[1]))
        return moves

    def pawn_attacks(self, color: str, source: Square, forward : int) ->  MoveList:
        moves = list()
        if source[1] + 1 < self.board.shape[1] \
            and self.board[source[0]+forward, source[1] + 1].p_color != "" \
            and self.board[source[0]+forward, source[1] + 1].p_color != color \
            and self.makes_check(source, (source[0]+forward, source[1]+1)) is False :
            moves.append((source[0]+forward, source[1] + 1))
        if source[1] - 1 >= 0 \
            and self.board[source[0]+forward, source[1] - 1].p_color != "" \
            and self.board[source[0]+forward, source[1] - 1].p_color != color \
            and self.makes_check(source, (source[0]+forward, source[1]-1)) is False:
            moves.append((source[0]+forward, source[1] - 1))
        if self.board[source].en_passant_left == True \
            and self.makes_check(source, (source[0]+forward, source[1]-1)) is False:
            moves.append((source[0]+forward, source[1]-1))
            self.board[source].en_passant_left ==False
        if self.board[source].en_passant_right == True \
            and self.makes_check(source, (source[0]+forward, source[1]+1)) is False:
            moves.append((source[0]+forward, source[1]+1))
            self.board[source].en_passant_right ==False
        return moves
    
    def all_pieces(self, color : str) -> MoveList:
        '''
        :param: color - string, color of the pieces you want the list of locations for
        '''
        return [(rank, file) for rank in range(len(self.board)) \
                                        for file in range(len(self.board)) \
                                         if self.board[rank, file].p_color == color]
    
    def makes_check(self, source : tuple, dest : tuple ) -> bool:
        '''
        function to evaluate if a move makes a check
        
        :param: source, tuple representation of the space before the move
        :param: des, tuple representation of the space after the move
        '''
        # TODO -> FINISH THIS
        cb_temp = deepcopy(self)
        color, p_type = cb_temp.board[source].get_piece()
        cb_temp.board[source].clear_space()
        cb_temp.board[dest].place_piece(color, p_type)
        other = 'white' if color == 'black' else 'black'
        #for piece in cb_temp.all_pieces(other):
        #    for move in cb_temp.piece_legal_moves:
        #        pass

        del(cb_temp)
        return False
        
        
    
    def get_square(self, square : str ) -> Square:
        '''
        :param: square - string representation of square (e.g. 'b2')
        
        return - reference to chess square object
        '''
        if len(square) > 2:
            print("Udate format, must be a length 2 string like 'B2' (case insensitive)")
            return (1, 1)
        if square[0].upper() not in self.files:
            print("Square not found, {} not in files: \n{}".format(square[0].upper(), self.files))
            return (1, 1)
        return self.board[int(square[1])-1, self.files.index(square[0].upper())]
    
    def make_move(self, source : tuple, dest : tuple) -> None:
        '''
        :param: source - tuple representation of the rank and file source
        :param: dest - tuple representation of the rank and file destination
        '''
        pass
            
        