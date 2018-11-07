from typing import List, Tuple
Piece = Tuple[str, str]

class ChessSquare(object):
    '''
    A container class for chess pieces.
    '''
    def __init__(self, rank : int, file : str) -> None:
        self.is_empty = True
        self.p_color = ""
        self.p_type = ""
        self.en_passant_left = False
        self.en_passant_right = False
        self.original_position = True
        
    def place_piece(self, p_color : str, p_type : str) -> None:
        self.p_color = p_color
        self.p_type = p_type
        
    def get_piece(self) -> Piece: 
        return (self.p_color, self.p_type)
        
    def clear_space(self) -> None: 
        self.p_color = ""
        self.p_type = ""
        
    def __str__(self) -> str : 
        if self.p_type == "":
            return "\t-"
        return "{}-{}".format(self.p_color, self.p_type)
                