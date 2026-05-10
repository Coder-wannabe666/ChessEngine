import chess

# Standard chess values in centipawns
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 300,
    chess.BISHOP: 300,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000
}

# Piece-Square Tables (PST)
# Visually mapped from A8 to H8
PAWN_PST = [
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0
]

KNIGHT_PST = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50
]

BISHOP_PST = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20
]

# Discourage King from wandering in the middle game
KING_PST = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20
]

def get_pst_value(piece_type, square, is_white):
    rank = chess.square_rank(square)
    file = chess.square_file(square)

    if is_white:
        visual_index = (7 - rank) * 8 + file
    else:
        # Flip the board for Black so they push pawns down
        visual_index = rank * 8 + file

    if piece_type == chess.PAWN: return PAWN_PST[visual_index]
    if piece_type == chess.KNIGHT: return KNIGHT_PST[visual_index]
    if piece_type == chess.BISHOP: return BISHOP_PST[visual_index]
    if piece_type == chess.KING: return KING_PST[visual_index]

    return 0

def evaluate_board(board):
    # Positive: White advantage, Negative: Black advantage
    if board.is_checkmate():
        return -99999 if board.turn == chess.WHITE else 99999

    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    eval_white = 0
    eval_black = 0

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            val = PIECE_VALUES.get(piece.piece_type, 0)
            pst_val = get_pst_value(piece.piece_type, square, piece.color == chess.WHITE)

            if piece.color == chess.WHITE:
                eval_white += val + pst_val
            else:
                eval_black += val + pst_val

    return eval_white - eval_black