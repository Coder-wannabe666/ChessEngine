import chess

# Standard chess values in centipawns (1 pawn = 100)
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 300,
    chess.BISHOP: 300,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 90000
}

def evaluate_board(board):
    # Positive: White advantage, Negative: Black advantage
    if board.is_checkmate():
        if board.turn == chess.WHITE:
            return -99999 # Black wins
        else:
            return 99999  # White wins

    if board.is_stalemate() or board.is_insufficient_material():
        return 0 # Draw

    evaluation = 0
    for piece_type in PIECE_VALUES:
        evaluation += len(board.pieces(piece_type, chess.WHITE)) * PIECE_VALUES[piece_type]
        evaluation -= len(board.pieces(piece_type, chess.BLACK)) * PIECE_VALUES[piece_type]

    return evaluation