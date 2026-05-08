import chess

PIECE_VALUES = {
    chess.PAWN: 10,
    chess.KNIGHT: 30,
    chess.BISHOP: 30,
    chess.ROOK: 50,
    chess.QUEEN: 90,
    chess.KING: 9000
}

def evaluate_board(board):
    # Positive: White advantage, Negative: Black advantage
    if board.is_checkmate():
        if board.turn == chess.WHITE:
            return -9999 # Black wins
        else:
            return 9999  # White wins

    if board.is_stalemate() or board.is_insufficient_material():
        return 0 # Draw

    evaluation = 0
    for piece_type in PIECE_VALUES:
        evaluation += len(board.pieces(piece_type, chess.WHITE)) * PIECE_VALUES[piece_type]
        evaluation -= len(board.pieces(piece_type, chess.BLACK)) * PIECE_VALUES[piece_type]

    return evaluation