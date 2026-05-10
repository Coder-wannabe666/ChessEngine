import chess
import torch
import numpy as np

def board_to_tensor(board):
    # Create 14x8x8 tensor filled with zeros
    tensor = np.zeros((14, 8, 8), dtype=np.float32)

    # Piece mapping to layer indices
    piece_idx = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }

    # Fill first 12 layers with pieces
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row = 7 - (square // 8)
            col = square % 8
            layer = piece_idx[piece.symbol()]
            tensor[layer][row][col] = 1.0

    # Layer 12: Turn indicator (1 for White, 0 for Black)
    if board.turn == chess.WHITE:
        tensor[12, :, :] = 1.0

    # Layer 13: Legal moves map
    for move in board.legal_moves:
        to_row = 7 - (move.to_square // 8)
        to_col = move.to_square % 8
        tensor[13][to_row][to_col] = 1.0

    return torch.from_numpy(tensor)