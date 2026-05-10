import chess
import torch
from functools import lru_cache
from nn.model import ChessNet
from nn.encoder import board_to_tensor
from evaluation import evaluate_board as classical_eval

# Piece values for move ordering
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000
}

# Load the trained Neural Network
device = torch.device("cpu")
model = ChessNet()
model.load_state_dict(torch.load("nn/chess_model.pth", map_location=device, weights_only=True))
model.eval()

@lru_cache(maxsize=100000)
def get_ai_prediction(fen):
    # Fast caching for AI evaluations based on FEN strings
    # This prevents calculating the same tensor twice and massively speeds up the engine
    board = chess.Board(fen)
    board_tensor = board_to_tensor(board).unsqueeze(0).to(device)
    with torch.no_grad():
        # Scale AI output to centipawns (calibrated to max 500 to avoid extreme values)
        return model(board_tensor).item() * 500

def evaluate_node(board):
    # Terminal states
    if board.is_checkmate():
        return -99999 if board.turn == chess.WHITE else 99999
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    # Get both evaluations
    classic_eval = classical_eval(board)
    ai_eval = get_ai_prediction(board.fen())

    # HYBRID EVALUATION: 30% AI / 70% Classical Algorithm
    return (classic_eval * 0.7) + (ai_eval * 0.3)

def score_move(board, move):
    # Base score for move sorting (MVV-LVA)
    score = 0
    if board.is_capture(move):
        attacker = board.piece_at(move.from_square)
        if board.is_en_passant(move):
            victim_val = PIECE_VALUES[chess.PAWN]
        else:
            victim = board.piece_at(move.to_square)
            victim_val = PIECE_VALUES.get(victim.piece_type, 0) if victim else 0

        attacker_val = PIECE_VALUES.get(attacker.piece_type, 0) if attacker else 0
        score += 10 * victim_val - attacker_val

    if move.promotion:
        score += PIECE_VALUES.get(move.promotion, 0)
    return score

def order_moves(board, moves):
    # Sort moves prioritizing captures
    return sorted(moves, key=lambda m: score_move(board, m), reverse=True)

def quiescence_search(board, alpha, beta, maximizing_player):
    # Stand pat using our hybrid AI evaluation
    stand_pat = evaluate_node(board)

    if maximizing_player:
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat
    else:
        if stand_pat <= alpha:
            return alpha
        if beta > stand_pat:
            beta = stand_pat

    # Explore captures
    captures = [move for move in board.legal_moves if board.is_capture(move)]
    captures = order_moves(board, captures)

    if maximizing_player:
        max_eval = stand_pat
        for move in captures:
            board.push(move)
            eval = quiescence_search(board, alpha, beta, False)
            board.pop()
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = stand_pat
        for move in captures:
            board.push(move)
            eval = quiescence_search(board, alpha, beta, True)
            board.pop()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

def minimax(board, depth, alpha, beta, maximizing_player):
    if depth == 0 or board.is_game_over():
        return quiescence_search(board, alpha, beta, maximizing_player)

    moves = order_moves(board, list(board.legal_moves))

    if maximizing_player:
        max_eval = -float('inf')
        for move in moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for move in moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

def get_top_moves(board, depth, count=3):
    # Core function to retrieve best moves for the UI
    moves_with_evals = []
    alpha = -float('inf')
    beta = float('inf')
    is_maximizing = board.turn == chess.WHITE

    moves = order_moves(board, list(board.legal_moves))

    for move in moves:
        board.push(move)
        eval = minimax(board, depth - 1, alpha, beta, not is_maximizing)
        board.pop()
        moves_with_evals.append((eval, move))

    moves_with_evals.sort(key=lambda x: x[0], reverse=is_maximizing)
    return moves_with_evals[:count]