import chess
import torch
from nn.model import ChessNet
from nn.encoder import board_to_tensor
from evaluation import evaluate_board as classical_eval

# Piece values for move ordering (MVV-LVA)
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

def evaluate_board_nn(board):
    # Terminal states
    if board.is_checkmate():
        return -99999 if board.turn == chess.WHITE else 99999
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    # Convert board state to tensor and add batch dimension
    board_tensor = board_to_tensor(board).unsqueeze(0).to(device)

    with torch.no_grad():
        # Get AI prediction (-1.0 to 1.0 -> -1000 to 1000)
        ai_eval = model(board_tensor).item() * 1000

    # Get classical evaluation
    classic_eval = classical_eval(board)

    # Hybrid Evaluation: 90% Classic, 10% AI
    return (classic_eval * 0.9) + (ai_eval * 0.1)

def score_move(board, move):
    # Calculate base score for move sorting
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
    # Evaluate current position using Neural Net
    stand_pat = evaluate_board_nn(board)

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

    # Generate and order capturing moves only
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
    # Drop into quiescence search
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
    # Find the top moves for the UI panel
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