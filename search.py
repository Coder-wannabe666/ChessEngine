import chess
from evaluation import evaluate_board

def minimax(board, depth, alpha, beta, maximizing_player):
    if depth == 0 or board.is_game_over():
        return evaluate_board(board)

    if maximizing_player:
        max_eval = -float('inf')
        for move in board.legal_moves:
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
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, True)
            board.pop()

            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

def get_top_moves(board, depth, count=3):
    # Returns a list of tuples: (evaluation_score, move)
    moves_with_evals = []
    alpha = -float('inf')
    beta = float('inf')

    is_maximizing = board.turn == chess.WHITE

    for move in board.legal_moves:
        board.push(move)
        eval = minimax(board, depth - 1, alpha, beta, not is_maximizing)
        board.pop()
        moves_with_evals.append((eval, move))

    # Sort moves: highest eval first for White, lowest first for Black
    moves_with_evals.sort(key=lambda x: x[0], reverse=is_maximizing)

    return moves_with_evals[:count]