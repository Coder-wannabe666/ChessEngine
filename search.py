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

def get_best_move(board, depth):
    best_move = None
    alpha = -float('inf')
    beta = float('inf')

    is_maximizing = board.turn == chess.WHITE
    best_eval = -float('inf') if is_maximizing else float('inf')

    for move in board.legal_moves:
        board.push(move)
        eval = minimax(board, depth - 1, alpha, beta, not is_maximizing)
        board.pop()

        if is_maximizing:
            if eval > best_eval:
                best_eval = eval
                best_move = move
            alpha = max(alpha, eval)
        else:
            if eval < best_eval:
                best_eval = eval
                best_move = move
            beta = min(beta, eval)

    return best_move