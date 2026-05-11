import chess
import chess.polyglot
import torch
from functools import lru_cache
from nn.model import ChessNet
from nn.encoder import board_to_tensor
from evaluation import evaluate_board as classical_eval

# --- Configuration & Tuning ---
DEPTH = 4
AI_WEIGHT = 0.15
AI_MULTIPLIER = 120

# --- Global Caches & Counters ---
TRANSPOSITION_TABLE = {}
KILLER_MOVES = [[None] * 2 for _ in range(30)]
NODES_COUNT = 0

PIECE_VALUES = {
    chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
    chess.ROOK: 500, chess.QUEEN: 900
}

# --- Neural Network Setup ---
device = torch.device("cpu")
model = ChessNet()
model.load_state_dict(torch.load("nn/chess_model.pth", map_location=device, weights_only=True))
model.eval()

@lru_cache(maxsize=50000)
def get_ai_score(fen):
    """Deep intuition - only called at the end of the main search."""
    board = chess.Board(fen)
    tensor = board_to_tensor(board).unsqueeze(0).to(device)
    with torch.no_grad():
        return model(tensor).item() * AI_MULTIPLIER

def fast_eval(board, current_depth, use_ai=False):
    """Evaluation including mate distance penalties."""
    # Checkmate detection (with distance penalty)
    if board.is_checkmate():
        if board.turn == chess.WHITE:
            return -99999 + current_depth
        else:
            return 99999 - current_depth

    if board.is_stalemate():
        return 0

    score = classical_eval(board)

    if use_ai:
        ai_intuition = get_ai_score(board.fen())
        score = (score * (1 - AI_WEIGHT)) + (ai_intuition * AI_WEIGHT)

    return score

def move_value(board, move, current_depth):
    """Assigns a value to a move for sorting (MVV-LVA and Killer Moves)."""
    if board.is_capture(move):
        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)

        victim_val = PIECE_VALUES.get(victim.piece_type, 0) if victim else 0
        attacker_val = PIECE_VALUES.get(attacker.piece_type, 0) if attacker else 0

        if board.is_en_passant(move):
            victim_val = PIECE_VALUES[chess.PAWN]

        return 10000 + victim_val - attacker_val

    if move in KILLER_MOVES[current_depth]:
        return 5000

    return 0

def quiescence(board, alpha, beta, maximizing, current_depth, q_depth=0):
    """Fast tactical search evaluating only captures."""
    global NODES_COUNT
    NODES_COUNT += 1

    # STRICT DRAW DETECTION BEFORE ANYTHING ELSE
    if board.is_repetition(2) or board.is_fifty_moves() or board.is_insufficient_material():
        return 0

    stand_pat = fast_eval(board, current_depth, use_ai=False)

    # HARD LIMIT to prevent infinite CPU loops
    if q_depth >= 4:
        return stand_pat

    if maximizing:
        if stand_pat >= beta: return beta
        alpha = max(alpha, stand_pat)

        for move in sorted(board.legal_moves, key=lambda m: board.is_capture(m), reverse=True):
            if board.is_capture(move):
                board.push(move)
                alpha = max(alpha, quiescence(board, alpha, beta, False, current_depth + 1, q_depth + 1))
                board.pop()
                if alpha >= beta: break
        return alpha
    else:
        if stand_pat <= alpha: return alpha
        beta = min(beta, stand_pat)

        for move in sorted(board.legal_moves, key=lambda m: board.is_capture(m), reverse=True):
            if board.is_capture(move):
                board.push(move)
                beta = min(beta, quiescence(board, alpha, beta, True, current_depth + 1, q_depth + 1))
                board.pop()
                if beta <= alpha: break
        return beta

def minimax(board, depth, alpha, beta, maximizing, current_depth):
    """Main Alpha-Beta search with Transposition Table."""
    global NODES_COUNT
    NODES_COUNT += 1

    # 1. STRICT DRAW DETECTION BEFORE CHECKING MEMORY!
    # Using is_repetition(2) forces the engine to push forward instead of looping
    if current_depth > 0 and (board.is_repetition(2) or board.is_fifty_moves() or board.is_insufficient_material()):
        return 0

    board_hash = chess.polyglot.zobrist_hash(board)

    # 2. Check Transposition Table
    if board_hash in TRANSPOSITION_TABLE:
        stored_depth, stored_eval = TRANSPOSITION_TABLE[board_hash]
        if stored_depth >= depth:
            return stored_eval

    if depth == 0 or board.is_game_over():
        return quiescence(board, alpha, beta, maximizing, current_depth, 0)

    legal_moves = sorted(board.legal_moves, key=lambda m: move_value(board, m, current_depth), reverse=True)
    best_val = -float('inf') if maximizing else float('inf')

    for move in legal_moves:
        board.push(move)
        res = minimax(board, depth - 1, alpha, beta, not maximizing, current_depth + 1)
        board.pop()

        if maximizing:
            best_val = max(best_val, res)
            alpha = max(alpha, res)
        else:
            best_val = min(best_val, res)
            beta = min(beta, res)

        if beta <= alpha:
            if not board.is_capture(move):
                KILLER_MOVES[current_depth][1] = KILLER_MOVES[current_depth][0]
                KILLER_MOVES[current_depth][0] = move
            break

    TRANSPOSITION_TABLE[board_hash] = (depth, best_val)
    return best_val

# CHANGED: Default book_path is now komodo.bin
def get_book_moves(board, count=3, book_path="komodo.bin"):
    """Finds multiple popular moves from the Polyglot opening book."""
    moves = []
    try:
        with chess.polyglot.open_reader(book_path) as reader:
            for entry in reader.find_all(board):
                moves.append(entry.move)
                if len(moves) >= count:
                    break
    except FileNotFoundError:
        pass
    return moves

def get_top_moves(board, depth, count=3):
    """Main entry point. Checks book first, then searches."""
    global NODES_COUNT
    NODES_COUNT = 0

    book_moves = get_book_moves(board, count)
    if book_moves:
        results = []
        is_max = board.turn == chess.WHITE

        for i, move in enumerate(book_moves):
            board.push(move)
            static_score = classical_eval(board)
            board.pop()

            bonus = (3 - i) * 0.05

            if is_max:
                final_score = static_score + bonus
            else:
                final_score = static_score - bonus

            results.append((final_score, move))

        results.sort(key=lambda x: x[0], reverse=is_max)
        print("Book moves found!")
        return results

    is_max = board.turn == chess.WHITE
    legal_moves = list(board.legal_moves)
    results = []

    for d in range(1, depth + 1):
        results = []
        for move in sorted(legal_moves, key=lambda m: move_value(board, m, 0), reverse=True):
            board.push(move)
            val = minimax(board, d - 1, -float('inf'), float('inf'), not is_max, 1)
            board.pop()
            results.append((val, move))

        results.sort(key=lambda x: x[0], reverse=is_max)
        legal_moves = [r[1] for r in results]

        print(f"Depth {d} complete. Nodes searched: {NODES_COUNT}")

    return results[:count]