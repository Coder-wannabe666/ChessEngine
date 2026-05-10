import chess
import chess.polyglot
import torch
from functools import lru_cache
from nn.model import ChessNet
from nn.encoder import board_to_tensor
from evaluation import evaluate_board as classical_eval

# --- Settings ---
DEPTH = 4
AI_WEIGHT = 0.06
AI_MULTIPLIER = 120

# Global caches
TRANSPOSITION_TABLE = {} # {hash: (depth, eval, flag)}
KILLER_MOVES = [[None] * 2 for _ in range(20)] # Stores 2 killer moves per depth

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

def fast_eval(board, use_ai=False):
    """Main evaluation function."""
    if board.is_checkmate():
        return -99999 if board.turn == chess.WHITE else 99999
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    score = classical_eval(board)
    if use_ai:
        # Mix in AI only when requested (for the main search depth)
        score = (score * (1 - AI_WEIGHT)) + (get_ai_score(board.fen()) * AI_WEIGHT)
    return score

def move_value(board, move, depth):
    """Heuristic to prioritize moves that are likely to cause cutoffs."""
    if board.is_capture(move):
        # MVV-LVA: Most Valuable Victim - Least Valuable Attacker
        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)
        victim_val = 0 if not victim else PIECE_VALUES.get(victim.piece_type, 0)
        attacker_val = 0 if not attacker else PIECE_VALUES.get(attacker.piece_type, 0)
        return 10000 + victim_val - attacker_val

    # Prioritize Killer Moves (moves that caused cutoffs at this depth previously)
    if move in KILLER_MOVES[depth]:
        return 5000

    return 0

def quiescence(board, alpha, beta, maximizing):
    """Fast tactical search using ONLY classical evaluation for speed."""
    stand_pat = fast_eval(board, use_ai=False)

    if maximizing:
        if stand_pat >= beta: return beta
        alpha = max(alpha, stand_pat)
        for move in sorted(board.legal_moves, key=lambda m: board.is_capture(m), reverse=True):
            if board.is_capture(move):
                board.push(move)
                alpha = max(alpha, quiescence(board, alpha, beta, False))
                board.pop()
                if alpha >= beta: break
        return alpha
    else:
        if stand_pat <= alpha: return alpha
        beta = min(beta, stand_pat)
        for move in sorted(board.legal_moves, key=lambda m: board.is_capture(m), reverse=True):
            if board.is_capture(move):
                board.push(move)
                beta = min(beta, quiescence(board, alpha, beta, True))
                board.pop()
                if beta <= alpha: break
        return beta

def minimax(board, depth, alpha, beta, maximizing, current_depth):
    board_hash = chess.polyglot.zobrist_hash(board)

    # TT Lookup
    if board_hash in TRANSPOSITION_TABLE:
        d, ev, _ = TRANSPOSITION_TABLE[board_hash]
        if d >= depth: return ev

    if depth == 0 or board.is_game_over():
        # Only use AI at the 'leaf' node of the main search
        return quiescence(board, alpha, beta, maximizing)

    # Sort moves using our heuristic
    legal_moves = sorted(board.legal_moves, key=lambda m: move_value(board, m, current_depth), reverse=True)

    best_val = -float('inf') if maximizing else float('inf')

    for move in legal_moves:
        board.push(move)
        res = minimax(board, depth - 1, alpha, beta, not maximizing, current_depth + 1)
        board.pop()

        if maximizing:
            if res > best_val:
                best_val = res
            alpha = max(alpha, res)
        else:
            if res < best_val:
                best_val = res
            beta = min(beta, res)

        if beta <= alpha:
            # Beta Cutoff: Update Killer Moves
            if not board.is_capture(move):
                KILLER_MOVES[current_depth][1] = KILLER_MOVES[current_depth][0]
                KILLER_MOVES[current_depth][0] = move
            break

    TRANSPOSITION_TABLE[board_hash] = (depth, best_val, 0)
    return best_val

def get_top_moves(board, depth, count=3):
    is_max = board.turn == chess.WHITE
    legal_moves = list(board.legal_moves)
    results = []

    # Iterative Deepening
    for d in range(1, depth + 1):
        results = []
        for move in sorted(legal_moves, key=lambda m: move_value(board, m, 0), reverse=True):
            board.push(move)
            # Use AI only for root-level evaluations
            val = minimax(board, d - 1, -float('inf'), float('inf'), not is_max, 1)
            board.pop()
            results.append((val, move))

        results.sort(key=lambda x: x[0], reverse=is_max)
        legal_moves = [r[1] for r in results]

    return results[:count]

# Helper for move ordering
PIECE_VALUES = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330, chess.ROOK: 500, chess.QUEEN: 900}