import chess
import chess.polyglot
import torch
from nn.model import ChessNet
from nn.encoder import board_to_tensor
from evaluation import evaluate_board as classical_eval, is_endgame

# --- Configuration ---
DEPTH = 4
AI_WEIGHT = 0.2       # Balance classical math and NN intuition
AI_MULTIPLIER = 120   # NN output scaling
SEARCH_ID = 0

# --- TT flag constants ---
TT_EXACT      = 0
TT_LOWERBOUND = 1   # beta cutoff — score is a lower bound
TT_UPPERBOUND = 2   # all-node   — score is an upper bound

# --- Caches & Counters ---
TRANSPOSITION_TABLE = {}     # hash -> (depth, score, flag)
KILLER_MOVES = [[None] * 2 for _ in range(64)]
NODES_COUNT = 0
AI_CACHE = {}

PIECE_VALUES = {
    chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
    chess.ROOK: 500, chess.QUEEN: 900
}

# --- Neural Network Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ChessNet().to(device)
try:
    model.load_state_dict(torch.load("nn/chess_model.pth", map_location=device, weights_only=True))
    model.eval()
    print(f"Model loaded on {device}")
except FileNotFoundError:
    print("Warning: nn/chess_model.pth not found.")

def get_ai_score(board):
    """Deep intuition - optimized caching with Zobrist Hash."""
    board_hash = chess.polyglot.zobrist_hash(board)
    if board_hash in AI_CACHE:
        return AI_CACHE[board_hash]

    tensor = board_to_tensor(board).unsqueeze(0).to(device)
    with torch.no_grad():
        score = model(tensor).item() * AI_MULTIPLIER

    AI_CACHE[board_hash] = score
    return score

def fast_eval(board, current_depth):
    """Pure classical evaluation for speed."""
    if board.is_checkmate():
        return -99999 + current_depth if board.turn == chess.WHITE else 99999 - current_depth
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    return classical_eval(board)

def move_value(board, move, current_depth):
    """Sorts moves: Captures first (MVV-LVA), then Killer Moves."""
    if board.is_capture(move):
        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)
        v_val = PIECE_VALUES.get(victim.piece_type, 0) if victim else 0
        a_val = PIECE_VALUES.get(attacker.piece_type, 0) if attacker else 0
        return 10000 + v_val - a_val

    if move in KILLER_MOVES[current_depth]:
        return 5000
    return 0

def quiescence(board, alpha, beta, maximizing, current_depth, q_depth=0):
    """Fast tactical search using generate_legal_captures."""
    global NODES_COUNT
    NODES_COUNT += 1

    stand_pat = fast_eval(board, current_depth)
    if q_depth >= 4:
        return stand_pat

    if maximizing:
        if stand_pat >= beta:
            return beta
        alpha = max(alpha, stand_pat)
        for move in board.generate_legal_captures():
            board.push(move)
            alpha = max(alpha, quiescence(board, alpha, beta, False, current_depth + 1, q_depth + 1))
            board.pop()
            if alpha >= beta:
                break
        return alpha
    else:
        if stand_pat <= alpha:
            return alpha
        beta = min(beta, stand_pat)
        for move in board.generate_legal_captures():
            board.push(move)
            beta = min(beta, quiescence(board, alpha, beta, True, current_depth + 1, q_depth + 1))
            board.pop()
            if beta <= alpha:
                break
        return beta

# --- Futility margin per depth ---
FUTILITY_MARGIN = {1: 150, 2: 300}   # centipawns

def minimax(board, depth, alpha, beta, maximizing, current_depth, original_id):
    """
    Alpha-Beta search with:
      - Null Move Pruning   (NMP)  — biggest speed gain in middlegame
      - TT with bound flags        — more cache hits
      - Late Move Reductions (LMR) — reduce depth for late quiet moves
      - Futility Pruning           — prune hopeless quiet moves at depth 1-2
    """
    global NODES_COUNT
    if original_id != SEARCH_ID:
        return 0

    NODES_COUNT += 1
    orig_alpha = alpha

    # --- Immediate draw detection ---
    if current_depth > 0 and (board.is_repetition(2) or board.is_fifty_moves()):
        return 0

    # --- Transposition Table lookup ---
    board_hash = chess.polyglot.zobrist_hash(board)
    tt_entry = TRANSPOSITION_TABLE.get(board_hash)
    if tt_entry is not None:
        tt_depth, tt_score, tt_flag = tt_entry
        if tt_depth >= depth:
            if tt_flag == TT_EXACT:
                return tt_score
            elif tt_flag == TT_LOWERBOUND:
                alpha = max(alpha, tt_score)
            elif tt_flag == TT_UPPERBOUND:
                beta = min(beta, tt_score)
            if alpha >= beta:
                return tt_score

    # --- Leaf / game-over ---
    if depth <= 0 or board.is_game_over():
        return quiescence(board, alpha, beta, maximizing, current_depth)

    in_check = board.is_check()

    # ----------------------------------------------------------------
    # NULL MOVE PRUNING
    # Skip if: in check, depth <= 1, or endgame (zugzwang risk)
    # ----------------------------------------------------------------
    NULL_R = 2
    if (depth >= 2
            and not in_check
            and current_depth > 0
            and not is_endgame(board)):
        board.push(chess.Move.null())
        null_score = minimax(board, depth - 1 - NULL_R, alpha, beta,
                             not maximizing, current_depth + 1, original_id)
        board.pop()

        if maximizing and null_score >= beta:
            return beta
        if not maximizing and null_score <= alpha:
            return alpha

    # --- Move ordering ---
    legal_moves = sorted(board.legal_moves,
                         key=lambda m: move_value(board, m, current_depth),
                         reverse=True)

    best_val = -float('inf') if maximizing else float('inf')

    # ----------------------------------------------------------------
    # FUTILITY PRUNING  (depth 1 and 2, not in check)
    # ----------------------------------------------------------------
    do_futility = (not in_check and depth in FUTILITY_MARGIN)
    if do_futility:
        static_eval = classical_eval(board)
        futility_margin = FUTILITY_MARGIN[depth]

    for move_idx, move in enumerate(legal_moves):
        is_capture = board.is_capture(move)
        is_killer  = move in KILLER_MOVES[current_depth]
        is_promo   = move.promotion is not None

        # Futility pruning: skip quiet moves that can't improve alpha/beta
        if do_futility and not is_capture and not is_killer and not is_promo:
            if maximizing and static_eval + futility_margin <= alpha:
                continue
            if not maximizing and static_eval - futility_margin >= beta:
                continue

        board.push(move)
        gives_check = board.is_check()
        board.pop()

        # ----------------------------------------------------------------
        # LATE MOVE REDUCTIONS (LMR)
        # Reduce depth for quiet, non-killer moves after the first few
        # ----------------------------------------------------------------
        reduction = 0
        if (depth >= 3
                and move_idx >= 3
                and not is_capture
                and not is_killer
                and not is_promo
                and not in_check
                and not gives_check):
            reduction = 1
            if move_idx >= 6:
                reduction = 2

        board.push(move)
        res = minimax(board, depth - 1 - reduction, alpha, beta,
                      not maximizing, current_depth + 1, original_id)

        # Re-search at full depth if LMR beat alpha (don't miss good moves)
        if reduction > 0 and (
                (maximizing and res > alpha) or
                (not maximizing and res < beta)):
            res = minimax(board, depth - 1, alpha, beta,
                          not maximizing, current_depth + 1, original_id)

        board.pop()

        if maximizing:
            best_val = max(best_val, res)
            alpha = max(alpha, res)
        else:
            best_val = min(best_val, res)
            beta = min(beta, res)

        if beta <= alpha:
            if not is_capture:
                KILLER_MOVES[current_depth][1] = KILLER_MOVES[current_depth][0]
                KILLER_MOVES[current_depth][0] = move
            break

    # --- Store TT entry with correct flag ---
    if best_val <= orig_alpha:
        tt_flag = TT_UPPERBOUND
    elif best_val >= beta:
        tt_flag = TT_LOWERBOUND
    else:
        tt_flag = TT_EXACT

    TRANSPOSITION_TABLE[board_hash] = (depth, best_val, tt_flag)
    return best_val

def get_book_moves(board, count=3, book_path="komodo.bin"):
    """Finds popular moves from the Opening Book."""
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
    global NODES_COUNT, SEARCH_ID
    SEARCH_ID += 1
    current_id = SEARCH_ID
    NODES_COUNT = 0

    # 1. Opening Book Check
    book_moves = get_book_moves(board, count)
    if book_moves:
        results = []
        is_max = board.turn == chess.WHITE
        for i, move in enumerate(book_moves):
            board.push(move)
            static_score = classical_eval(board)
            board.pop()
            bonus = (3 - i) * 0.05
            final_score = static_score + bonus if is_max else static_score - bonus
            results.append((final_score, move))
        results.sort(key=lambda x: x[0], reverse=is_max)
        return results

    # 2. Iterative Deepening with AI Root Intuition
    is_max = board.turn == chess.WHITE
    legal_moves = list(board.legal_moves)
    results = []

    # AI intuition for root moves ONCE
    ai_evals = {}
    for move in legal_moves:
        board.push(move)
        ai_evals[move] = get_ai_score(board)
        board.pop()

    for d in range(1, depth + 1):
        results = []
        for move in sorted(legal_moves,
                           key=lambda m: move_value(board, m, 0),
                           reverse=True):
            board.push(move)
            val = minimax(board, d - 1, -float('inf'), float('inf'),
                          not is_max, 1, current_id)
            board.pop()

            ai_score = ai_evals[move]
            final_val = (val * (1 - AI_WEIGHT)) + (ai_score * AI_WEIGHT)
            results.append((final_val, move))

        results.sort(key=lambda x: x[0], reverse=is_max)
        legal_moves = [r[1] for r in results]
        print(f"Depth {d} complete. Nodes: {NODES_COUNT}")

    return results[:count]