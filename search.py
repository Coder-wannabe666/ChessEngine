import chess
import chess.polyglot
import torch
from nn.model import ChessNet
from nn.encoder import board_to_tensor
from evaluation import evaluate_board as classical_eval, is_endgame

# --- Configuration ---
DEPTH        = 5
AI_WEIGHT    = 0.2
AI_MULTIPLIER = 120
SEARCH_ID    = 0

ASPIRATION_WINDOW = 50   # centipawns; widen to full window on fail

# --- TT flag constants ---
TT_EXACT      = 0
TT_LOWERBOUND = 1
TT_UPPERBOUND = 2

# --- Caches & Counters ---
# TT entry: (depth, score, flag, best_move)
# Storing best_move enables hash-move ordering and PV reconstruction.
TRANSPOSITION_TABLE = {}
KILLER_MOVES = [[None] * 2 for _ in range(64)]
NODES_COUNT  = 0
AI_CACHE     = {}

PIECE_VALUES = {
    chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
    chess.ROOK: 500, chess.QUEEN: 900
}

# --- Neural Network Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = ChessNet().to(device)
try:
    model.load_state_dict(torch.load("nn/chess_model.pth", map_location=device, weights_only=True))
    model.eval()
    print(f"Model loaded on {device}")
except FileNotFoundError:
    print("Warning: nn/chess_model.pth not found.")

# ── Neural network ────────────────────────────────────────────────────

def get_ai_score(board):
    board_hash = chess.polyglot.zobrist_hash(board)
    if board_hash in AI_CACHE:
        return AI_CACHE[board_hash]
    tensor = board_to_tensor(board).unsqueeze(0).to(device)
    with torch.no_grad():
        score = model(tensor).item() * AI_MULTIPLIER
    AI_CACHE[board_hash] = score
    return score

# ── Evaluation helpers ───────────────────────────────────────────────

def fast_eval(board, current_depth):
    if board.is_checkmate():
        return -99999 + current_depth if board.turn == chess.WHITE else 99999 - current_depth
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    return classical_eval(board)

def move_value(board, move, current_depth, tt_move=None):
    """
    Move ordering priority:
      1. Hash move (from TT)       — most likely best move, huge stability gain
      2. Captures (MVV-LVA)
      3. Killer moves
      4. Quiet moves
    """
    if move == tt_move:
        return 30000   # above all other moves

    if board.is_capture(move):
        victim   = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)
        v_val = PIECE_VALUES.get(victim.piece_type,   0) if victim   else 0
        a_val = PIECE_VALUES.get(attacker.piece_type, 0) if attacker else 0
        return 10000 + v_val - a_val

    if move in KILLER_MOVES[current_depth]:
        return 5000

    return 0

# ── Quiescence search ────────────────────────────────────────────────

def quiescence(board, alpha, beta, maximizing, current_depth, q_depth=0):
    global NODES_COUNT
    NODES_COUNT += 1

    stand_pat = fast_eval(board, current_depth)
    if q_depth >= 4:
        return stand_pat

    if maximizing:
        if stand_pat >= beta: return beta
        alpha = max(alpha, stand_pat)
        for move in board.generate_legal_captures():
            board.push(move)
            alpha = max(alpha, quiescence(board, alpha, beta, False, current_depth + 1, q_depth + 1))
            board.pop()
            if alpha >= beta: break
        return alpha
    else:
        if stand_pat <= alpha: return alpha
        beta = min(beta, stand_pat)
        for move in board.generate_legal_captures():
            board.push(move)
            beta = min(beta, quiescence(board, alpha, beta, True, current_depth + 1, q_depth + 1))
            board.pop()
            if beta <= alpha: break
        return beta

# ── Futility margins ─────────────────────────────────────────────────

FUTILITY_MARGIN = {1: 150, 2: 300}

# ── Main Alpha-Beta ──────────────────────────────────────────────────

def minimax(board, depth, alpha, beta, maximizing, current_depth, original_id):
    """
    Alpha-Beta with:
      • Hash-move ordering   (best move from TT tried first — biggest stability fix)
      • Null Move Pruning
      • TT with EXACT / LOWER / UPPER flags + stored best_move
      • Late Move Reductions (LMR)
      • Futility Pruning
    """
    global NODES_COUNT
    if original_id != SEARCH_ID:
        return 0

    NODES_COUNT += 1
    orig_alpha = alpha

    if current_depth > 0 and (board.is_repetition(2) or board.is_fifty_moves()):
        return 0

    board_hash = chess.polyglot.zobrist_hash(board)

    # --- TT lookup ---
    tt_move  = None
    tt_entry = TRANSPOSITION_TABLE.get(board_hash)
    if tt_entry is not None:
        tt_depth, tt_score, tt_flag, tt_move = tt_entry
        if tt_depth >= depth:
            if tt_flag == TT_EXACT:
                return tt_score
            elif tt_flag == TT_LOWERBOUND:
                alpha = max(alpha, tt_score)
            elif tt_flag == TT_UPPERBOUND:
                beta  = min(beta,  tt_score)
            if alpha >= beta:
                return tt_score

    if depth <= 0 or board.is_game_over():
        return quiescence(board, alpha, beta, maximizing, current_depth)

    in_check = board.is_check()

    # --- Null Move Pruning ---
    NULL_R = 2
    if depth >= 2 and not in_check and current_depth > 0 and not is_endgame(board):
        board.push(chess.Move.null())
        null_score = minimax(board, depth - 1 - NULL_R, alpha, beta,
                             not maximizing, current_depth + 1, original_id)
        board.pop()
        if maximizing and null_score >= beta: return beta
        if not maximizing and null_score <= alpha: return alpha

    # --- Move ordering (hash move first) ---
    legal_moves = sorted(
        board.legal_moves,
        key=lambda m: move_value(board, m, current_depth, tt_move),
        reverse=True
    )

    best_val  = -float('inf') if maximizing else float('inf')
    best_move = None   # track locally for TT storage

    # --- Futility Pruning setup ---
    do_futility = (not in_check and depth in FUTILITY_MARGIN)
    if do_futility:
        static_eval    = classical_eval(board)
        futility_margin = FUTILITY_MARGIN[depth]

    for move_idx, move in enumerate(legal_moves):
        is_capture = board.is_capture(move)
        is_killer  = move in KILLER_MOVES[current_depth]
        is_promo   = move.promotion is not None

        if do_futility and not is_capture and not is_killer and not is_promo:
            if maximizing     and static_eval + futility_margin <= alpha: continue
            if not maximizing and static_eval - futility_margin >= beta:  continue

        board.push(move)
        gives_check = board.is_check()
        board.pop()

        # --- LMR ---
        reduction = 0
        if (depth >= 3 and move_idx >= 3
                and not is_capture and not is_killer and not is_promo
                and not in_check   and not gives_check):
            reduction = 1 if move_idx < 6 else 2

        board.push(move)
        res = minimax(board, depth - 1 - reduction, alpha, beta,
                      not maximizing, current_depth + 1, original_id)

        if reduction > 0 and (
                (maximizing and res > alpha) or (not maximizing and res < beta)):
            res = minimax(board, depth - 1, alpha, beta,
                          not maximizing, current_depth + 1, original_id)
        board.pop()

        improved = (maximizing and res > best_val) or (not maximizing and res < best_val)
        if improved:
            best_val  = res
            best_move = move

        if maximizing: alpha = max(alpha, res)
        else:          beta  = min(beta,  res)

        if beta <= alpha:
            if not is_capture:
                KILLER_MOVES[current_depth][1] = KILLER_MOVES[current_depth][0]
                KILLER_MOVES[current_depth][0] = move
            break

    # --- Store TT with best_move ---
    if best_val <= orig_alpha: tt_flag = TT_UPPERBOUND
    elif best_val >= beta:     tt_flag = TT_LOWERBOUND
    else:                      tt_flag = TT_EXACT

    TRANSPOSITION_TABLE[board_hash] = (depth, best_val, tt_flag, best_move)
    return best_val

# ── PV reconstruction ────────────────────────────────────────────────

def get_pv_line(board, max_depth):
    """
    Reconstructs the Principal Variation by following the best_move
    stored in TT at each position. Returns a list of chess.Move objects.

    Call this AFTER get_top_moves completes (TT is populated).
    Works on a copy of the board — does not mutate the caller's board.
    """
    pv    = []
    temp  = board.copy()
    seen  = set()

    for _ in range(max_depth * 2):   # *2 to capture both sides of the line
        h = chess.polyglot.zobrist_hash(temp)
        if h in seen:
            break
        seen.add(h)

        entry = TRANSPOSITION_TABLE.get(h)
        if not entry or entry[3] is None:
            break

        move = entry[3]
        if move not in temp.legal_moves:
            break

        pv.append(move)
        temp.push(move)

    return pv

# ── Opening book ─────────────────────────────────────────────────────

def get_book_moves(board, count=3, book_path="komodo.bin"):
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

# ── Main entry point ─────────────────────────────────────────────────

def get_top_moves(board, depth, count=3):
    """
    Returns (results, pv_line) where:
      results  = [(score, move), ...] top 'count' moves
      pv_line  = [move, move, ...]   best continuation from root position

    Changes vs previous version:
      • Aspiration windows in iterative deepening (reduces eval swings)
      • Hash-move ordering stabilises results across depths
      • PV line returned alongside top moves
    """
    global NODES_COUNT, SEARCH_ID
    SEARCH_ID += 1
    current_id = SEARCH_ID
    NODES_COUNT = 0

    # 1. Opening Book
    book_moves = get_book_moves(board, count)
    if book_moves:
        results = []
        is_max  = board.turn == chess.WHITE
        for i, move in enumerate(book_moves):
            board.push(move)
            static_score = classical_eval(board)
            board.pop()
            bonus      = (3 - i) * 0.05
            final_score = static_score + bonus if is_max else static_score - bonus
            results.append((final_score, move))
        results.sort(key=lambda x: x[0], reverse=is_max)
        pv = get_pv_line(board, depth)
        return results, pv

    # 2. Iterative Deepening + Aspiration Windows
    is_max      = board.turn == chess.WHITE
    legal_moves = list(board.legal_moves)
    results     = []
    prev_score  = None   # track best score from previous iteration

    # AI intuition at root (called once per move, not recursively)
    ai_evals = {}
    for move in legal_moves:
        board.push(move)
        ai_evals[move] = get_ai_score(board)
        board.pop()

    for d in range(1, depth + 1):
        # ---- Aspiration window setup ----
        if prev_score is None or d <= 2:
            asp_alpha, asp_beta = -float('inf'), float('inf')
            use_asp = False
        else:
            asp_alpha = prev_score - ASPIRATION_WINDOW
            asp_beta  = prev_score + ASPIRATION_WINDOW
            use_asp   = True

        results     = []
        failed_asp  = False

        sorted_root = sorted(legal_moves,
                             key=lambda m: move_value(board, m, 0),
                             reverse=True)

        for move in sorted_root:
            board.push(move)
            val = minimax(board, d - 1, asp_alpha, asp_beta,
                          not is_max, 1, current_id)
            board.pop()

            # Aspiration fail — re-search this depth with full window
            if use_asp and (val <= asp_alpha or val >= asp_beta):
                failed_asp = True
                break

            ai_score  = ai_evals[move]
            final_val = (val * (1 - AI_WEIGHT)) + (ai_score * AI_WEIGHT)
            results.append((final_val, move))

        if failed_asp:
            # Widen to full window and redo entire depth
            results = []
            for move in sorted_root:
                board.push(move)
                val = minimax(board, d - 1, -float('inf'), float('inf'),
                              not is_max, 1, current_id)
                board.pop()
                ai_score  = ai_evals[move]
                final_val = (val * (1 - AI_WEIGHT)) + (ai_score * AI_WEIGHT)
                results.append((final_val, move))

        results.sort(key=lambda x: x[0], reverse=is_max)
        legal_moves = [r[1] for r in results]  # re-order for next iteration

        if results:
            prev_score = int(results[0][0])

        print(f"Depth {d} complete. Nodes: {NODES_COUNT}  Best: {results[0][1].uci() if results else '?'}  Score: {prev_score}")

    pv = get_pv_line(board, depth)
    return results[:count], pv