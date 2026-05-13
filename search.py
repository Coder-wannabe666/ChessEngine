import chess
import math
import chess.polyglot
import torch
from nn.model import ChessNet
from nn.encoder import board_to_tensor
from evaluation import evaluate_board as classical_eval, is_endgame

# --- Configuration ---
DEPTH        = 6            # depth 6
AI_WEIGHT    = 0.2
AI_MULTIPLIER = 120
SEARCH_ID    = 0

ASPIRATION_WINDOW = 40   # tighter window → more stable eval between positions

# --- TT flag constants ---
TT_EXACT      = 0
TT_LOWERBOUND = 1
TT_UPPERBOUND = 2

# --- Caches & Counters ---
# TT entry: (depth, score, flag, best_move)
TRANSPOSITION_TABLE = {}
KILLER_MOVES = [[None] * 2 for _ in range(64)]
NODES_COUNT  = 0
AI_CACHE     = {}

# History heuristic: (piece_type, to_square) → score
# Quiet moves that cause beta-cutoffs gain score → better ordering → more cutoffs
HISTORY      = {}
HISTORY_MAX  = 8000   # cap to avoid overflow relative to killer (5000)

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
      1. Hash move (from TT)  30000   — best move from previous search
      2. Captures (MVV-LVA)   10000+  — good captures first
      3. Killer moves          5000   — quiet moves that caused cutoffs
      4. History heuristic    0–3999  — quiet moves ordered by past success
      5. Other quiet moves       0
    """
    if move == tt_move:
        return 30000

    if board.is_capture(move):
        victim   = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)
        v_val = PIECE_VALUES.get(victim.piece_type,   0) if victim   else 0
        a_val = PIECE_VALUES.get(attacker.piece_type, 0) if attacker else 0
        mvv_lva = v_val - a_val
        # Bad captures (losing exchanges) scored below quiet moves
        if a_val > v_val and board.is_attacked_by(not attacker.color, move.to_square):
            return -1000 + mvv_lva   # still try but after quiets
        return 10000 + mvv_lva

    if move in KILLER_MOVES[current_depth]:
        return 5000

    # History heuristic for quiet moves
    piece = board.piece_at(move.from_square)
    if piece:
        return HISTORY.get((piece.piece_type, move.to_square), 0)

    return 0

# ── Simplified SEE (Static Exchange Evaluation) ─────────────────────

def see_sign(board, move):
    """
    Returns True if the capture is likely winning or neutral.
    Fast approximation: compares attacker value with victim value.
    Losing captures (Qxp defended by p) are ordered after quiet moves.
    """
    victim   = board.piece_at(move.to_square)
    attacker = board.piece_at(move.from_square)
    if not victim or not attacker:
        return True  # en passant or edge case — allow
    v_val = PIECE_VALUES.get(victim.piece_type,   0)
    a_val = PIECE_VALUES.get(attacker.piece_type, 0)
    if a_val <= v_val:
        return True   # equal or winning trade — always good
    # Attacker is more valuable: only good if destination is undefended
    return not board.is_attacked_by(not attacker.color, move.to_square)


# ── Quiescence search ────────────────────────────────────────────────

# Maximum material gain from any single capture (used for delta pruning)
DELTA_MARGIN = 200   # safety buffer: allow for positional upside after capture

def quiescence(board, alpha, beta, maximizing, current_depth, q_depth=0):
    """
    Quiescence search with delta pruning:
    Skip captures where even getting the piece for free can't beat alpha.
    This removes most losing-capture branches without missing anything important.
    """
    global NODES_COUNT
    NODES_COUNT += 1

    stand_pat = fast_eval(board, current_depth)
    if q_depth >= 6:
        return stand_pat

    if maximizing:
        if stand_pat >= beta: return beta
        alpha = max(alpha, stand_pat)
        for move in board.generate_legal_captures():
            # Delta pruning: if even capturing the piece + margin can't beat alpha, skip
            victim = board.piece_at(move.to_square)
            if victim:
                gain = PIECE_VALUES.get(victim.piece_type, 0)
                if stand_pat + gain + DELTA_MARGIN < alpha:
                    continue
            board.push(move)
            alpha = max(alpha, quiescence(board, alpha, beta, False, current_depth + 1, q_depth + 1))
            board.pop()
            if alpha >= beta: break
        return alpha
    else:
        if stand_pat <= alpha: return alpha
        beta = min(beta, stand_pat)
        for move in board.generate_legal_captures():
            victim = board.piece_at(move.to_square)
            if victim:
                gain = PIECE_VALUES.get(victim.piece_type, 0)
                if stand_pat - gain - DELTA_MARGIN > beta:
                    continue
            board.push(move)
            beta = min(beta, quiescence(board, alpha, beta, True, current_depth + 1, q_depth + 1))
            board.pop()
            if beta <= alpha: break
        return beta

# ── Futility margins ─────────────────────────────────────────────────

FUTILITY_MARGIN = {1: 150, 2: 300, 3: 500}   # razoring at depth 3

# Reverse Futility Pruning (static null move): if eval - margin >= beta, prune.
# Much cheaper than NMP: no board.push needed.
RFP_MARGIN = {1: 150, 2: 250, 3: 350, 4: 450}

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

    in_check   = board.is_check()
    static_val = classical_eval(board)   # used by both RFP and futility

    # --- Reverse Futility Pruning (static null move) ---
    # If static eval is already way above beta, we can prune without searching.
    # Much cheaper than NMP (no board.push). Very effective at depth 1-4.
    if (depth in RFP_MARGIN and not in_check):
        if maximizing     and static_val - RFP_MARGIN[depth] >= beta:  return static_val
        if not maximizing and static_val + RFP_MARGIN[depth] <= alpha: return static_val

    # --- Internal Iterative Deepening (IID) ---
    # If we have no TT move at depth >= 4, do a quick shallow pre-search to
    # populate the TT. The hash move found improves move ordering dramatically.
    if tt_move is None and depth >= 4:
        minimax(board, max(1, depth - 3), alpha, beta,
                maximizing, current_depth, original_id)
        tt_entry2 = TRANSPOSITION_TABLE.get(board_hash)
        if tt_entry2:
            tt_move = tt_entry2[3]

    # --- Null Move Pruning (adaptive R: 3 at depth>=6, 2 otherwise) ---
    NULL_R = 3 if depth >= 6 else 2
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

    # --- Futility Pruning setup (reuse static_val already computed above) ---
    do_futility     = (not in_check and depth in FUTILITY_MARGIN)
    futility_margin = FUTILITY_MARGIN.get(depth, 0)
    static_eval     = static_val   # already computed

    for move_idx, move in enumerate(legal_moves):
        is_capture   = board.is_capture(move)
        is_killer    = move in KILLER_MOVES[current_depth]
        is_promo     = move.promotion is not None
        # gives_check: use board method — NO push/pop needed (saves ~2 push/pop per move)
        gives_check  = board.gives_check(move)

        if do_futility and not is_capture and not is_killer and not is_promo:
            if maximizing     and static_eval + futility_margin <= alpha: continue
            if not maximizing and static_eval - futility_margin >= beta:  continue

        # --- Move Count Based Pruning (late move pruning) ---
        # At low depths, skip very late quiet moves entirely (not just reduce)
        if (depth <= 2 and move_idx >= 8
                and not is_capture and not is_killer and not is_promo
                and not in_check   and not gives_check):
            continue

        # --- LMR (log-based formula, no separate push/pop needed) ---
        reduction = 0
        if (depth >= 3 and move_idx >= 2
                and not is_capture and not is_killer and not is_promo
                and not in_check   and not gives_check):
            reduction = max(1, int(math.log(depth) * math.log(move_idx + 1) / 1.5))
            reduction = min(reduction, depth - 1)

        board.push(move)

        # --- PVS (Principal Variation Search) ---
        # First move: full window. Subsequent moves: null window, re-search if needed.
        if move_idx == 0:
            res = minimax(board, depth - 1 - reduction, alpha, beta,
                          not maximizing, current_depth + 1, original_id)
        else:
            # Null window search [alpha, alpha+1] or [beta-1, beta]
            if maximizing:
                nw_alpha, nw_beta = alpha, alpha + 1
            else:
                nw_alpha, nw_beta = beta - 1, beta
            res = minimax(board, depth - 1 - reduction, nw_alpha, nw_beta,
                          not maximizing, current_depth + 1, original_id)
            # Re-search with full window if null window beat alpha (or improved beta)
            if ((maximizing and res > alpha) or (not maximizing and res < beta)):
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
                piece = board.piece_at(move.from_square)
                if piece:
                    key = (piece.piece_type, move.to_square)
                    HISTORY[key] = min(HISTORY_MAX,
                                       HISTORY.get(key, 0) + depth * depth)
            break

    # --- Store TT with best_move ---
    if best_val <= orig_alpha: tt_flag = TT_UPPERBOUND
    elif best_val >= beta:     tt_flag = TT_LOWERBOUND
    else:                      tt_flag = TT_EXACT

    # Cap TT size to ~800k entries to avoid memory bloat slowing down dict lookups
    if len(TRANSPOSITION_TABLE) < 800_000:
        TRANSPOSITION_TABLE[board_hash] = (depth, best_val, tt_flag, best_move)
    elif board_hash in TRANSPOSITION_TABLE:
        # Always overwrite existing entry if we have a deeper result
        existing = TRANSPOSITION_TABLE[board_hash]
        if depth >= existing[0]:
            TRANSPOSITION_TABLE[board_hash] = (depth, best_val, tt_flag, best_move)
    return best_val

# ── PV reconstruction ────────────────────────────────────────────────

def get_pv_line(board, max_depth, root_move=None):
    """
    Reconstructs the Principal Variation by following best_move entries in TT.

    root_move: if provided, forced as the first step regardless of TT root entry.
    This is necessary when the top move was chosen after blending AI scores —
    the TT root entry may point to a different (pure-minimax) best move, causing
    the displayed line to start with a different move than TOP MOVES shows.
    """
    pv   = []
    temp = board.copy()
    seen = set()

    # Force the blended best move as step 1
    if root_move is not None and root_move in temp.legal_moves:
        pv.append(root_move)
        temp.push(root_move)

    for _ in range(max_depth * 2):
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

def get_top_moves(board, depth, count=3, on_depth=None):
    """
    Returns (results, pv_line) where:
      results  = [(score, move), ...] top 'count' moves
      pv_line  = [move, move, ...]   best continuation from root position

    on_depth: optional callback(depth, results, pv) called after each iteration.
    Enables progressive display — GUI can show depth-N results without waiting
    for the full search to complete.
    """
    global NODES_COUNT, SEARCH_ID, HISTORY, KILLER_MOVES
    SEARCH_ID += 1
    current_id = SEARCH_ID
    NODES_COUNT = 0
    # Age history: halve all scores each search to keep recent info more relevant
    for k in HISTORY:
        HISTORY[k] >>= 1
    # Reset killer moves for clean ordering each search
    KILLER_MOVES = [[None] * 2 for _ in range(64)]

    # 1. Opening Book — collect as a set for bonus lookup.
    #    Full minimax always runs: book moves just get a bonus so they
    #    bubble to the top while still having correct scores and a full PV.
    book_move_set = set(get_book_moves(board, count=6))
    BOOK_BONUS    = 30   # centipawns added to book moves in final blend

    # 2. Iterative Deepening + Aspiration Windows
    is_max      = board.turn == chess.WHITE
    legal_moves = list(board.legal_moves)
    results     = []

    # Seed aspiration window from TT (reduces horizon jumps when user plays a move).
    # If the position was already evaluated in a previous search, use that score
    # as the center of the aspiration window from iteration 1.
    _init_hash = chess.polyglot.zobrist_hash(board)
    _tt_init   = TRANSPOSITION_TABLE.get(_init_hash)
    if _tt_init and _tt_init[2] == TT_EXACT:
        prev_score = _tt_init[1]
    else:
        prev_score = None

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

        results    = []
        failed_asp = False

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

            # display_score = pure minimax (what GUI shows — consistent with next position)
            # order_score   = blended + book bonus (used only for sorting)
            ai_score    = ai_evals[move]
            book_bonus  = BOOK_BONUS if move in book_move_set else 0
            order_score = (val * (1 - AI_WEIGHT)) + (ai_score * AI_WEIGHT)
            order_score += book_bonus if is_max else -book_bonus
            results.append((val, order_score, move))

        if failed_asp:
            # Widen to full window and redo entire depth
            results = []
            for move in sorted_root:
                board.push(move)
                val = minimax(board, d - 1, -float('inf'), float('inf'),
                              not is_max, 1, current_id)
                board.pop()
                ai_score    = ai_evals[move]
                book_bonus  = BOOK_BONUS if move in book_move_set else 0
                order_score = (val * (1 - AI_WEIGHT)) + (ai_score * AI_WEIGHT)
                order_score += book_bonus if is_max else -book_bonus
                results.append((val, order_score, move))

        results.sort(key=lambda x: x[1], reverse=is_max)   # sort by order_score
        legal_moves = [r[2] for r in results]               # re-order for next iter

        if results:
            prev_score = int(results[0][0])  # track pure minimax for aspiration window

        is_book = "book" if results and results[0][2] in book_move_set else ""
        print(f"Depth {d} complete. Nodes: {NODES_COUNT}  Best: {results[0][2].uci() if results else '?'} {is_book}  Score: {results[0][0] if results else '?'}")

        # Progressive display: fire callback after every depth
        if on_depth and results:
            _root = results[0][2]
            _pv   = get_pv_line(board, d, root_move=_root)
            on_depth(d, [(r[0], r[2]) for r in results[:count]], _pv)

    root_move = results[0][2] if results else None
    pv = get_pv_line(board, depth, root_move=root_move)
    # Return (display_score, move) — strip internal order_score before returning
    return [(r[0], r[2]) for r in results[:count]], pv