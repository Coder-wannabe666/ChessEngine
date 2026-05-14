import chess
import math
import chess.polyglot
import torch
from nn.model import ChessNet
from nn.encoder import board_to_tensor
from evaluation import evaluate_board as classical_eval, is_endgame, fast_material_pst_eval

# --- Configuration ---
DEPTH        = 6
AI_WEIGHT    = 0.2
AI_MULTIPLIER = 120
SEARCH_ID    = 0

ASPIRATION_WINDOW = 40

# --- TT flag constants ---
TT_EXACT      = 0
TT_LOWERBOUND = 1
TT_UPPERBOUND = 2

# --- Caches & Counters ---
TRANSPOSITION_TABLE = {}
KILLER_MOVES = [[None] * 2 for _ in range(64)]
NODES_COUNT  = 0
AI_CACHE     = {}

# History heuristic
HISTORY      = {}
HISTORY_MAX  = 8000

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


# ── Evaluation helpers ───────────────────────────────────────────────

def fast_eval(board, current_depth):
    # Used in Q-Search.
    if board.is_checkmate():
        return -99999 + current_depth if board.turn == chess.WHITE else 99999 - current_depth
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    return classical_eval(board)

def move_value(board, move, current_depth, tt_move=None):
    # Move ordering priority.
    if move == tt_move:
        return 30000

    if board.is_capture(move):
        victim   = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)
        v_val = PIECE_VALUES.get(victim.piece_type,   0) if victim   else 0
        a_val = PIECE_VALUES.get(attacker.piece_type, 0) if attacker else 0
        mvv_lva = v_val - a_val
        if a_val > v_val and board.is_attacked_by(not attacker.color, move.to_square):
            return -1000 + mvv_lva
        return 10000 + mvv_lva

    if move in KILLER_MOVES[current_depth]:
        return 5000

    piece = board.piece_at(move.from_square)
    if piece:
        return HISTORY.get((piece.piece_type, move.to_square), 0)

    return 0


# ── Simplified SEE (Static Exchange Evaluation) ─────────────────────

def see_sign(board, move):
    # Fast approximation for winning capture.
    victim   = board.piece_at(move.to_square)
    attacker = board.piece_at(move.from_square)
    if not victim or not attacker:
        return True
    v_val = PIECE_VALUES.get(victim.piece_type,   0)
    a_val = PIECE_VALUES.get(attacker.piece_type, 0)
    if a_val <= v_val:
        return True
    return not board.is_attacked_by(not attacker.color, move.to_square)


# ── Quiescence search ────────────────────────────────────────────────

DELTA_MARGIN = 200

def quiescence(board, alpha, beta, maximizing, current_depth, q_depth=0):
    # Quiescence search with delta pruning.
    global NODES_COUNT
    NODES_COUNT += 1

    stand_pat = fast_eval(board, current_depth)
    if q_depth >= 6:
        return stand_pat

    if maximizing:
        if stand_pat >= beta: return beta
        alpha = max(alpha, stand_pat)
        for move in board.generate_legal_captures():
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

FUTILITY_MARGIN = {1: 150, 2: 300, 3: 500}

RFP_MARGIN = {1: 150, 2: 250, 3: 350, 4: 450}

# ── Main Alpha-Beta ──────────────────────────────────────────────────

def minimax(board, depth, alpha, beta, maximizing, current_depth, original_id):
    # Alpha-Beta with LMR and Null Move Pruning.
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
        # Call Q-search at leaf nodes.
        return quiescence(board, alpha, beta, maximizing, current_depth)

    in_check   = board.is_check()

    # Use full eval for accurate pruning margins
    static_val = classical_eval(board)

    # --- Reverse Futility Pruning ---
    if (depth in RFP_MARGIN and not in_check):
        if maximizing     and static_val - RFP_MARGIN[depth] >= beta:  return static_val
        if not maximizing and static_val + RFP_MARGIN[depth] <= alpha: return static_val

    # --- Internal Iterative Deepening ---
    if tt_move is None and depth >= 4:
        minimax(board, max(1, depth - 3), alpha, beta,
                maximizing, current_depth, original_id)
        tt_entry2 = TRANSPOSITION_TABLE.get(board_hash)
        if tt_entry2:
            tt_move = tt_entry2[3]

    # --- Null Move Pruning ---
    NULL_R = 3 if depth >= 6 else 2
    if depth >= 2 and not in_check and current_depth > 0 and not is_endgame(board):
        board.push(chess.Move.null())
        null_score = minimax(board, depth - 1 - NULL_R, alpha, beta,
                             not maximizing, current_depth + 1, original_id)
        board.pop()
        if maximizing and null_score >= beta: return beta
        if not maximizing and null_score <= alpha: return alpha

    # --- Move ordering ---
    legal_moves = sorted(
        board.legal_moves,
        key=lambda m: move_value(board, m, current_depth, tt_move),
        reverse=True
    )

    best_val  = -999999 if maximizing else 999999
    best_move = None

    do_futility     = (not in_check and depth in FUTILITY_MARGIN)
    futility_margin = FUTILITY_MARGIN.get(depth, 0)
    static_eval     = static_val

    for move_idx, move in enumerate(legal_moves):
        is_capture   = board.is_capture(move)
        is_killer    = move in KILLER_MOVES[current_depth]
        is_promo     = move.promotion is not None
        gives_check  = board.gives_check(move)

        if do_futility and not is_capture and not is_killer and not is_promo:
            if maximizing     and static_eval + futility_margin <= alpha: continue
            if not maximizing and static_eval - futility_margin >= beta:  continue

        # --- Late Move Pruning ---
        if (depth <= 2 and move_idx >= 8
                and not is_capture and not is_killer and not is_promo
                and not in_check   and not gives_check):
            continue

        # --- LMR ---
        reduction = 0
        if (depth >= 3 and move_idx >= 2
                and not is_capture and not is_killer and not is_promo
                and not in_check   and not gives_check):
            reduction = max(1, int(math.log(depth) * math.log(move_idx + 1) / 1.5))
            reduction = min(reduction, depth - 1)

        board.push(move)

        # --- PVS ---
        if move_idx == 0:
            res = minimax(board, depth - 1 - reduction, alpha, beta,
                          not maximizing, current_depth + 1, original_id)
        else:
            if maximizing:
                nw_alpha, nw_beta = alpha, alpha + 1
            else:
                nw_alpha, nw_beta = beta - 1, beta
            res = minimax(board, depth - 1 - reduction, nw_alpha, nw_beta,
                          not maximizing, current_depth + 1, original_id)
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

    # --- Store TT ---
    if best_val <= orig_alpha: tt_flag = TT_UPPERBOUND
    elif best_val >= beta:     tt_flag = TT_LOWERBOUND
    else:                      tt_flag = TT_EXACT

    if len(TRANSPOSITION_TABLE) < 800_000:
        TRANSPOSITION_TABLE[board_hash] = (depth, best_val, tt_flag, best_move)
    elif board_hash in TRANSPOSITION_TABLE:
        existing = TRANSPOSITION_TABLE[board_hash]
        if depth >= existing[0]:
            TRANSPOSITION_TABLE[board_hash] = (depth, best_val, tt_flag, best_move)
    return best_val


# ── PV reconstruction ────────────────────────────────────────────────

def get_pv_line(board, max_depth, root_move=None):
    # Reconstruct PV from TT.
    pv   = []
    temp = board.copy()
    seen = set()

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
    # Read opening book moves.
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
    # Main search loop.
    global NODES_COUNT, SEARCH_ID, HISTORY, KILLER_MOVES
    SEARCH_ID += 1
    current_id = SEARCH_ID
    NODES_COUNT = 0

    for k in HISTORY:
        HISTORY[k] >>= 1
    KILLER_MOVES = [[None] * 2 for _ in range(64)]

    book_move_set = set(get_book_moves(board, count=6))
    BOOK_BONUS    = 30

    is_max      = board.turn == chess.WHITE
    legal_moves = list(board.legal_moves)
    results     = []

    _init_hash = chess.polyglot.zobrist_hash(board)
    _tt_init   = TRANSPOSITION_TABLE.get(_init_hash)
    if _tt_init and _tt_init[2] == TT_EXACT:
        prev_score = _tt_init[1]
    else:
        prev_score = None

    # --- Batched AI eval ---
    tensors = []
    moves_to_predict = []
    ai_evals = {}

    for move in legal_moves:
        board.push(move)
        board_hash = chess.polyglot.zobrist_hash(board)

        if board_hash in AI_CACHE:
            ai_evals[move] = AI_CACHE[board_hash]
        else:
            tensors.append(board_to_tensor(board))
            moves_to_predict.append((board_hash, move))
        board.pop()

    if tensors:
        batch_tensor = torch.stack(tensors).to(device)
        with torch.no_grad():
            scores = model(batch_tensor).squeeze().tolist()

        if isinstance(scores, float):
            scores = [scores]

        for (h, m), score in zip(moves_to_predict, scores):
            scaled_score = score * AI_MULTIPLIER
            AI_CACHE[h] = scaled_score
            ai_evals[m] = scaled_score

    for d in range(1, depth + 1):
        if prev_score is None or d <= 2:
            asp_alpha, asp_beta = -999999, 999999
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
            # Full heavy evaluation only happens at leaf nodes now.
            val = minimax(board, d - 1, asp_alpha, asp_beta,
                          not is_max, 1, current_id)
            board.pop()

            if use_asp and (val <= asp_alpha or val >= asp_beta):
                failed_asp = True
                break

            ai_score    = ai_evals[move]
            book_bonus  = BOOK_BONUS if move in book_move_set else 0
            order_score = (val * (1 - AI_WEIGHT)) + (ai_score * AI_WEIGHT)
            order_score += book_bonus if is_max else -book_bonus
            results.append((val, order_score, move))

        if failed_asp:
            results = []
            for move in sorted_root:
                board.push(move)
                val = minimax(board, d - 1, -999999, 999999,
              not is_max, 1, current_id)
                board.pop()
                ai_score    = ai_evals[move]
                book_bonus  = BOOK_BONUS if move in book_move_set else 0
                order_score = (val * (1 - AI_WEIGHT)) + (ai_score * AI_WEIGHT)
                order_score += book_bonus if is_max else -book_bonus
                results.append((val, order_score, move))

        results.sort(key=lambda x: x[1], reverse=is_max)
        legal_moves = [r[2] for r in results]

        if results:
            prev_score = int(results[0][0])

        is_book = "book" if results and results[0][2] in book_move_set else ""
        print(f"Depth {d} complete. Nodes: {NODES_COUNT}  Best: {results[0][2].uci() if results else '?'} {is_book}  Score: {results[0][0] if results else '?'}")

        if on_depth and results:
            _root = results[0][2]
            _pv   = get_pv_line(board, d, root_move=_root)
            on_depth(d, [(r[0], r[2]) for r in results[:count]], _pv)

    root_move = results[0][2] if results else None
    pv = get_pv_line(board, depth, root_move=root_move)
    return [(r[0], r[2]) for r in results[:count]], pv