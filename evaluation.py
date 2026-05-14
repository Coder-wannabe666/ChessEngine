import chess

# --- Piece-Square Tables (PST) ---
# Defined from White's perspective; auto-mirrored for Black.

PAWN_PST = [
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0
]

PAWN_ENDGAME_PST = [
     0,   0,   0,   0,   0,   0,   0,   0,
    120, 120, 120, 120, 120, 120, 120, 120,
     80,  80,  80,  80,  80,  80,  80,  80,
     50,  50,  50,  50,  50,  50,  50,  50,
     30,  30,  30,  30,  30,  30,  30,  30,
     15,  15,  15,  15,  15,  15,  15,  15,
      0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0
]

KNIGHT_PST = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50
]

BISHOP_PST = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20
]

ROOK_PST = [
      0,  0,  0,  0,  0,  0,  0,  0,
      5, 10, 10, 10, 10, 10, 10,  5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
      0,  0,  0,  5,  5,  0,  0,  0
]

QUEEN_PST = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20
]

KING_PST = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20
]

KING_ENDGAME_PST = [
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50
]

MG_TABLES = {
    chess.PAWN:   PAWN_PST,
    chess.KNIGHT: KNIGHT_PST,
    chess.BISHOP: BISHOP_PST,
    chess.ROOK:   ROOK_PST,
    chess.QUEEN:  QUEEN_PST,
    chess.KING:   KING_PST,
}

EG_TABLES = {
    chess.PAWN:   PAWN_ENDGAME_PST,
    chess.KNIGHT: KNIGHT_PST,
    chess.BISHOP: BISHOP_PST,
    chess.ROOK:   ROOK_PST,
    chess.QUEEN:  QUEEN_PST,
    chess.KING:   KING_ENDGAME_PST,
}

PIECE_VALUES = {
    chess.PAWN:   100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK:   500,
    chess.QUEEN:  900,
    chess.KING:   20000
}

# Passed pawn bonus indexed by rank (0–7 = rank 1–8 from own perspective).
# Endgame values are much higher — a passed pawn is a game-deciding asset.
PASSED_BONUS_MG = [0,  5, 10, 20,  40,  70, 110,  0]
PASSED_BONUS_EG = [0, 20, 40, 70, 120, 180, 260,  0]

def is_endgame(board):
    return len(board.piece_map()) <= 10

# ── Pawn structure ──────────────────────────────────────────────────────

def _is_passed_pawn(piece_map, square, color):
    # Check if pawn is passed.
    file  = chess.square_file(square)
    rank  = chess.square_rank(square)
    enemy = not color

    check_files = {file}
    if file > 0: check_files.add(file - 1)
    if file < 7: check_files.add(file + 1)

    ranks_ahead = range(rank + 1, 8) if color == chess.WHITE else range(0, rank)
    ranks_set   = set(ranks_ahead)

    for sq, p in piece_map.items():
        if (p.piece_type == chess.PAWN and p.color == enemy
                and chess.square_file(sq) in check_files
                and chess.square_rank(sq) in ranks_set):
            return False
    return True


def pawn_structure_score(piece_map, endgame):
    # Eval passed, isolated, doubled pawns.
    score = 0

    white_pawns, black_pawns = [], []
    white_files, black_files = [], []

    for sq, p in piece_map.items():
        if p.piece_type == chess.PAWN:
            if p.color == chess.WHITE:
                white_pawns.append(sq)
                white_files.append(chess.square_file(sq))
            else:
                black_pawns.append(sq)
                black_files.append(chess.square_file(sq))

    for sq in white_pawns:
        rank = chess.square_rank(sq)
        file = chess.square_file(sq)

        if _is_passed_pawn(piece_map, sq, chess.WHITE):
            score += PASSED_BONUS_EG[rank] if endgame else PASSED_BONUS_MG[rank]

        if not any(f in white_files for f in (file - 1, file + 1) if 0 <= f <= 7):
            score -= 25 if endgame else 15   # isolated

        if white_files.count(file) > 1:
            score -= 15                       # doubled

    for sq in black_pawns:
        rank = chess.square_rank(sq)
        file = chess.square_file(sq)

        if _is_passed_pawn(piece_map, sq, chess.BLACK):
            score -= PASSED_BONUS_EG[7 - rank] if endgame else PASSED_BONUS_MG[7 - rank]

        if not any(f in black_files for f in (file - 1, file + 1) if 0 <= f <= 7):
            score += 25 if endgame else 15   # isolated

        if black_files.count(file) > 1:
            score += 15                       # doubled

    return score


# ── Hanging / loose piece detection ─────────────────────────────────────

def hanging_piece_score(board, piece_map):
    # Penalize hanging pieces.
    score = 0

    for square, piece in piece_map.items():
        pt = piece.piece_type
        if pt in (chess.KING, chess.PAWN):
            continue

        enemy     = not piece.color
        piece_val = PIECE_VALUES[pt]

        if not board.is_attacked_by(enemy, square):
            continue

        attacker_sqs = board.attackers(enemy, square)
        min_atk_val  = min(
            PIECE_VALUES.get(board.piece_at(sq).piece_type, 0)
            for sq in attacker_sqs if board.piece_at(sq)
        )

        defended = board.is_attacked_by(piece.color, square)

        if min_atk_val < piece_val:
            penalty = (piece_val - min_atk_val) if not defended else (piece_val - min_atk_val) // 3
        elif not defended:
            penalty = piece_val // 8
        else:
            continue

        score += -penalty if piece.color == chess.WHITE else penalty

    return score


# ── Castling & king safety ───────────────────────────────────────────────

_CASTLE_KINGSIDE_EMPTY  = {chess.WHITE: {chess.F1, chess.G1},
                            chess.BLACK: {chess.F8, chess.G8}}
_CASTLE_QUEENSIDE_EMPTY = {chess.WHITE: {chess.B1, chess.C1, chess.D1},
                            chess.BLACK: {chess.B8, chess.C8, chess.D8}}

_CASTLED_KING_SQUARES   = {chess.WHITE: {chess.G1, chess.C1},
                            chess.BLACK: {chess.G8, chess.C8}}

def castling_score(board, piece_map, endgame):
    # Reward castling, penalize lost rights.
    if endgame:
        return 0

    score = 0

    for color in (chess.WHITE, chess.BLACK):
        sign       = 1 if color == chess.WHITE else -1
        king_sq    = board.king(color)
        if king_sq is None:
            continue

        king_rank  = chess.square_rank(king_sq)
        king_file  = chess.square_file(king_sq)
        back_rank  = 0 if color == chess.WHITE else 7

        has_ks = board.has_kingside_castling_rights(color)
        has_qs = board.has_queenside_castling_rights(color)

        if king_sq in _CASTLED_KING_SQUARES[color]:
            score += sign * 60
        elif has_ks or has_qs:
            score += sign * 15
            occupied = set(piece_map.keys())
            if has_ks and not (_CASTLE_KINGSIDE_EMPTY[color] & occupied):
                score += sign * 25
            if has_qs and not (_CASTLE_QUEENSIDE_EMPTY[color] & occupied):
                score += sign * 20
        else:
            score -= sign * 50

        if king_rank == back_rank and king_file == 4:
            for rook_sq, p in piece_map.items():
                if p.piece_type == chess.ROOK and p.color == color:
                    if chess.square_rank(rook_sq) == back_rank:
                        rook_file = chess.square_file(rook_sq)
                        if rook_file < king_file or rook_file > king_file:
                            score -= sign * 30

    return score


# ── Main evaluation ──────────────────────────────────────────────────────

def evaluate_board(board):
    # Full static eval.
    piece_map = board.piece_map()
    endgame   = len(piece_map) <= 10
    tables    = EG_TABLES if endgame else MG_TABLES

    score = 0
    for square, piece in piece_map.items():
        pt     = piece.piece_type
        pst_sq = chess.square_mirror(square) if piece.color == chess.WHITE else square
        val    = PIECE_VALUES.get(pt, 0) + tables[pt][pst_sq]
        score += val if piece.color == chess.WHITE else -val

    score += pawn_structure_score(piece_map, endgame)
    score += hanging_piece_score(board, piece_map)
    score += castling_score(board, piece_map, endgame)

    return score

def fast_material_pst_eval(board, piece_map):
    # Quick eval for Q-search and pruning.
    endgame = len(piece_map) <= 10
    tables = EG_TABLES if endgame else MG_TABLES
    score = 0
    for square, piece in piece_map.items():
        pt = piece.piece_type
        pst_sq = chess.square_mirror(square) if piece.color == chess.WHITE else square
        val = PIECE_VALUES.get(pt, 0) + tables[pt][pst_sq]
        score += val if piece.color == chess.WHITE else -val
    return score