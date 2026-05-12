import pygame
import chess
import sys

# Layout sizes
BOARD_SIZE  = 512
BOARD_PAD   = 30
PANEL_WIDTH = 390
BOARD_AREA  = BOARD_SIZE + BOARD_PAD * 2
WIDTH       = BOARD_AREA + PANEL_WIDTH
HEIGHT      = BOARD_AREA
DIMENSION   = 8
SQ_SIZE     = BOARD_SIZE // DIMENSION
DEPTH       = 4

# Colors
C_BG         = ( 10,  10,  17)
C_FRAME      = ( 25,  23,  40)
C_FRAME_LINE = (168, 135,  52)
C_FRAME_IN   = ( 42,  38,  65)
C_SQ_LIGHT   = (237, 214, 179)
C_SQ_DARK    = (165, 117,  74)
C_LAST_L     = (203, 209,  90)
C_LAST_D     = (161, 158,  44)
C_SELECT     = (248, 238,  80)
C_CHECK      = (200,  38,  38)
C_LEGAL_DARK = ( 18,  18,  18)

C_PANEL      = ( 14,  13,  22)
C_PANEL_HDR  = ( 20,  18,  34)
C_DIVIDER    = ( 38,  36,  58)
C_GOLD       = (210, 173,  52)
C_GOLD_DIM   = (120, 100,  28)
C_TEXT       = (232, 228, 218)
C_TEXT_DIM   = (118, 116, 140)
C_EVAL_POS   = ( 96, 196,  86)
C_EVAL_NEG   = (216,  86,  86)
C_EVAL_ZERO  = (130, 128, 152)
C_BTN        = ( 28,  26,  44)
C_BTN_BDR    = ( 52,  50,  78)
C_BTN_EDIT   = ( 36,  18,  18)
C_WARN       = (220, 100,  50)

IMAGES = {}

def load_images():
    # Load and scale pieces
    pieces = ['p', 'n', 'b', 'r', 'q', 'k', 'P', 'N', 'B', 'R', 'Q', 'K']
    try:
        for piece in pieces:
            color = 'w' if piece.isupper() else 'b'
            filename = f"Images/{color}{piece.upper()}.png"
            IMAGES[piece] = pygame.transform.smoothscale(
                pygame.image.load(filename), (SQ_SIZE, SQ_SIZE)
            )
    except FileNotFoundError as e:
        print(f"Error: Image not found! {e}")
        sys.exit()

def get_screen_pos(square, flipped):
    # Convert chess square to screen coords
    rank = chess.square_rank(square)
    file = chess.square_file(square)
    if flipped:
        draw_row, draw_col = rank, 7 - file
    else:
        draw_row, draw_col = 7 - rank, file
    return (BOARD_PAD + draw_col * SQ_SIZE, BOARD_PAD + draw_row * SQ_SIZE)

def get_square_from_pos(pos, flipped):
    # Convert screen coords to chess square
    x, y = pos
    bx, by = x - BOARD_PAD, y - BOARD_PAD
    if bx < 0 or bx >= BOARD_SIZE or by < 0 or by >= BOARD_SIZE:
        return None
    draw_col, draw_row = bx // SQ_SIZE, by // SQ_SIZE
    if flipped:
        rank, file = draw_row, 7 - draw_col
    else:
        rank, file = 7 - draw_row, draw_col
    return chess.square(file, rank)

def is_light_square(square):
    # Check if square is light colored
    return (chess.square_rank(square) + chess.square_file(square)) % 2 == 1

def draw_board(screen, board, last_move, flipped):
    # Fill background and draw borders
    screen.fill(C_BG)
    pygame.draw.rect(screen, C_FRAME, (0, 0, BOARD_AREA, HEIGHT))
    pygame.draw.rect(screen, C_FRAME_LINE, (BOARD_PAD - 3, BOARD_PAD - 3, BOARD_SIZE + 6, BOARD_SIZE + 6), 2)
    pygame.draw.rect(screen, C_FRAME_IN, (BOARD_PAD - 1, BOARD_PAD - 1, BOARD_SIZE + 2, BOARD_SIZE + 2), 1)

    last_from = last_move.from_square if last_move else None
    last_to   = last_move.to_square   if last_move else None

    # Render squares
    for sq in chess.SQUARES:
        light = is_light_square(sq)
        if sq in (last_from, last_to):
            color = C_LAST_L if light else C_LAST_D
        else:
            color = C_SQ_LIGHT if light else C_SQ_DARK
        x, y = get_screen_pos(sq, flipped)
        pygame.draw.rect(screen, color, (x, y, SQ_SIZE, SQ_SIZE))

def draw_coordinates(screen, font, flipped):
    # Draw rank and file labels
    for i in range(DIMENSION):
        actual_rank = i if flipped else 7 - i
        label = str(actual_rank + 1)
        light_sq = (actual_rank + 0) % 2 == 1
        color = C_SQ_DARK if light_sq else C_SQ_LIGHT
        surf = font.render(label, True, color)
        screen.blit(surf, (BOARD_PAD + 3, BOARD_PAD + i * SQ_SIZE + 4))

        actual_file = 7 - i if flipped else i
        label = chr(ord('a') + actual_file)
        light_sq = (0 + actual_file) % 2 == 1
        color = C_SQ_DARK if light_sq else C_SQ_LIGHT
        surf = font.render(label, True, color)
        screen.blit(surf,
                    (BOARD_PAD + i * SQ_SIZE + SQ_SIZE - surf.get_width() - 4,
                     BOARD_PAD + BOARD_SIZE - surf.get_height() - 3))

def draw_highlight(screen, square, flipped):
    # Draw selection highlight
    if square is None: return
    x, y = get_screen_pos(square, flipped)
    hl = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)
    hl.fill((*C_SELECT, 160))
    screen.blit(hl, (x, y))

def draw_check(screen, board, flipped):
    # Highlight king in check
    if not board.is_check(): return
    king_sq = board.king(board.turn)
    if king_sq is None: return
    x, y = get_screen_pos(king_sq, flipped)
    ov = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)
    ov.fill((*C_CHECK, 140))
    screen.blit(ov, (x, y))

def draw_legal_moves(screen, board, selected_square, flipped):
    # Show legal moves for selected piece
    if selected_square is None: return
    dot_surf = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)
    for move in board.legal_moves:
        if move.from_square != selected_square: continue
        x, y = get_screen_pos(move.to_square, flipped)
        dot_surf.fill((0, 0, 0, 0))
        if board.is_capture(move):
            pygame.draw.circle(dot_surf, (*C_LEGAL_DARK, 90),
                               (SQ_SIZE // 2, SQ_SIZE // 2), SQ_SIZE // 2 - 3, 5)
        else:
            pygame.draw.circle(dot_surf, (*C_LEGAL_DARK, 72),
                               (SQ_SIZE // 2, SQ_SIZE // 2), SQ_SIZE // 7)
        screen.blit(dot_surf, (x, y))

def draw_pieces(screen, board, flipped):
    # Draw all pieces
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            x, y = get_screen_pos(square, flipped)
            screen.blit(IMAGES[piece.symbol()], (x, y))

def draw_divider(screen, y, px, pw):
    # Draw horizontal line
    pygame.draw.line(screen, C_DIVIDER, (px + 16, y), (px + pw - 16, y), 1)

def eval_color(score_cp):
    # Get color based on eval score
    if   score_cp >  25: return C_EVAL_POS
    elif score_cp < -25: return C_EVAL_NEG
    else:                return C_EVAL_ZERO

def draw_eval_bar(screen, px, y, pw, score_cp):
    # Render eval bar visualization
    BAR_H    = 8
    BAR_W    = pw - 32
    bar_x    = px + 16
    clamped  = max(-600, min(600, score_cp))
    white_w  = int((clamped + 600) / 1200 * BAR_W)
    black_w  = BAR_W - white_w

    pygame.draw.rect(screen, C_DIVIDER, (bar_x, y, BAR_W, BAR_H), border_radius=4)
    if black_w > 0:
        pygame.draw.rect(screen, ( 36,  34,  52), (bar_x, y, black_w, BAR_H), border_radius=4)
    if white_w > 0:
        pygame.draw.rect(screen, (220, 218, 208), (bar_x + black_w, y, white_w, BAR_H), border_radius=4)

    mid = bar_x + BAR_W // 2
    pygame.draw.line(screen, C_BG, (mid, y), (mid, y + BAR_H), 1)
    return BAR_H + 6

def draw_sidebar(screen, fonts, top_moves, is_calculating, board, edit_mode, selected_edit_piece):
    # Main sidebar logic
    px = BOARD_AREA
    pw = PANEL_WIDTH
    palette_rects = {}

    pygame.draw.rect(screen, C_PANEL, (px, 0, pw, HEIGHT))

    hdr_h = 68
    pygame.draw.rect(screen, C_PANEL_HDR, (px, 0, pw, hdr_h))
    pygame.draw.line(screen, C_GOLD,     (px, hdr_h),     (px + pw, hdr_h),     2)
    pygame.draw.line(screen, C_GOLD_DIM, (px, hdr_h + 3), (px + pw, hdr_h + 3), 1)

    title = fonts['title'].render(
        "BOARD EDITOR" if edit_mode else "ENGINE ANALYSIS", True,
        (220, 100, 80) if edit_mode else C_GOLD)
    screen.blit(title, (px + 20, 20))

    sub = fonts['tiny'].render("Alpha-Beta + Neural Net", True, C_TEXT_DIM)
    screen.blit(sub, (px + 20, hdr_h - sub.get_height() - 6))

    cy = hdr_h + 14

    # New prominent turn indicator
    turn_str = "WHITE TO MOVE" if board.turn == chess.WHITE else "BLACK TO MOVE"
    turn_bg  = (240, 240, 240) if board.turn == chess.WHITE else (25, 25, 25)
    turn_fg  = (10, 10, 10) if board.turn == chess.WHITE else (240, 240, 240)

    pygame.draw.rect(screen, turn_bg, (px + 16, cy, pw - 32, 34), border_radius=6)
    pygame.draw.rect(screen, C_GOLD_DIM if board.turn == chess.BLACK else turn_bg, (px + 16, cy, pw - 32, 34), 1, border_radius=6)

    t_surf = fonts['title'].render(turn_str, True, turn_fg)
    screen.blit(t_surf, (px + 16 + (pw - 32 - t_surf.get_width()) // 2, cy + (34 - t_surf.get_height()) // 2))
    cy += 34 + 14

    draw_divider(screen, cy, px, pw)
    cy += 12

    if edit_mode:
        # Editor layout
        for text in ["[C] Clear board", "[R] Reset to start", "[T] Toggle side to move"]:
            s = fonts['info'].render(text, True, C_TEXT_DIM)
            screen.blit(s, (px + 20, cy))
            cy += s.get_height() + 6

        cy += 8
        draw_divider(screen, cy, px, pw)
        cy += 12

        pal = fonts['info'].render("Place (L-click)   Remove (R-click)", True, C_TEXT_DIM)
        screen.blit(pal, (px + 20, cy))
        cy += pal.get_height() + 10

        white_pieces = ['P', 'N', 'B', 'R', 'Q', 'K']
        black_pieces = ['p', 'n', 'b', 'r', 'q', 'k']
        sq = 40

        # Draw piece palette
        for row, pieces in enumerate((white_pieces, black_pieces)):
            for i, p in enumerate(pieces):
                rx = px + 18 + i * (sq + 6)
                ry = cy + row * (sq + 8)
                bg = C_GOLD if selected_edit_piece == p else C_BTN
                bd = C_GOLD if selected_edit_piece == p else C_BTN_BDR
                pygame.draw.rect(screen, bg, (rx, ry, sq, sq), border_radius=6)
                pygame.draw.rect(screen, bd, (rx, ry, sq, sq), 1, border_radius=6)
                img = pygame.transform.smoothscale(IMAGES[p], (sq - 4, sq - 4))
                screen.blit(img, (rx + 2, ry + 2))

                # Store active rects
                palette_rects[p] = pygame.Rect(rx, ry, sq, sq)

        cy += 2 * (sq + 8) + 14

    else:
        # Analysis layout
        if not board.is_valid():
            warn = fonts['info'].render("⚠  Invalid position", True, C_WARN)
            screen.blit(warn, (px + 20, cy))
            cy += warn.get_height() + 8

        elif is_calculating:
            calc = fonts['info'].render(f"Searching depth {DEPTH} …", True, C_TEXT_DIM)
            screen.blit(calc, (px + 20, cy))
            cy += calc.get_height() + 8
            ticks = (pygame.time.get_ticks() // 400) % 4
            dots = fonts['title'].render("●" * ticks + "○" * (3 - ticks), True, C_GOLD_DIM)
            screen.blit(dots, (px + 20, cy))
            cy += dots.get_height() + 8

        elif top_moves:
            top_score = top_moves[0][0]
            cy += draw_eval_bar(screen, px, cy, pw, top_score)

            if abs(top_score) >= 90000:
                eval_str = "Forced Mate"
                e_col    = C_EVAL_POS if top_score > 0 else C_EVAL_NEG
            else:
                cp = top_score / 100
                sign = "+" if cp >= 0 else ""
                eval_str = f"{sign}{cp:.2f}"
                e_col = eval_color(top_score)

            eval_surf = fonts['eval'].render(eval_str, True, e_col)
            screen.blit(eval_surf, (px + pw - eval_surf.get_width() - 18, cy))
            cy += eval_surf.get_height() + 8
            draw_divider(screen, cy, px, pw)
            cy += 12

            lbl = fonts['tiny'].render("TOP MOVES", True, C_TEXT_DIM)
            screen.blit(lbl, (px + 20, cy))
            cy += lbl.get_height() + 10

            # Render top engine moves
            for i, (score, move) in enumerate(top_moves):
                try: move_san = board.san(move)
                except Exception: move_san = move.uci()

                card_h = 52
                card_x = px + 14
                card_w = pw - 28
                card_y = cy

                card_bg = C_BTN if i == 0 else C_PANEL
                pygame.draw.rect(screen, card_bg, (card_x, card_y, card_w, card_h), border_radius=8)
                if i == 0:
                    pygame.draw.rect(screen, C_GOLD, (card_x, card_y, card_w, card_h), 1, border_radius=8)

                badge_col = C_GOLD if i == 0 else C_BTN_BDR
                bx, bcy = card_x + 18, card_y + card_h // 2
                pygame.draw.circle(screen, badge_col, (bx, bcy), 14)
                n_surf = fonts['badge'].render(str(i + 1), True, C_PANEL if i == 0 else C_TEXT_DIM)
                screen.blit(n_surf, (bx - n_surf.get_width() // 2, bcy - n_surf.get_height() // 2))

                m_surf = fonts['move'].render(move_san, True, C_TEXT if i == 0 else C_TEXT_DIM)
                screen.blit(m_surf, (card_x + 40, card_y + 8))

                if abs(score) >= 90000:
                    sc_str = "Mate"
                    sc_col = C_EVAL_POS if score > 0 else C_EVAL_NEG
                else:
                    cp = score / 100
                    sc_str = f"{'+' if cp >= 0 else ''}{cp:.2f}"
                    sc_col = eval_color(score)

                sc_surf = fonts['move'].render(sc_str, True, sc_col)
                screen.blit(sc_surf, (card_x + 40, card_y + card_h - sc_surf.get_height() - 8))

                mini_w = card_w - 100
                mini_x = card_x + card_w - mini_w - 16
                mini_y = card_y + card_h // 2 - 3
                clamped = max(-600, min(600, score))
                ww = int((clamped + 600) / 1200 * mini_w)
                pygame.draw.rect(screen, C_DIVIDER, (mini_x, mini_y, mini_w, 6), border_radius=3)
                if ww > 0:
                    pygame.draw.rect(screen, (200, 196, 186), (mini_x + (mini_w - ww), mini_y, ww, 6), border_radius=3)

                cy += card_h + 6

    # Bottom buttons
    btn_h  = 40
    btn_y1 = HEIGHT - btn_h * 2 - 22
    btn_y2 = HEIGHT - btn_h - 10
    bx     = px + 14
    bw     = pw - 28

    for btn_y, label, is_active_edit in (
            (btn_y1, "Exit Editor  [E]" if edit_mode else "Board Editor  [E]", edit_mode),
            (btn_y2, "Flip Board  [F]", False)):

        bg = C_BTN_EDIT if is_active_edit else C_BTN
        bd = (180, 80, 60) if is_active_edit else C_BTN_BDR
        pygame.draw.rect(screen, bg, (bx, btn_y, bw, btn_h), border_radius=8)
        pygame.draw.rect(screen, bd, (bx, btn_y, bw, btn_h), 1, border_radius=8)
        t = fonts['btn'].render(label, True, C_TEXT)
        screen.blit(t, (bx + (bw - t.get_width()) // 2, btn_y + (btn_h - t.get_height()) // 2))

    edit_button_rect = pygame.Rect(bx, btn_y1, bw, btn_h)
    flip_button_rect = pygame.Rect(bx, btn_y2, bw, btn_h)

    return flip_button_rect, edit_button_rect, palette_rects

def get_promotion_choice(screen):
    # Render promotion menu overlay
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 180))
    screen.blit(overlay, (0, 0))

    font = pygame.font.SysFont("Georgia", 28, bold=True)
    txt  = font.render("Promote:  Q · R · B · N", True, C_GOLD)
    screen.blit(txt, (WIDTH // 2 - txt.get_width() // 2,
                      HEIGHT // 2 - txt.get_height() // 2))
    pygame.display.flip()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q: return chess.QUEEN
                if event.key == pygame.K_r: return chess.ROOK
                if event.key == pygame.K_b: return chess.BISHOP
                if event.key == pygame.K_n: return chess.KNIGHT