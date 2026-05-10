import pygame
import chess
import sys
import concurrent.futures
from search import get_top_moves

# Configuration
BOARD_SIZE = 512
PANEL_WIDTH = 400
WIDTH, HEIGHT = BOARD_SIZE + PANEL_WIDTH, BOARD_SIZE
DIMENSION = 8
SQ_SIZE = BOARD_SIZE // DIMENSION
DEPTH = 4

# Colors
COLOR_LIGHT = (240, 217, 181)
COLOR_DARK = (181, 136, 99)
COLOR_HIGHLIGHT = (186, 202, 68)
COLOR_LEGAL_DOT = (100, 100, 100)
COLOR_PANEL_BG = (40, 40, 40)
COLOR_TEXT = (255, 255, 255)

IMAGES = {}

def load_images():
    # Load and scale piece images
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
    # Convert chess square to screen coordinates
    rank = chess.square_rank(square)
    file = chess.square_file(square)
    if flipped:
        draw_row = rank
        draw_col = 7 - file
    else:
        draw_row = 7 - rank
        draw_col = file
    return draw_col * SQ_SIZE, draw_row * SQ_SIZE

def get_square_from_pos(pos, flipped):
    # Convert screen coordinates to chess square
    x, y = pos
    if x >= BOARD_SIZE:
        return None
    draw_col = x // SQ_SIZE
    draw_row = y // SQ_SIZE

    if flipped:
        rank = draw_row
        file = 7 - draw_col
    else:
        rank = 7 - draw_row
        file = draw_col
    return chess.square(file, rank)

def draw_board(screen):
    # Draw background squares
    for row in range(DIMENSION):
        for col in range(DIMENSION):
            color = COLOR_LIGHT if (row + col) % 2 == 0 else COLOR_DARK
            pygame.draw.rect(screen, color, pygame.Rect(col * SQ_SIZE, row * SQ_SIZE, SQ_SIZE, SQ_SIZE))

def draw_coordinates(screen, font_coord, flipped):
    # Draw a-h and 1-8 coordinates
    for i in range(DIMENSION):
        # Ranks
        square_color_is_light = (i + 0) % 2 == 0
        text_color = COLOR_DARK if square_color_is_light else COLOR_LIGHT
        actual_rank = i if flipped else 7 - i
        rank_char = str(actual_rank + 1)
        text = font_coord.render(rank_char, True, text_color)
        screen.blit(text, (5, i * SQ_SIZE + 5))

        # Files
        square_color_is_light = (7 + i) % 2 == 0
        text_color = COLOR_DARK if square_color_is_light else COLOR_LIGHT
        actual_file = 7 - i if flipped else i
        file_char = chr(ord('a') + actual_file)
        text = font_coord.render(file_char, True, text_color)
        screen.blit(text, (i * SQ_SIZE + SQ_SIZE - 15, BOARD_SIZE - 22))

def draw_highlight(screen, square, flipped):
    # Highlight selected square
    if square is not None:
        x, y = get_screen_pos(square, flipped)
        pygame.draw.rect(screen, COLOR_HIGHLIGHT, pygame.Rect(x, y, SQ_SIZE, SQ_SIZE))

def draw_legal_moves(screen, board, selected_square, flipped):
    # Draw dots for legal moves
    if selected_square is not None:
        for move in board.legal_moves:
            if move.from_square == selected_square:
                x, y = get_screen_pos(move.to_square, flipped)
                center_x = x + SQ_SIZE // 2
                center_y = y + SQ_SIZE // 2
                pygame.draw.circle(screen, COLOR_LEGAL_DOT, (center_x, center_y), SQ_SIZE // 6)

def draw_pieces(screen, board, flipped):
    # Draw pieces on the board
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            x, y = get_screen_pos(square, flipped)
            piece_str = piece.symbol()
            screen.blit(IMAGES[piece_str], pygame.Rect(x, y, SQ_SIZE, SQ_SIZE))

def draw_sidebar(screen, font_large, font_small, top_moves, is_calculating, board, edit_mode, selected_edit_piece):
    # Panel background
    pygame.draw.rect(screen, COLOR_PANEL_BG, pygame.Rect(BOARD_SIZE, 0, PANEL_WIDTH, HEIGHT))

    # Title
    title_text = "Board Editor" if edit_mode else "Engine Analysis"
    title_color = (255, 100, 100) if edit_mode else COLOR_TEXT
    title = font_large.render(title_text, True, title_color)
    screen.blit(title, (BOARD_SIZE + 20, 20))

    turn_str = "White to move" if board.turn == chess.WHITE else "Black to move"

    if edit_mode:
        # Editor Instructions
        c_text = font_small.render("[C] Clear Board", True, (180, 180, 180))
        r_text = font_small.render("[R] Reset Start Pos", True, (180, 180, 180))
        t_text = font_small.render(f"[T] Turn: {turn_str}", True, (200, 200, 100))

        screen.blit(c_text, (BOARD_SIZE + 20, 60))
        screen.blit(r_text, (BOARD_SIZE + 20, 90))
        screen.blit(t_text, (BOARD_SIZE + 20, 120))

        pal_text = font_small.render("Palette (L-Click place, R-Click remove):", True, COLOR_TEXT)
        screen.blit(pal_text, (BOARD_SIZE + 20, 160))

        # Draw piece palette
        white_pieces = ['P', 'N', 'B', 'R', 'Q', 'K']
        black_pieces = ['p', 'n', 'b', 'r', 'q', 'k']

        for i, p in enumerate(white_pieces):
            rect = pygame.Rect(BOARD_SIZE + 20 + i*34, 200, 32, 32)
            if selected_edit_piece == p:
                pygame.draw.rect(screen, COLOR_HIGHLIGHT, rect)
            img = pygame.transform.smoothscale(IMAGES[p], (32, 32))
            screen.blit(img, rect)

        for i, p in enumerate(black_pieces):
            rect = pygame.Rect(BOARD_SIZE + 20 + i*34, 240, 32, 32)
            if selected_edit_piece == p:
                pygame.draw.rect(screen, COLOR_HIGHLIGHT, rect)
            img = pygame.transform.smoothscale(IMAGES[p], (32, 32))
            screen.blit(img, rect)

    else:
        # Normal Analysis Mode
        turn_text = font_small.render(turn_str, True, (180, 180, 180))
        screen.blit(turn_text, (BOARD_SIZE + 20, 60))

        if not board.is_valid():
            err_text = font_small.render("Invalid Position (Missing Kings?)", True, (255, 100, 100))
            screen.blit(err_text, (BOARD_SIZE + 20, 100))
        elif is_calculating:
            calc_text = font_small.render(f"Calculating Depth {DEPTH}...", True, (200, 200, 100))
            screen.blit(calc_text, (BOARD_SIZE + 20, 100))
        else:
            for i, (eval_score, move) in enumerate(top_moves):
                eval_str = f"{(eval_score / 100):+.2f}" if abs(eval_score) < 90000 else "Mate"
                try:
                    move_san = board.san(move)
                except:
                    move_san = move.uci()

                text = f"{i+1}. {move_san} ({eval_str})"
                move_surface = font_small.render(text, True, COLOR_TEXT)
                screen.blit(move_surface, (BOARD_SIZE + 20, 110 + i * 40))

# Sidebar Buttons
    edit_button_rect = pygame.Rect(BOARD_SIZE + 20, HEIGHT - 110, PANEL_WIDTH - 40, 40)
    pygame.draw.rect(screen, (80, 80, 80), edit_button_rect, border_radius=5)
    btn_text = "Exit Editor (key: E)" if edit_mode else "Edit Mode (key: E)"
    edit_text = font_small.render(btn_text, True, COLOR_TEXT)
    screen.blit(edit_text, (BOARD_SIZE + 35, HEIGHT - 100))

    flip_button_rect = pygame.Rect(BOARD_SIZE + 20, HEIGHT - 60, PANEL_WIDTH - 40, 40)
    pygame.draw.rect(screen, (80, 80, 80), flip_button_rect, border_radius=5)
    flip_text = font_small.render("Flip Board (key: F)", True, COLOR_TEXT)
    screen.blit(flip_text, (BOARD_SIZE + 35, HEIGHT - 50))

    return flip_button_rect, edit_button_rect

def get_promotion_choice(screen):
    # Overlay for pawn promotion
    overlay = pygame.Surface((WIDTH, HEIGHT))
    overlay.set_alpha(200)
    overlay.fill((0, 0, 0))
    screen.blit(overlay, (0, 0))

    font = pygame.font.SysFont("Arial", 30, bold=True)
    text1 = font.render("Press: Q, R, B, or N", True, (255, 255, 255))
    screen.blit(text1, (WIDTH // 2 - text1.get_width() // 2, HEIGHT // 2))
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

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Python Chess Engine")
    clock = pygame.time.Clock()

    font_large = pygame.font.SysFont("Arial", 28, bold=True)
    font_small = pygame.font.SysFont("Arial", 20, bold=False)
    font_coord = pygame.font.SysFont("Arial", 14, bold=True)

    load_images()
    board = chess.Board()
    selected_square = None

    top_moves = []
    needs_eval = True
    is_calculating = False
    board_flipped = False
    edit_mode = False
    selected_edit_piece = 'P'

    flip_button_rect = pygame.Rect(0,0,0,0)
    edit_button_rect = pygame.Rect(0,0,0,0)

    # Thread pool for background engine calculations
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = None
    last_eval_fen = ""

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                sys.exit()

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_f:
                    board_flipped = not board_flipped
                elif event.key == pygame.K_e:
                    edit_mode = not edit_mode
                    selected_square = None
                    if not edit_mode: needs_eval = True

                # Setup controls when in edit mode
                if edit_mode:
                    if event.key == pygame.K_c:
                        board.clear()
                    elif event.key == pygame.K_r:
                        board.reset()
                    elif event.key == pygame.K_t:
                        board.turn = not board.turn

            elif event.type == pygame.MOUSEBUTTONDOWN:
                location = pygame.mouse.get_pos()

                # Handle UI button clicks
                if flip_button_rect.collidepoint(location):
                    board_flipped = not board_flipped
                    continue
                if edit_button_rect.collidepoint(location):
                    edit_mode = not edit_mode
                    selected_square = None
                    if not edit_mode: needs_eval = True
                    continue

                # Check palette clicks in edit mode
                if edit_mode and location[0] >= BOARD_SIZE:
                    if 200 <= location[1] <= 232:
                        idx = (location[0] - (BOARD_SIZE + 20)) // 34
                        if 0 <= idx < 6: selected_edit_piece = ['P', 'N', 'B', 'R', 'Q', 'K'][idx]
                    elif 240 <= location[1] <= 272:
                        idx = (location[0] - (BOARD_SIZE + 20)) // 34
                        if 0 <= idx < 6: selected_edit_piece = ['p', 'n', 'b', 'r', 'q', 'k'][idx]
                    continue

                clicked_square = get_square_from_pos(location, board_flipped)

                if clicked_square is not None:
                    if edit_mode:
                        # Editor placement / removal
                        if event.button == 1: # Left Click: Place piece
                            new_piece = chess.Piece.from_symbol(selected_edit_piece)

                            # Prevent multiple kings: remove existing king of the same color
                            if new_piece.piece_type == chess.KING:
                                for sq in list(board.pieces(chess.KING, new_piece.color)):
                                    board.remove_piece_at(sq)

                            board.set_piece_at(clicked_square, new_piece)
                        elif event.button == 3: # Right Click: Remove piece
                            board.remove_piece_at(clicked_square)
                    else:
                        # Normal Play Mode (Left Click only)
                        if event.button == 1:
                            if selected_square is None:
                                if board.piece_at(clicked_square) and board.piece_at(clicked_square).color == board.turn:
                                    selected_square = clicked_square
                            else:
                                move = chess.Move(selected_square, clicked_square)
                                piece = board.piece_at(selected_square)

                                if piece and piece.piece_type == chess.PAWN:
                                    if (piece.color == chess.WHITE and chess.square_rank(clicked_square) == 7) or \
                                       (piece.color == chess.BLACK and chess.square_rank(clicked_square) == 0):
                                        promotion_piece = get_promotion_choice(screen)
                                        move = chess.Move(selected_square, clicked_square, promotion=promotion_piece)

                                if move in board.legal_moves:
                                    board.push(move)
                                    selected_square = None
                                    needs_eval = True
                                    top_moves = []
                                else:
                                    if board.piece_at(clicked_square) and board.piece_at(clicked_square).color == board.turn:
                                        selected_square = clicked_square
                                    else:
                                        selected_square = None

        # Process background thread results
        if future is not None and future.done() and is_calculating:
            is_calculating = False
            if board.fen() == last_eval_fen:
                top_moves = future.result()

        # Start new background calculation
        if needs_eval and not is_calculating and not board.is_game_over() and not edit_mode:
            if board.is_valid():
                last_eval_fen = board.fen()
                future = executor.submit(get_top_moves, board.copy(), DEPTH, 3)
                is_calculating = True
                needs_eval = False
            else:
                top_moves = []
                needs_eval = False

        # Draw UI
        draw_board(screen)
        draw_coordinates(screen, font_coord, board_flipped)

        if not edit_mode:
            draw_highlight(screen, selected_square, board_flipped)
            draw_legal_moves(screen, board, selected_square, board_flipped)

        draw_pieces(screen, board, board_flipped)

        flip_button_rect, edit_button_rect = draw_sidebar(
            screen, font_large, font_small, top_moves, is_calculating, board, edit_mode, selected_edit_piece
        )

        pygame.display.flip()
        clock.tick(30)

if __name__ == "__main__":
    main()