import pygame
import chess
import sys
from search import get_top_moves

# Configuration
BOARD_SIZE = 512
PANEL_WIDTH = 250
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
    # Load and scale piece images from 'Images' directory
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

def draw_board(screen):
    for row in range(DIMENSION):
        for col in range(DIMENSION):
            color = COLOR_LIGHT if (row + col) % 2 == 0 else COLOR_DARK
            pygame.draw.rect(screen, color, pygame.Rect(col * SQ_SIZE, row * SQ_SIZE, SQ_SIZE, SQ_SIZE))

def draw_highlight(screen, square):
    if square is not None:
        row = 7 - (square // 8)
        col = square % 8
        pygame.draw.rect(screen, COLOR_HIGHLIGHT, pygame.Rect(col * SQ_SIZE, row * SQ_SIZE, SQ_SIZE, SQ_SIZE))

def draw_legal_moves(screen, board, selected_square):
    # Draw a dot on all valid destination squares for the selected piece
    if selected_square is not None:
        for move in board.legal_moves:
            if move.from_square == selected_square:
                row = 7 - (move.to_square // 8)
                col = move.to_square % 8
                center_x = col * SQ_SIZE + SQ_SIZE // 2
                center_y = row * SQ_SIZE + SQ_SIZE // 2
                pygame.draw.circle(screen, COLOR_LEGAL_DOT, (center_x, center_y), SQ_SIZE // 6)

def draw_pieces(screen, board):
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row = 7 - (square // 8)
            col = square % 8
            piece_str = piece.symbol()
            screen.blit(IMAGES[piece_str], pygame.Rect(col * SQ_SIZE, row * SQ_SIZE, SQ_SIZE, SQ_SIZE))

def draw_sidebar(screen, font_large, font_small, top_moves, is_calculating, board):
    # Background for the side panel
    pygame.draw.rect(screen, COLOR_PANEL_BG, pygame.Rect(BOARD_SIZE, 0, PANEL_WIDTH, HEIGHT))

    title = font_large.render("Engine Analysis", True, COLOR_TEXT)
    screen.blit(title, (BOARD_SIZE + 20, 20))

    # Check board.turn instead of turn
    turn_str = "White to move" if board.turn == chess.WHITE else "Black to move"
    turn_text = font_small.render(turn_str, True, (180, 180, 180))
    screen.blit(turn_text, (BOARD_SIZE + 20, 60))

    if is_calculating:
        calc_text = font_small.render(f"Calculating Depth {DEPTH}...", True, (200, 200, 100))
        screen.blit(calc_text, (BOARD_SIZE + 20, 100))
    else:
        for i, (eval_score, move) in enumerate(top_moves):
            # Format evaluation display
            eval_str = f"{(eval_score / 100):+.2f}" if abs(eval_score) < 9000 else "Mate"
            text = f"{i+1}. {board.san(move)} ({eval_str})"
            move_surface = font_small.render(text, True, COLOR_TEXT)
            screen.blit(move_surface, (BOARD_SIZE + 20, 110 + i * 40))

def get_promotion_choice(screen):
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
    pygame.display.set_caption("Python Chess Engine - Analysis Mode")
    clock = pygame.time.Clock()

    font_large = pygame.font.SysFont("Arial", 28, bold=True)
    font_small = pygame.font.SysFont("Arial", 20, bold=False)

    load_images()
    board = chess.Board()
    selected_square = None

    top_moves = []
    needs_eval = True
    is_calculating = False

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                sys.exit()

            # Block clicks while engine is calculating
            elif event.type == pygame.MOUSEBUTTONDOWN and not is_calculating:
                location = pygame.mouse.get_pos()

                # Check if click is on the board
                if location[0] < BOARD_SIZE:
                    col = location[0] // SQ_SIZE
                    row = location[1] // SQ_SIZE
                    clicked_square = (7 - row) * 8 + col

                    if selected_square is None:
                        # Only select if it's the current player's piece
                        if board.piece_at(clicked_square) and board.piece_at(clicked_square).color == board.turn:
                            selected_square = clicked_square
                    else:
                        move = chess.Move(selected_square, clicked_square)

                        if board.piece_at(selected_square) and board.piece_at(selected_square).piece_type == chess.PAWN:
                            if chess.square_rank(clicked_square) in [0, 7]:
                                promotion_piece = get_promotion_choice(screen)
                                move = chess.Move(selected_square, clicked_square, promotion=promotion_piece)

                        if move in board.legal_moves:
                            board.push(move)
                            selected_square = None
                            needs_eval = True # Trigger engine recalculation
                            top_moves = []
                        else:
                            if board.piece_at(clicked_square) and board.piece_at(clicked_square).color == board.turn:
                                selected_square = clicked_square
                            else:
                                selected_square = None

        # Draw UI
        draw_board(screen)
        draw_highlight(screen, selected_square)
        draw_legal_moves(screen, board, selected_square)
        draw_pieces(screen, board)
        draw_sidebar(screen, font_large, font_small, top_moves, is_calculating, board)
        pygame.display.flip()

        # Calculate moves after UI updates so "Calculating..." text is shown
        if needs_eval and not board.is_game_over():
            is_calculating = True

            # Force one last UI update to show 'calculating' state before blocking the thread
            draw_sidebar(screen, font_large, font_small, top_moves, is_calculating, board)
            pygame.display.flip()

            top_moves = get_top_moves(board, DEPTH, count=3)
            is_calculating = False
            needs_eval = False

        clock.tick(15)

if __name__ == "__main__":
    main()