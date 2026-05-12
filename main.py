import pygame
import chess
import sys
import concurrent.futures
from search import get_top_moves
import gui

def main():
    pygame.init()

    VIRTUAL_WIDTH  = gui.WIDTH
    VIRTUAL_HEIGHT = gui.HEIGHT

    screen         = pygame.display.set_mode((VIRTUAL_WIDTH, VIRTUAL_HEIGHT), pygame.RESIZABLE)
    render_surface = pygame.Surface((VIRTUAL_WIDTH, VIRTUAL_HEIGHT))

    pygame.display.set_caption("Chess Engine")
    clock = pygame.time.Clock()

    current_w, current_h = VIRTUAL_WIDTH, VIRTUAL_HEIGHT
    offset_x, offset_y   = 0, 0
    scale = 1.0

    def try_fonts(names, size, bold=False):
        for name in names:
            try:
                f = pygame.font.SysFont(name, size, bold=bold)
                f.render("A", True, (0, 0, 0))
                return f
            except Exception:
                pass
        return pygame.font.SysFont(None, size, bold=bold)

    fonts = {
        'title': try_fonts(["Georgia", "Palatino Linotype", "Palatino"], 19, bold=True),
        'eval' : try_fonts(["Consolas", "Courier New", "Courier"], 22, bold=True),
        'move' : try_fonts(["Consolas", "Courier New", "Courier"], 17),
        'info' : try_fonts(["Trebuchet MS", "Tahoma", "Verdana"], 16),
        'btn'  : try_fonts(["Trebuchet MS", "Tahoma", "Verdana"], 15),
        'badge': try_fonts(["Georgia", "Palatino Linotype"], 14, bold=True),
        'tiny' : try_fonts(["Trebuchet MS", "Tahoma"], 13),
        'coord': try_fonts(["Georgia", "Palatino Linotype"], 12, bold=True),
    }

    gui.load_images()

    board               = chess.Board()
    selected_square     = None
    last_move           = None
    top_moves           = []
    pv_line             = []     # ← Principal Variation line
    needs_eval          = True
    is_calculating      = False
    board_flipped       = False
    edit_mode           = False
    selected_edit_piece = 'P'

    flip_button_rect = pygame.Rect(0, 0, 0, 0)
    edit_button_rect = pygame.Rect(0, 0, 0, 0)
    palette_rects    = {}

    executor      = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future        = None
    last_eval_fen = ""

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                sys.exit()

            elif event.type == pygame.VIDEORESIZE:
                current_w, current_h = event.w, event.h
                scale    = min(current_w / VIRTUAL_WIDTH, current_h / VIRTUAL_HEIGHT)
                offset_x = (current_w - int(VIRTUAL_WIDTH  * scale)) // 2
                offset_y = (current_h - int(VIRTUAL_HEIGHT * scale)) // 2

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_f:
                    board_flipped = not board_flipped
                elif event.key == pygame.K_e:
                    edit_mode = not edit_mode
                    selected_square = None
                    if not edit_mode:
                        needs_eval = True

                if edit_mode:
                    if event.key == pygame.K_c:
                        board.clear()
                    elif event.key == pygame.K_r:
                        board.reset()
                    elif event.key == pygame.K_t:
                        board.turn = not board.turn

            elif event.type == pygame.MOUSEBUTTONDOWN:
                raw_x, raw_y = event.pos
                virt_x = int((raw_x - offset_x) / scale)
                virt_y = int((raw_y - offset_y) / scale)
                location = (virt_x, virt_y)

                if not (0 <= virt_x <= VIRTUAL_WIDTH and 0 <= virt_y <= VIRTUAL_HEIGHT):
                    continue

                if flip_button_rect.collidepoint(location):
                    board_flipped = not board_flipped
                    continue
                if edit_button_rect.collidepoint(location):
                    edit_mode = not edit_mode
                    selected_square = None
                    if not edit_mode:
                        needs_eval = True
                    continue

                if edit_mode and location[0] >= gui.BOARD_AREA:
                    for piece_char, rect in palette_rects.items():
                        if rect.collidepoint(location):
                            selected_edit_piece = piece_char
                            break
                    continue

                clicked_square = gui.get_square_from_pos(location, board_flipped)

                if clicked_square is not None:
                    if edit_mode:
                        if event.button == 1:
                            new_piece = chess.Piece.from_symbol(selected_edit_piece)
                            if new_piece.piece_type == chess.KING:
                                for sq in list(board.pieces(chess.KING, new_piece.color)):
                                    board.remove_piece_at(sq)
                            board.set_piece_at(clicked_square, new_piece)
                        elif event.button == 3:
                            board.remove_piece_at(clicked_square)
                    else:
                        if event.button == 1:
                            if selected_square is None:
                                if (board.piece_at(clicked_square) and
                                        board.piece_at(clicked_square).color == board.turn):
                                    selected_square = clicked_square
                            else:
                                move  = chess.Move(selected_square, clicked_square)
                                piece = board.piece_at(selected_square)

                                if piece and piece.piece_type == chess.PAWN:
                                    if ((piece.color == chess.WHITE and
                                         chess.square_rank(clicked_square) == 7) or
                                        (piece.color == chess.BLACK and
                                         chess.square_rank(clicked_square) == 0)):
                                        promo = gui.get_promotion_choice(screen)
                                        move  = chess.Move(selected_square, clicked_square,
                                                           promotion=promo)

                                if move in board.legal_moves:
                                    last_move = move
                                    board.push(move)
                                    selected_square = None
                                    executor.shutdown(wait=False, cancel_futures=True)
                                    executor       = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                                    is_calculating = False
                                    needs_eval     = True
                                    top_moves      = []
                                    pv_line        = []
                                else:
                                    if (board.piece_at(clicked_square) and
                                            board.piece_at(clicked_square).color == board.turn):
                                        selected_square = clicked_square
                                    else:
                                        selected_square = None

        # ── Process background thread result ──────────────────────────
        if future is not None and future.done() and is_calculating:
            is_calculating = False
            if board.fen() == last_eval_fen:
                result = future.result()
                # get_top_moves now returns (results, pv_line)
                if isinstance(result, tuple):
                    top_moves, pv_line = result
                else:
                    top_moves = result   # fallback for book moves if structure differs
                    pv_line   = []

        # ── Start new background search ───────────────────────────────
        if (needs_eval and not is_calculating
                and not board.is_game_over() and not edit_mode):
            if board.is_valid():
                last_eval_fen  = board.fen()
                future         = executor.submit(get_top_moves, board.copy(), gui.DEPTH, 3)
                is_calculating = True
                needs_eval     = False
            else:
                top_moves  = []
                pv_line    = []
                needs_eval = False

        # ── Draw ──────────────────────────────────────────────────────
        gui.draw_board(render_surface, board, last_move, board_flipped)
        gui.draw_coordinates(render_surface, fonts['coord'], board_flipped)

        if not edit_mode:
            gui.draw_highlight(render_surface, selected_square, board_flipped)
            gui.draw_check(render_surface, board, board_flipped)
            gui.draw_legal_moves(render_surface, board, selected_square, board_flipped)

        gui.draw_pieces(render_surface, board, board_flipped)

        flip_button_rect, edit_button_rect, palette_rects = gui.draw_sidebar(
            render_surface, fonts, top_moves, is_calculating, board,
            edit_mode, selected_edit_piece,
            pv_line=pv_line)   # ← pass PV line to sidebar

        screen.fill((0, 0, 0))
        scaled = pygame.transform.smoothscale(
            render_surface,
            (int(VIRTUAL_WIDTH * scale), int(VIRTUAL_HEIGHT * scale))
        )
        screen.blit(scaled, (offset_x, offset_y))

        pygame.display.flip()
        clock.tick(30)

if __name__ == "__main__":
    main()