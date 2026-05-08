import chess
from search import get_best_move

def print_board(board):
    print("\n-----------------")
    print(board)
    print("-----------------\n")

def main():
    board = chess.Board()
    DEPTH = 3

    print("=== Python Chess Engine ===")
    print("You play White. Enter moves in UCI format (e.g., 'e2e4', 'g1f3').")

    while not board.is_game_over():
        print_board(board)

        if board.turn == chess.WHITE:
            user_move = input("Your move: ")
            try:
                move = chess.Move.from_uci(user_move)
                if move in board.legal_moves:
                    board.push(move)
                else:
                    print("Illegal move. Try again.")
            except ValueError:
                print("Error: Invalid format. Use UCI format (e.g., e2e4).")
        else:
            print(f"Engine computing at depth {DEPTH}...")
            engine_move = get_best_move(board, DEPTH)

            if engine_move:
                print(f"Engine plays: {engine_move.uci()}")
                board.push(engine_move)
            else:
                print("Error: Engine found no move.")
                break

    print_board(board)
    print("Game Over!")
    print(f"Result: {board.result()}")

if __name__ == "__main__":
    main()