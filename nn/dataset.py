import chess.pgn
import torch
from torch.utils.data import Dataset
import numpy as np
import os
import zstandard as zstd
import io

from encoder import board_to_tensor

MAX_GAMES = 1000

class ChessDataset(Dataset):
    """
    PyTorch Dataset class to load our pre-processed chess positions.
    """
    def __init__(self, data_file):
        print(f"Loading data from {data_file}...")
        data = torch.load(data_file)
        self.inputs = data['inputs']
        self.labels = data['labels']
        print(f"Loaded {len(self.inputs)} positions.")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

def process_zst_to_tensors(zst_path, output_path, max_games=MAX_GAMES):
    """
    Reads a ZST compressed PGN file on the fly, plays through the games,
    converts positions to tensors, and saves them to a PyTorch .pt file.
    """
    if not os.path.exists(zst_path):
        print(f"Error: ZST file not found at {zst_path}")
        return

    print(f"Processing compressed file {zst_path}...")
    inputs = []
    labels = []

    # Open the compressed file and stream it directly
    with open(zst_path, "rb") as compressed_file:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(compressed_file) as reader:
            text_stream = io.TextIOWrapper(reader, encoding='utf-8')

            games_processed = 0
            while games_processed < max_games:
                game = chess.pgn.read_game(text_stream)
                if game is None:
                    break # End of file

                result = game.headers.get("Result")
                if result == "1-0":
                    label = 1.0
                elif result == "0-1":
                    label = -1.0
                elif result == "1/2-1/2":
                    label = 0.0
                else:
                    continue

                board = game.board()
                for move in game.mainline_moves():
                    board.push(move)
                    tensor = board_to_tensor(board)
                    inputs.append(tensor)
                    labels.append(torch.tensor([label], dtype=torch.float32))

                games_processed += 1
                if games_processed % 100 == 0:
                    print(f"Processed {games_processed} games...")

    print(f"Finished parsing. Total positions extracted: {len(inputs)}")
    print("Stacking tensors into batches... this might take a moment.")

    X = torch.stack(inputs)
    y = torch.stack(labels)

    torch.save({'inputs': X, 'labels': y}, output_path)
    print(f"Successfully saved tensor data to {output_path}")

if __name__ == "__main__":
    zst_file = "games.pgn.zst"
    output_file = "dataset.pt"


    process_zst_to_tensors(zst_file, output_file, max_games=MAX_GAMES)