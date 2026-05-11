import chess.pgn
import torch
from torch.utils.data import Dataset
import numpy as np
import os
import zstandard as zstd
import io
import random
import math

from encoder import board_to_tensor

# Increased for overnight training
MAX_GAMES = 100000

class ChessDataset(Dataset):
    """
    PyTorch Dataset class to load pre-processed chess positions.
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

def process_zst_to_tensors(zst_path, output_path, max_games=10000, keep_prob=0.05):
    """
    Reads a ZST compressed PGN and extracts engine evaluations.
    """
    if not os.path.exists(zst_path):
        print(f"Error: ZST file not found at {zst_path}")
        return

    print(f"Processing compressed file {zst_path}...")
    print(f"Target: {max_games} games. Keep probability: {keep_prob*100}%")

    inputs = []
    labels = []

    with open(zst_path, "rb") as compressed_file:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(compressed_file) as reader:
            text_stream = io.TextIOWrapper(reader, encoding='utf-8')

            games_processed = 0
            games_scanned = 0

            while games_processed < max_games:
                games_scanned += 1

                # Randomly skip games to diversify dataset
                if random.random() > keep_prob:
                    if not chess.pgn.skip_game(text_stream):
                        break
                    continue

                game = chess.pgn.read_game(text_stream)
                if game is None:
                    break

                board = game.board()

                # Iterate through moves and extract [%eval] tags
                for node in game.mainline():
                    board.push(node.move)

                    eval_score = node.eval()
                    if eval_score is not None:
                        # Normalize score (mate = 100 pawns)
                        score = eval_score.white().score(mate_score=10000) / 100.0

                        # Compress to [-1, 1] range using tanh
                        label = math.tanh(score / 4.0)

                        tensor = board_to_tensor(board)
                        inputs.append(tensor)
                        labels.append(torch.tensor([label], dtype=torch.float32))

                games_processed += 1
                if games_processed % 500 == 0:
                    print(f"Processed {games_processed}/{max_games} games...")

    print(f"Stacking {len(inputs)} positions...")
    X = torch.stack(inputs)
    y = torch.stack(labels)

    torch.save({'inputs': X, 'labels': y}, output_path)
    print(f"Successfully saved tensor data to {output_path}")

if __name__ == "__main__":
    zst_file = "games.pgn.zst"
    output_file = "dataset.pt"

    process_zst_to_tensors(zst_file, output_file, max_games=MAX_GAMES, keep_prob=0.2)