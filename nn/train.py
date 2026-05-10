import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import ChessDataset
from model import ChessNet

def train_model():
    # Hyperparameters
    batch_size = 512
    learning_rate = 0.0005
    epochs = 10
    dataset_path = "dataset.pt"
    model_save_path = "chess_model.pth"

    # Check for GPU acceleration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load pre-processed tensor data
    print("Loading dataset...")
    dataset = ChessDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, loss function, and optimizer
    model = ChessNet().to(device)
    criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1, factor=0.5, verbose=True)

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(dataloader):
            # Move data to GPU if available
            inputs, labels = inputs.to(device), labels.to(device)

            # Clear previous gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize weights
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Print progress every 100 batches
            if (i + 1) % 100 == 0:
                avg_loss = running_loss / 100
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataloader)}], Loss: {avg_loss:.4f}")
                running_loss = 0.0

    # Save the trained model weights
    torch.save(model.state_dict(), model_save_path)
    print(f"Training complete. Model saved to {model_save_path}")

if __name__ == "__main__":
    train_model()