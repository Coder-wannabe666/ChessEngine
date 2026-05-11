import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import ChessDataset
from model import ChessNet

def train_model():
    # Hyperparameters optimized for RTX 3050 (4GB VRAM)
    batch_size = 512
    learning_rate = 0.0001
    epochs = 30
    dataset_path = "dataset.pt"
    model_save_path = "chess_model.pth"

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load pre-processed dataset
    print("Loading dataset...")
    try:
        dataset = ChessDataset(dataset_path)
    except FileNotFoundError:
        print(f"Error: {dataset_path} not found. Run dataset.py first.")
        return

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Model, Loss, and Optimizer
    model = ChessNet().to(device)
    criterion = nn.HuberLoss() # Robust to outliers
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Scheduler to reduce LR when loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 100 == 0:
                avg_loss = running_loss / 100
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataloader)}], Loss: {avg_loss:.4f}")
                running_loss = 0.0

        # Step the scheduler after each epoch
        epoch_loss = running_loss / len(dataloader)
        scheduler.step(epoch_loss)

        # Save checkpoint after each epoch
        torch.save(model.state_dict(), f"chess_model_epoch_{epoch+1}.pth")

    # Final save
    torch.save(model.state_dict(), model_save_path)
    print(f"Training complete. Model saved to {model_save_path}")

if __name__ == "__main__":
    train_model()