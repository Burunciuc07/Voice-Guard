import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import VoiceDataset
from model import VoiceGuardCNN
from tqdm import tqdm

def train():
    data_dir = "dataset" # Assumes REAL and FAKE folders are in the dataset directory
    dataset = VoiceDataset(data_dir=data_dir)
    
    if len(dataset) == 0:
        print(f"No samples found in {data_dir}/REAL or {data_dir}/FAKE")
        print("Please ensure you have placed .wav files in these folders before training.")
        return

    # Train/Val split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    if val_size == 0 or train_size == 0:
        print("Not enough dataset samples for splitting. Falling back to training on all data...")
        train_dataset = dataset
        val_dataset = dataset
    else:
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = VoiceGuardCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    epochs = 15
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for i, (inputs, labels) in enumerate(train_pbar):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            train_pbar.set_postfix(loss=f"{running_loss/(i+1):.4f}")
            
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        val_acc = 100 * correct / total if total > 0 else 0
        print(f"Epoch {epoch+1} Summary - Train Loss: {running_loss/len(train_loader):.4f} | Val Loss: {val_loss/max(1, len(val_loader)):.4f} | Val Acc: {val_acc:.2f}%\n")

    torch.save(model.state_dict(), "voiceguard.pth")
    print("Model saved to voiceguard.pth")

if __name__ == "__main__":
    if not os.path.exists("dataset/REAL"):
        os.makedirs("dataset/REAL", exist_ok=True)
    if not os.path.exists("dataset/FAKE"):
        os.makedirs("dataset/FAKE", exist_ok=True)
    train()
