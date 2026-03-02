import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm

# Import the model and dataloader we just built
from model import AIMediaDetector
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_pipeline.dataloader import get_dataloaders

def load_config():
    with open("configs/training_configs.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    
    # Initialize Weights & Biases for experiment tracking
    wandb.init(
        project="ai-media-detector",
        config=config,
        name=f"efficientnet-b4-cifake"
    )
    
    # Dynamic device selection (CUDA for GCP/Nvidia, MPS for Mac, or CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Training on device: {device}")

    # Load Data
    train_path = "data/processed/cifake_tensors/train"
    test_path = "data/processed/cifake_tensors/test"
    train_loader, val_loader = get_dataloaders(train_path, test_path, batch_size=config['batch_size'])

    # Initialize Model
    model = AIMediaDetector(pretrained=True).to(device)
    
    # Freeze the base layers initially
    for param in model.backbone.parameters():
        param.requires_grad = False
    # Ensure the classification head remains unfrozen
    for param in model.backbone.classifier.parameters():
        param.requires_grad = True

    # Setup Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate_frozen'], 
        weight_decay=config['weight_decay']
    )
    
    scaler = torch.amp.GradScaler(device.type) if device.type == 'cuda' else None

    epochs = config['epochs']
    
    print("Starting Training Loop...")
    for epoch in range(epochs):
        # Unfreeze backbone after 5 epochs for fine-tuning
        if epoch == 5:
            print("Unfreezing base layers for full fine-tuning...")
            for param in model.backbone.parameters():
                param.requires_grad = True
            # Update learning rate for unfrozen training
            for param_group in optimizer.param_groups:
                param_group['lr'] = config['learning_rate_unfrozen']
        
        # --- TRAINING PASS ---
        model.train()
        train_loss, train_correct, total = 0.0, 0, 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            
            # Mixed precision training if on CUDA
            if scaler:
                with torch.amp.autocast(device_type=device.type):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # Calculate metrics
            train_loss += loss.item()
            predictions = (torch.sigmoid(outputs) >= 0.5).float()
            train_correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            loop.set_postfix(loss=loss.item(), acc=train_correct/total)

        # Log training metrics to wandb
        wandb.log({"train_loss": train_loss / len(train_loader), "train_accuracy": train_correct / total, "epoch": epoch + 1})

    # Save the final model checkpoint
    os.makedirs("models/checkpoints", exist_ok=True)
    checkpoint_path = f"models/checkpoints/efficientnet_b4_final.pth"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Training complete. Model saved to {checkpoint_path}")
    wandb.finish()

if __name__ == "__main__":
    main()