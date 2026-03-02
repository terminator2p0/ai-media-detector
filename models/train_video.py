import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.model import AIMediaDetector
from data_pipeline.video_dataloader import get_video_dataloaders

def load_config():
    with open("configs/training_configs.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    
    # Initialize wandb for this new video training run
    wandb.init(
        project="ai-media-detector",
        config=config,
        name="efficientnet-b4-video-finetune"
    )
    
    # We can safely use MPS on Mac for standard PyTorch model training
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Load Video Data
    train_csv = "data/train_manifest.csv"
    val_csv = "data/val_manifest.csv"
    root_dir = "data/processed/cropped_faces"
    train_loader, val_loader = get_video_dataloaders(train_csv, val_csv, root_dir, batch_size=config['batch_size'])

    # Initialize Model and load the CIFAKE weights
    print("Loading base model previously trained on CIFAKE...")
    model = AIMediaDetector(pretrained=False).to(device)
    base_checkpoint_path = "models/checkpoints/efficientnet_b4_final.pth"
    
    if not os.path.exists(base_checkpoint_path):
        print("Error: CIFAKE base model not found. Please ensure efficientnet_b4_final.pth exists.")
        return
        
    model.load_state_dict(torch.load(base_checkpoint_path, map_location=device, weights_only=True))
    
    # Since the model is already highly accurate, we unfreeze all layers 
    # and use a very small learning rate for gentle fine-tuning
    for param in model.parameters():
        param.requires_grad = True

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate_unfrozen'], weight_decay=config['weight_decay'])
    
    scaler = torch.amp.GradScaler(device.type) if device.type == 'cuda' else None

    # We only need 3-5 epochs since the base model is already very smart
    epochs = 5 
    
    print("Starting Video Fine-Tuning Loop...")
    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, total = 0.0, 0, 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            
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

            train_loss += loss.item()
            predictions = (torch.sigmoid(outputs) >= 0.5).float()
            train_correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            loop.set_postfix(loss=loss.item(), acc=train_correct/total)

        # Log metrics to wandb
        wandb.log({"video_train_loss": train_loss / len(train_loader), "video_train_acc": train_correct / total, "epoch": epoch + 1})

    # Save the new video-specific model checkpoint
    os.makedirs("models/checkpoints", exist_ok=True)
    save_path = "models/checkpoints/efficientnet_b4_video_final.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Video training complete. Model saved to {save_path}")
    wandb.finish()

if __name__ == "__main__":
    main()