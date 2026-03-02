import os
import torch
from torch.optim import AdamW
from transformers import AutoModelForAudioClassification
from tqdm import tqdm

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_pipeline.audio_dataloader import get_audio_dataloaders

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Starting Audio Fine-Tuning on: {device}")

    # 1. Load the Data
    train_loader, val_loader, extractor = get_audio_dataloaders(batch_size=4)

    # 2. Initialize the Base Model
    print("Loading base Wav2Vec2 model (this will initialize a fresh classification head)...")
    model = AutoModelForAudioClassification.from_pretrained(
        "facebook/wav2vec2-base", 
        num_labels=2
    )
    
    # Data Engineering Hack: Freeze the heavy acoustic feature extractor 
    # so we only train the classification head. This saves massive amounts of Mac memory!
    model.freeze_feature_encoder() 
    model.to(device)

    # 3. Setup Optimizer
    optimizer = AdamW(model.parameters(), lr=3e-5)
    num_epochs = 3 

    # 4. Training Loop
    print("\nStarting Training...")
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch in loop:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs.loss
            total_train_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            loop.set_postfix(loss=loss.item())

        # 5. Validation Loop
        model.eval()
        total_eval_accuracy = 0
        
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            
            predictions = torch.argmax(outputs.logits, dim=-1)
            total_eval_accuracy += (predictions == batch['labels']).float().mean().item()
            
        print(f"\nEnd of Epoch {epoch+1} | Val Accuracy: {(total_eval_accuracy/len(val_loader))*100:.2f}%\n")

    # 6. Save Model
    os.makedirs("models/checkpoints", exist_ok=True)
    save_path = "models/checkpoints/wav2vec2_audio_final.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Training Complete! Audio model saved to {save_path}")

if __name__ == "__main__":
    main()