import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import os
import shutil
from datetime import datetime
from models.inference_orchestrator import MediaForensicsOrchestrator

def train_on_feedback():
    # 1. Device Setup (Mac MPS)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"--- 🧠 Starting Model Refinement on: {device} ---")

    # 2. Initialize Orchestrator and Extract Model
    # This ensures we are loading the exact architecture and current weights
    orchestrator = MediaForensicsOrchestrator()
    model = orchestrator.visual_model
    model.to(device)
    model.train() 

    # 3. Data Loading Logic
    # The folder structure is data/feedback_loop/real and data/feedback_loop/fake
    feedback_path = "data/feedback_loop"
    
    # Check if we have enough data (at least one file in each class is ideal)
    real_count = len(os.listdir(os.path.join(feedback_path, "real")))
    fake_count = len(os.listdir(os.path.join(feedback_path, "fake")))
    
    if (real_count + fake_count) < 1:
        print("❌ Error: No feedback data found. Go to the app and mark some mistakes first!")
        return

    print(f"--- Found {real_count} Real and {fake_count} Fake feedback samples ---")

    # Use the orchestrator's existing transform to stay consistent
    transform = orchestrator.img_transform
    dataset = datasets.ImageFolder(root=feedback_path, transform=transform)
    
    # Note: dataset.class_to_idx will map 'fake' to 0 and 'real' to 1 usually.
    # We need to ensure 'fake' is 1 and 'real' is 0 for our sigmoid logic.
    # In ImageFolder, labels are assigned alphabetically: fake=0, real=1.
    # Our model: AI/Deepfake > 0.5 (Label 1). So we swap them if needed.
    
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    # 4. Optimizer & Loss
    # Using a micro-learning rate to prevent 'Catastrophic Forgetting'
    optimizer = optim.Adam(model.parameters(), lr=1e-6)
    criterion = nn.BCEWithLogitsLoss()

    # 5. Fine-Tuning Loop
    epochs = 5
    print(f"--- Refinement in progress for {epochs} epochs ---")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for inputs, labels in loader:
            inputs = inputs.to(device)
            # Adjusting labels: ImageFolder gives fake=0, real=1. 
            # Our model architecture expects fake=1, real=0.
            target_labels = (1 - labels).float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, target_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss = {epoch_loss/len(loader):.5f}")

    # 6. Backup and Save
    original_path = "models/checkpoints/efficientnet_b4_video_final.pth"
    backup_path = f"models/checkpoints/backups/model_bg_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
    
    os.makedirs("models/checkpoints/backups", exist_ok=True)
    if os.path.exists(original_path):
        shutil.copy(original_path, backup_path)
        print(f"--- Backup created at {backup_path} ---")

    torch.save(model.state_dict(), original_path)
    print(f"--- ✅ Success: Smarter model saved to {original_path} ---")

if __name__ == "__main__":
    train_on_feedback()