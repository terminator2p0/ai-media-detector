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
    # 1. Device Setup (Mac M-series Optimization)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"--- 🧠 Starting Model Refinement on: {device} ---")

    # 2. Initialize Orchestrator and Extract Model
    # This pulls your current 'EfficientNet' and its existing weights
    orchestrator = MediaForensicsOrchestrator()
    model = orchestrator.visual_model
    model.to(device)
    model.train() 

    # 3. Data Loading Logic
    feedback_path = "data/feedback_loop"
    
    # Ensure folders exist
    os.makedirs(os.path.join(feedback_path, "real"), exist_ok=True)
    os.makedirs(os.path.join(feedback_path, "fake"), exist_ok=True)
    
    real_count = len([f for f in os.listdir(os.path.join(feedback_path, "real")) if not f.endswith('.json')])
    fake_count = len([f for f in os.listdir(os.path.join(feedback_path, "fake")) if not f.endswith('.json')])
    
    if (real_count + fake_count) < 1:
        print("❌ Error: No feedback data found. Mark some mistakes in the app first!")
        return

    print(f"--- Found {real_count} Real and {fake_count} Fake feedback samples ---")

    # Use the orchestrator's existing transform for consistency
    transform = orchestrator.img_transform
    dataset = datasets.ImageFolder(root=feedback_path, transform=transform)
    
    # Batch size of 2 or 4 is best for very small "correction" datasets
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    # 4. Optimizer & Loss
    # Ultra-low learning rate (1e-6) to fine-tune without "breaking" the model
    optimizer = optim.Adam(model.parameters(), lr=1e-6)
    criterion = nn.BCEWithLogitsLoss()

    # 5. Fine-Tuning Loop
    epochs = 5
    print(f"--- Refinement in progress for {epochs} epochs ---")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for inputs, labels in loader:
            inputs = inputs.to(device)
            # ImageFolder maps alphabetically: fake=0, real=1.
            # We flip these to match our model: deepfake=1, authentic=0.
            target_labels = (1 - labels).float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, target_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss = {epoch_loss/len(loader):.5f}")

    # 6. Backup Current Model Before Overwriting
    original_path = "models/checkpoints/efficientnet_b4_video_final.pth"
    backup_dir = "models/checkpoints/backups"
    os.makedirs(backup_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = os.path.join(backup_dir, f"model_pre_refinement_{timestamp}.pth")
    
    if os.path.exists(original_path):
        shutil.copy(original_path, backup_path)
        print(f"--- 🛡️ Backup created: {backup_path} ---")

    # 7. Save the Refined Model
    torch.save(model.state_dict(), original_path)
    print(f"--- ✅ Success: Smarter model saved to {original_path} ---")

    # 8. Archive the Feedback Data
    # This prevents training on the same images repeatedly in the future
    archive_root = "data/archive"
    session_archive = os.path.join(archive_root, timestamp)
    os.makedirs(session_archive, exist_ok=True)
    
    for category in ['real', 'fake']:
        cat_path = os.path.join(feedback_path, category)
        for f in os.listdir(cat_path):
            src = os.path.join(cat_path, f)
            dst = os.path.join(session_archive, f)
            shutil.move(src, dst)
    
    print(f"--- 📦 Data archived to {session_archive}. Ready for the next loop! ---")

if __name__ == "__main__":
    train_on_feedback()