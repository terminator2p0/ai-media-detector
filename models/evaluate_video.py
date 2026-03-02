import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.model import AIMediaDetector
from data_pipeline.video_dataloader import DeepfakeDataset

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Evaluating Video Model on device: {device}")

    # 1. Setup Test Dataloader
    test_csv = "data/test_manifest.csv"
    root_dir = "data/processed/cropped_faces"
    
    if not os.path.exists(test_csv):
        print(f"Error: {test_csv} not found. Did you run build_manifest.py?")
        return

    # ImageNet standards required by EfficientNet-B4
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = DeepfakeDataset(csv_file=test_csv, root_dir=root_dir, transform=preprocess)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    # 2. Load the Fine-Tuned Model
    print("Loading fine-tuned video model weights...")
    model = AIMediaDetector(pretrained=False)
    checkpoint_path = "models/checkpoints/efficientnet_b4_video_final.pth"
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval() # Turns off dropout for clean evaluation

    # 3. Run Inference
    all_labels = []
    all_preds = []
    all_probs = []

    print("Running inference on the test frames. This might take a minute...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = torch.sigmoid(outputs).squeeze()
            
            if probs.dim() == 0:
                probs = probs.unsqueeze(0)
                
            preds = (probs >= 0.5).float()
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # 4. Calculate Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc_roc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)
    
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn)

    print("\n" + "="*30)
    print("🎥 FINAL VIDEO EVALUATION METRICS")
    print("="*30)
    print(f"Accuracy:            {accuracy * 100:.2f}%")
    print(f"F1 Score:            {f1:.4f}")
    print(f"AUC-ROC:             {auc_roc:.4f}")
    print(f"False Positive Rate: {fpr * 100:.2f}%")
    print("="*30 + "\n")

    # 5. Plot and Save Confusion Matrix
    plt.figure(figsize=(8, 6))
    # We'll use Reds to visually distinguish it from the Blue image matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', 
                xticklabels=['Real', 'Deepfake'], 
                yticklabels=['Real', 'Deepfake'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix: Real vs. Deepfake Videos')
    
    os.makedirs("models/results", exist_ok=True)
    cm_path = "models/results/video_confusion_matrix.png"
    plt.savefig(cm_path, bbox_inches='tight')
    print(f"Confusion matrix image saved to {cm_path}")

if __name__ == "__main__":
    main()