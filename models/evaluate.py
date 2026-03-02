import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

# Import your custom modules
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.model import AIMediaDetector
from data_pipeline.dataloader import get_dataloaders

def main():
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    # 2. Load the Test Data
    train_path = "data/processed/cifake_tensors/train"
    test_path = "data/processed/cifake_tensors/test"
    # We only need the test_loader here
    _, test_loader = get_dataloaders(train_path, test_path, batch_size=32)

    # 3. Load the Trained Model
    print("Loading model weights...")
    model = AIMediaDetector(pretrained=False) # No need to download base weights again
    checkpoint_path = "models/checkpoints/efficientnet_b4_final.pth"
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Model checkpoint not found at {checkpoint_path}")
        return
        
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval() # Set to evaluation mode (turns off dropout, batchnorm updates)

    # 4. Run Inference
    all_labels = []
    all_preds = []
    all_probs = []

    print("Running inference on the test set. This might take a few minutes...")
    with torch.no_grad(): # Disables gradient calculation to save memory/speed
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = torch.sigmoid(outputs).squeeze()
            
            # Handle edge case where batch size is 1 (squeeze removes the dimension)
            if probs.dim() == 0:
                probs = probs.unsqueeze(0)
                
            preds = (probs >= 0.5).float()
            
            # Store results for scikit-learn to process
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # 5. Calculate Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc_roc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)
    
    # Calculate False Positive Rate (FPR)
    # cm structure: [[True Negative, False Positive], [False Negative, True Positive]]
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn)

    print("\n" + "="*30)
    print("🎯 FINAL EVALUATION METRICS")
    print("="*30)
    print(f"Accuracy:            {accuracy * 100:.2f}%")
    print(f"F1 Score:            {f1:.4f}")
    print(f"AUC-ROC:             {auc_roc:.4f}")
    print(f"False Positive Rate: {fpr * 100:.2f}%")
    print("="*30 + "\n")

    # 6. Plot and Save Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'AI Generated'], 
                yticklabels=['Real', 'AI Generated'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix: Real vs. AI-Generated')
    
    os.makedirs("models/results", exist_ok=True)
    cm_path = "models/results/confusion_matrix.png"
    plt.savefig(cm_path, bbox_inches='tight')
    print(f"Confusion matrix image saved to {cm_path}")

if __name__ == "__main__":
    main()