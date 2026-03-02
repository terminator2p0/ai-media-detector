import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class DeepfakeDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv manifest file.
            root_dir (string): Directory with all the cropped face images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        # Load the manifest into a pandas DataFrame
        self.manifest = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Construct the full image path from the relative path in the CSV
        img_path = os.path.join(self.root_dir, self.manifest.iloc[idx]['image_path'])
        
        # Load image and ensure it's in RGB format
        image = Image.open(img_path).convert('RGB')
        label = float(self.manifest.iloc[idx]['label'])

        # Apply resizing and normalization dynamically
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)

def get_video_dataloaders(train_csv, val_csv, root_dir, batch_size=32):
    """Creates and returns PyTorch DataLoaders for the video frames."""
    
    # ImageNet standards required by EfficientNet-B4
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = DeepfakeDataset(csv_file=train_csv, root_dir=root_dir, transform=preprocess)
    val_dataset = DeepfakeDataset(csv_file=val_csv, root_dir=root_dir, transform=preprocess)

    # num_workers speeds up the ETL process from disk to GPU/CPU memory
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader

# Quick test block
if __name__ == "__main__":
    train_csv = "data/train_manifest.csv"
    val_csv = "data/val_manifest.csv"
    root_dir = "data/processed/cropped_faces"
    
    if os.path.exists(train_csv):
        print("Initializing dataloaders...")
        train_loader, val_loader = get_video_dataloaders(train_csv, val_csv, root_dir)
        
        images, labels = next(iter(train_loader))
        print("Batch loaded successfully!")
        print(f"Images batch shape: {images.shape} (Expected: torch.Size([32, 3, 224, 224]))")
        print(f"Labels batch shape: {labels.shape} (Expected: torch.Size([32]))")
    else:
        print("Manifest files not found. Check your paths!")