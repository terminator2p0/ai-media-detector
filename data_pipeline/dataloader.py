import os
import torch
from torch.utils.data import Dataset, DataLoader

class CIFakeDataset(Dataset):
    def __init__(self, data_dir):
        """
        Args:
            data_dir (string): Directory with all the tensor files.
        """
        self.data_dir = data_dir
        # Get a list of all .pt files in the directory
        self.file_names = [f for f in os.listdir(data_dir) if f.endswith('.pt')]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_path = os.path.join(self.data_dir, self.file_names[idx])
        # Load the dictionary containing 'tensor' and 'label'
        data = torch.load(file_path, weights_only=True) 
        
        image_tensor = data['tensor']
        # Convert label to float32 for Binary Cross Entropy Loss
        label = torch.tensor(data['label'], dtype=torch.float32) 

        return image_tensor, label

def get_dataloaders(train_dir, test_dir, batch_size=32):
    """Creates and returns PyTorch DataLoaders for train and test sets."""
    train_dataset = CIFakeDataset(train_dir)
    test_dataset = CIFakeDataset(test_dir)

    # num_workers allows for multi-process data loading, speeding up the pipeline
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader

# Quick test block to ensure it initializes correctly
if __name__ == "__main__":
    train_path = "data/processed/cifake_tensors/train"
    test_path = "data/processed/cifake_tensors/test"
    
    if os.path.exists(train_path) and os.path.exists(test_path):
        # Using batch size 32 from our training_config.yaml
        train_loader, test_loader = get_dataloaders(train_path, test_path, batch_size=32)
        
        # Fetch a single batch
        images, labels = next(iter(train_loader))
        print("Batch loaded successfully!")
        print(f"Images batch shape: {images.shape} (Expected: torch.Size([32, 3, 224, 224]))")
        print(f"Labels batch shape: {labels.shape} (Expected: torch.Size([32]))")
    else:
        print("Processed data directories not found. Check your paths!")