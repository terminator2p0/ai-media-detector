import os
import torch
from datasets import load_from_disk
from torchvision import transforms
from tqdm import tqdm

def main():
    input_dir = "data/raw/cifake"
    output_dir = "data/processed/cifake_tensors"
    
    print("Loading raw dataset from disk...")
    dataset = load_from_disk(input_dir)

    # Define the transformation pipeline
    # Resizing to 224x224 and normalizing using ImageNet standards [cite: 19, 20]
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Process both 'train' and 'test' splits
    for split in dataset.keys():
        print(f"\nProcessing {split} split...")
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        
        # Wrap the loop in tqdm for a progress bar
        for i, item in enumerate(tqdm(dataset[split])):
            image = item['image']
            label = item['label']
            
            # Ensure image is in RGB format before processing
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            # Apply resizing and normalization
            tensor = preprocess(image)
            
            # Save the processed tensor and its label (0 for Real, 1 for AI) [cite: 22]
            torch.save({'tensor': tensor, 'label': label}, os.path.join(split_dir, f"{i}.pt"))

    print("\nPreprocessing complete. Tensors saved to data/processed/")

if __name__ == "__main__":
    main()