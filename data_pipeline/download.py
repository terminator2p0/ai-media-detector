import os
from datasets import load_dataset

def main():
    # Define where the raw data will live
    output_dir = "data/raw/cifake"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Downloading CIFAKE dataset from HuggingFace...")
    # The HuggingFace identifier for the dataset 
    dataset = load_dataset("dragonintelligence/CIFAKE-image-dataset") 
    
    # Save the dataset to your local disk
    dataset.save_to_disk(output_dir)
    print(f"Dataset successfully downloaded and saved to {output_dir}")

if __name__ == "__main__":
    main()