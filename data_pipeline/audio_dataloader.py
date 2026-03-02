import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, Audio
from transformers import AutoFeatureExtractor

class AIFakeAudioDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, extractor):
        self.dataset = dataset
        self.extractor = extractor

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Extract the raw audio waveform array
        audio_array = item["audio"]["array"]
        
        # Wav2Vec2 processes 1D arrays (waveforms). We truncate to 3 seconds to save memory.
        inputs = self.extractor(
            audio_array, 
            sampling_rate=16000, 
            max_length=16000 * 3, 
            truncation=True, 
            padding="max_length", 
            return_tensors="pt"
        )
        
        # Map labels (0 = Human, 1 = AI)
        label = item["label"]
        
        return {
            "input_values": inputs.input_values.squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }

    def __len__(self):
        return len(self.dataset)

def get_audio_dataloaders(batch_size=4):
    print("Downloading Audio Deepfake dataset from Hugging Face...")
    # Using 2,000 samples for a fast MVP training run
    dataset = load_dataset("garystafford/deepfake-audio-detection", split="train[:1000]")
    
    # CRITICAL: Wav2Vec2 MUST have 16kHz audio. This forces on-the-fly resampling.
    print("Resampling dataset to 16kHz...")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    # Load the base feature extractor
    extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    
    # Split into Train and Validation (80/20)
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    
    train_data = AIFakeAudioDataset(dataset["train"], extractor)
    val_data = AIFakeAudioDataset(dataset["test"], extractor)
    
    # num_workers=0 is safer on Mac for audio processing
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    
    return train_loader, val_loader, extractor

if __name__ == "__main__":
    train_loader, _, _ = get_audio_dataloaders()
    batch = next(iter(train_loader))
    print("\nAudio Batch loaded successfully!")
    print(f"Input Values Shape: {batch['input_values'].shape}")
    print(f"Labels Shape: {batch['labels'].shape}")