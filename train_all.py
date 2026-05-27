"""Train all three detection models from acquired datasets.

Usage:
  python train_all.py                  # train all modalities
  python train_all.py --only image     # just the visual model
  python train_all.py --only audio
  python train_all.py --only text
  python train_all.py --epochs 10      # override epoch count

Expects data under data/training/{image,audio,text}/ — run
  python data_pipeline/acquire_datasets.py
first if those directories are empty.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import db

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

TRAINING_ROOT = "data/training"
CHECKPOINT_DIR = "models/checkpoints"


# ---------------------------------------------------------------------------
# Image training
# ---------------------------------------------------------------------------

class ImageFolderDataset(Dataset):
    """Reads data/training/image/{real,fake}/*.jpg"""

    LABEL_MAP = {"real": 0.0, "fake": 1.0}

    def __init__(self, root: str, transform):
        self.transform = transform
        self.samples: list[tuple[str, float]] = []
        for label_name, target in self.LABEL_MAP.items():
            d = os.path.join(root, label_name)
            if not os.path.isdir(d):
                continue
            for fname in os.listdir(d):
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                    self.samples.append((os.path.join(d, fname), target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from PIL import Image
        path, target = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), torch.tensor([target], dtype=torch.float32)


def train_image(epochs: int = 10, lr: float = 1e-4, batch_size: int = 32):
    from torchvision import transforms
    from models.model import AIMediaDetector

    root = os.path.join(TRAINING_ROOT, "image")
    if not os.path.isdir(root):
        print("No image training data found. Run acquire_datasets.py first.")
        return

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset = ImageFolderDataset(root, transform)
    if len(dataset) == 0:
        print("Image dataset is empty.")
        return

    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"Image dataset: {len(dataset)} samples (train={train_size}, val={val_size})")

    model = AIMediaDetector(pretrained=True).to(DEVICE)

    # Freeze backbone initially; train classifier head only
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.backbone.classifier.parameters():
        p.requires_grad = True

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    unfreeze_epoch = max(epochs // 3, 1)

    for epoch in range(epochs):
        if epoch == unfreeze_epoch:
            print(f"  Unfreezing backbone at epoch {epoch + 1}")
            for p in model.backbone.parameters():
                p.requires_grad = True
            optimizer = optim.AdamW(model.parameters(), lr=lr / 10)

        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for imgs, labels in tqdm(train_loader, desc=f"Image Epoch {epoch + 1}/{epochs}"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            preds = (torch.sigmoid(out) >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                out = model(imgs)
                preds = (torch.sigmoid(out) >= 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        train_acc = correct / max(total, 1) * 100
        val_acc = val_correct / max(val_total, 1) * 100
        print(f"  loss={running_loss / len(train_loader):.4f}  train_acc={train_acc:.1f}%  val_acc={val_acc:.1f}%")

    _save_checkpoint(model, "efficientnet_b4_video_final.pth", "image")
    print("Image training complete.\n")


# ---------------------------------------------------------------------------
# Audio training
# ---------------------------------------------------------------------------

class AudioFolderDataset(Dataset):
    """Reads data/training/audio/{real,fake}/*.wav"""

    def __init__(self, root: str, extractor, max_length_sec: int = 3):
        self.extractor = extractor
        self.max_len = 16000 * max_length_sec
        self.samples: list[tuple[str, int]] = []
        for label_name, target in [("real", 0), ("fake", 1)]:
            d = os.path.join(root, label_name)
            if not os.path.isdir(d):
                continue
            for fname in os.listdir(d):
                if fname.endswith(".wav"):
                    self.samples.append((os.path.join(d, fname), target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import librosa
        path, label = self.samples[idx]
        speech, _ = librosa.load(path, sr=16000)
        inputs = self.extractor(
            speech, sampling_rate=16000, max_length=self.max_len,
            truncation=True, padding="max_length", return_tensors="pt",
        )
        return {
            "input_values": inputs.input_values.squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def train_audio(epochs: int = 5, lr: float = 3e-5, batch_size: int = 4):
    from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

    root = os.path.join(TRAINING_ROOT, "audio")
    if not os.path.isdir(root):
        print("No audio training data found. Run acquire_datasets.py first.")
        return

    extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    dataset = AudioFolderDataset(root, extractor)
    if len(dataset) == 0:
        print("Audio dataset is empty.")
        return

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    print(f"Audio dataset: {len(dataset)} samples (train={train_size}, val={val_size})")

    model = AutoModelForAudioClassification.from_pretrained("facebook/wav2vec2-base", num_labels=2)
    model.freeze_feature_encoder()
    model.to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Audio Epoch {epoch + 1}/{epochs}"):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss += loss.item()

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                out = model(**batch)
                preds = torch.argmax(out.logits, dim=-1)
                val_correct += (preds == batch["labels"]).sum().item()
                val_total += batch["labels"].size(0)

        val_acc = val_correct / max(val_total, 1) * 100
        print(f"  loss={running_loss / len(train_loader):.4f}  val_acc={val_acc:.1f}%")

    _save_checkpoint(model, "wav2vec2_audio_final.pth", "audio")
    print("Audio training complete.\n")


# ---------------------------------------------------------------------------
# Text training
# ---------------------------------------------------------------------------

class TextDataset(Dataset):
    """Reads data/training/text/hc3_combined.jsonl"""

    def __init__(self, path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples: list[tuple[str, int]] = []
        with open(path) as fp:
            for line in fp:
                row = json.loads(line)
                label = 1 if row["label"] == "ai" else 0
                text = row["text"].strip()
                if text:
                    self.samples.append((text, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        enc = self.tokenizer(text, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def train_text(epochs: int = 3, lr: float = 2e-5, batch_size: int = 8):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    jsonl = os.path.join(TRAINING_ROOT, "text", "hc3_combined.jsonl")
    if not os.path.exists(jsonl):
        print("No text training data found. Run acquire_datasets.py first.")
        return

    model_id = "Oxidane/tmr-ai-text-detector"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dataset = TextDataset(jsonl, tokenizer)
    if len(dataset) == 0:
        print("Text dataset is empty.")
        return

    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    print(f"Text dataset: {len(dataset)} samples (train={train_size}, val={val_size})")

    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)
    model.to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Text Epoch {epoch + 1}/{epochs}"):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss += loss.item()

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                out = model(**batch)
                preds = torch.argmax(out.logits, dim=-1)
                val_correct += (preds == batch["labels"]).sum().item()
                val_total += batch["labels"].size(0)

        val_acc = val_correct / max(val_total, 1) * 100
        print(f"  loss={running_loss / len(train_loader):.4f}  val_acc={val_acc:.1f}%")

    save_dir = os.path.join(CHECKPOINT_DIR, "tmr_text_finetuned")
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Text model saved to {save_dir}\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(model, filename: str, modality: str):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    backup_dir = os.path.join(CHECKPOINT_DIR, "backups")
    os.makedirs(backup_dir, exist_ok=True)

    dest = os.path.join(CHECKPOINT_DIR, filename)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if os.path.exists(dest):
        shutil.copy(dest, os.path.join(backup_dir, f"{modality}_pre_{ts}.pth"))

    torch.save(model.state_dict(), dest)
    print(f"  Checkpoint saved: {dest}")


MODALITY_MAP = {
    "image": train_image,
    "audio": train_audio,
    "text": train_text,
}


def main():
    parser = argparse.ArgumentParser(description="Train all detection models")
    parser.add_argument("--only", choices=list(MODALITY_MAP.keys()))
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    args = parser.parse_args()

    db.init_db()
    print(f"Training on device: {DEVICE}\n")

    targets = [args.only] if args.only else list(MODALITY_MAP.keys())
    for mod in targets:
        print(f"\n{'='*60}")
        print(f"  Training: {mod.upper()}")
        print(f"{'='*60}")
        kwargs = {}
        if args.epochs:
            kwargs["epochs"] = args.epochs
        MODALITY_MAP[mod](**kwargs)

    print("\n--- All training complete. ---")


if __name__ == "__main__":
    main()
