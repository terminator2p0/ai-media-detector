"""Fine-tune the visual model on user-confirmed feedback samples.

Source of truth: the SQLite feedback table (see db.py). For each feedback row whose
stored_media_path still exists on disk, we build a (path, label) pair where
deepfake=1 and authentic=0. We fall back to the legacy ImageFolder layout under
data/feedback_loop/ if the DB has no rows yet, so older datasets keep working.
"""

import os
import shutil
from datetime import datetime
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import db
from models.inference_orchestrator import MediaForensicsOrchestrator


LABEL_TO_TARGET = {"fake": 1.0, "real": 0.0}


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}


def _load_as_image(path: str) -> Image.Image:
    """Load an image file or extract the middle frame from a video."""
    ext = os.path.splitext(path)[1].lower()
    if ext in IMAGE_EXTS:
        return Image.open(path).convert("RGB")
    if ext in VIDEO_EXTS:
        import cv2
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(total // 2, 0))
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise ValueError(f"Could not extract frame from {path}")
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    raise ValueError(f"Unsupported file type: {ext}")


class FeedbackImageDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, float]], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, target = self.samples[idx]
        img = _load_as_image(path)
        return self.transform(img), torch.tensor([target], dtype=torch.float32)


def collect_samples_from_db() -> List[Tuple[str, float]]:
    samples = []
    for row in db.training_samples():
        label = row["true_label"]
        if label not in LABEL_TO_TARGET:
            continue
        samples.append((row["stored_media_path"], LABEL_TO_TARGET[label]))
    return samples


def collect_samples_from_filesystem(root: str = "data/feedback_loop") -> List[Tuple[str, float]]:
    samples = []
    for label, target in LABEL_TO_TARGET.items():
        d = os.path.join(root, label)
        if not os.path.isdir(d):
            continue
        for fname in os.listdir(d):
            if fname.endswith(".json"):
                continue
            full = os.path.join(d, fname)
            if not os.path.isfile(full):
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext not in IMAGE_EXTS and ext not in VIDEO_EXTS:
                continue
            samples.append((full, target))
    return samples


def collect_samples() -> List[Tuple[str, float]]:
    """Union DB + filesystem feedback, deduped by absolute path.

    The DB may hold rows whose media lives elsewhere (or on another machine, e.g.
    Streamlit Cloud) while local files may not yet be in the DB. We take both and
    keep only samples whose media actually exists on disk.
    """
    seen: set[str] = set()
    merged: List[Tuple[str, float]] = []
    for path, target in collect_samples_from_db() + collect_samples_from_filesystem():
        if not path or not os.path.exists(path):
            continue
        ap = os.path.abspath(path)
        if ap in seen:
            continue
        seen.add(ap)
        merged.append((path, target))
    return merged


def train_on_feedback():
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"--- 🧠 Starting Model Refinement on: {device} ---")

    db.init_db()

    orchestrator = MediaForensicsOrchestrator()
    model = orchestrator.visual_model
    model.to(device)
    model.train()

    samples = collect_samples()
    print(f"--- Loaded {len(samples)} trainable samples (DB + filesystem, deduped) ---")

    if not samples:
        print("❌ No feedback data found. Mark some scans in the app first!")
        return

    real_n = sum(1 for _, t in samples if t == 0.0)
    fake_n = sum(1 for _, t in samples if t == 1.0)
    if real_n == 0 or fake_n == 0:
        print(f"⚠️  Refusing to train: data is single-class (real={real_n}, fake={fake_n}).")
        print("    Training on one class biases the model. Add the missing class first.")
        return

    real_count = sum(1 for _, t in samples if t == 0.0)
    fake_count = sum(1 for _, t in samples if t == 1.0)
    print(f"--- Distribution: real={real_count}, fake={fake_count} ---")

    dataset = FeedbackImageDataset(samples, transform=orchestrator.img_transform)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=1e-6)
    criterion = nn.BCEWithLogitsLoss()

    epochs = 5
    print(f"--- Refinement in progress for {epochs} epochs ---")
    for epoch in range(epochs):
        epoch_loss = 0.0
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}: Loss = {epoch_loss / max(len(loader), 1):.5f}")

    original_path = "models/checkpoints/efficientnet_b4_video_final.pth"
    backup_dir = "models/checkpoints/backups"
    os.makedirs(backup_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(backup_dir, f"model_pre_refinement_{timestamp}.pth")
    if os.path.exists(original_path):
        shutil.copy(original_path, backup_path)
        print(f"--- 🛡️ Backup created: {backup_path} ---")

    torch.save(model.state_dict(), original_path)
    print(f"--- ✅ Refined model saved to {original_path} ---")

    archive_root = "data/archive"
    session_archive = os.path.join(archive_root, timestamp)
    os.makedirs(session_archive, exist_ok=True)
    for category in ("real", "fake"):
        cat_path = os.path.join("data/feedback_loop", category)
        if not os.path.isdir(cat_path):
            continue
        for fname in os.listdir(cat_path):
            shutil.move(os.path.join(cat_path, fname), os.path.join(session_archive, fname))
    print(f"--- 📦 Data archived to {session_archive} ---")


if __name__ == "__main__":
    train_on_feedback()
