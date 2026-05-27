"""Download and prepare training datasets for all modalities.

Datasets used:
  Image — CIFAKE (CIFAR-10 reals + Stable Diffusion fakes), ~120k samples
  Audio — garystafford/deepfake-audio-detection, ~1.8k samples
  Text  — Hello-SimpleAI/HC3 (Human ChatGPT Comparison Corpus), ~24k QA pairs

Each downloader:
  1. Streams from HuggingFace Hub
  2. Saves processed files into data/training/{modality}/
  3. Registers every sample in the SQLite DB (predictions table with model_version='ground_truth')

Run:
  python data_pipeline/acquire_datasets.py            # all modalities
  python data_pipeline/acquire_datasets.py --only image
  python data_pipeline/acquire_datasets.py --only audio
  python data_pipeline/acquire_datasets.py --only text
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import db

TRAINING_ROOT = "data/training"


def acquire_images(max_samples: int | None = None):
    """Download CIFAKE and save as JPEGs organized into real/fake directories."""
    from datasets import load_dataset

    out_root = os.path.join(TRAINING_ROOT, "image")
    for label_name in ("real", "fake"):
        os.makedirs(os.path.join(out_root, label_name), exist_ok=True)

    print("Downloading CIFAKE image dataset...")
    ds = load_dataset("dragonintelligence/CIFAKE-image-dataset")

    count = 0
    for split in ds:
        print(f"  Processing split: {split} ({len(ds[split])} samples)")
        for i, item in enumerate(ds[split]):
            if max_samples and count >= max_samples:
                break
            img = item["image"]
            label = item["label"]  # 0=real, 1=fake
            label_name = "fake" if label == 1 else "real"
            pred_label = "AI-Generated" if label == 1 else "Authentic"

            fname = f"{split}_{i:06d}.jpg"
            fpath = os.path.join(out_root, label_name, fname)

            if not os.path.exists(fpath):
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img.save(fpath, "JPEG", quality=95)

            db.log_prediction(
                file_type="image",
                model_prediction=pred_label,
                confidence=100.0,
                raw_result={"source": "CIFAKE", "split": split, "ground_truth": label_name},
                file_hash=f"cifake_{split}_{i}",
                file_name=fname,
                model_version="ground_truth",
            )
            count += 1

        if max_samples and count >= max_samples:
            break

    print(f"  Images saved: {count} -> {out_root}")
    return count


def acquire_audio(max_samples: int | None = None):
    """Download deepfake-audio-detection and save as WAV files."""
    from datasets import load_dataset, Audio

    out_root = os.path.join(TRAINING_ROOT, "audio")
    for label_name in ("real", "fake"):
        os.makedirs(os.path.join(out_root, label_name), exist_ok=True)

    print("Downloading deepfake audio detection dataset...")
    ds = load_dataset("garystafford/deepfake-audio-detection", split="train")
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    import soundfile as sf

    count = 0
    for i, item in enumerate(ds):
        if max_samples and count >= max_samples:
            break
        label = item["label"]  # 0=real(human), 1=fake(AI)
        label_name = "fake" if label == 1 else "real"
        pred_label = "AI-Generated" if label == 1 else "Authentic"
        audio_array = item["audio"]["array"]
        sr = item["audio"]["sampling_rate"]

        fname = f"audio_{i:06d}.wav"
        fpath = os.path.join(out_root, label_name, fname)

        if not os.path.exists(fpath):
            sf.write(fpath, audio_array, sr)

        db.log_prediction(
            file_type="audio",
            model_prediction=pred_label,
            confidence=100.0,
            raw_result={"source": "garystafford/deepfake-audio-detection", "ground_truth": label_name},
            file_hash=f"audio_gary_{i}",
            file_name=fname,
            model_version="ground_truth",
        )
        count += 1

    print(f"  Audio samples saved: {count} -> {out_root}")
    return count


def acquire_text(max_samples: int | None = None):
    """Download HC3 and save human vs ChatGPT answers as JSONL."""
    from datasets import load_dataset

    out_root = os.path.join(TRAINING_ROOT, "text")
    os.makedirs(out_root, exist_ok=True)

    print("Downloading HC3 text dataset...")
    subsets = ["finance", "medicine", "open_qa", "reddit_eli5", "wiki_csai"]
    all_rows: list[dict] = []

    for subset in subsets:
        print(f"  Loading subset: {subset}")
        try:
            ds = load_dataset("Hello-SimpleAI/HC3", subset, split="train")
        except Exception as e:
            print(f"    Skipped {subset}: {e}")
            continue

        for item in ds:
            question = item.get("question", "")
            for ans in item.get("human_answers", []):
                all_rows.append({"text": ans, "label": "human", "source_subset": subset, "question": question})
            for ans in item.get("chatgpt_answers", []):
                all_rows.append({"text": ans, "label": "ai", "source_subset": subset, "question": question})

    if max_samples:
        all_rows = all_rows[:max_samples]

    out_file = os.path.join(out_root, "hc3_combined.jsonl")
    with open(out_file, "w") as fp:
        for row in all_rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    human_count = sum(1 for r in all_rows if r["label"] == "human")
    ai_count = sum(1 for r in all_rows if r["label"] == "ai")

    for i, row in enumerate(all_rows):
        pred = "AI-Generated" if row["label"] == "ai" else "Human-Written"
        db.log_prediction(
            file_type="text",
            model_prediction=pred,
            confidence=100.0,
            raw_result={"source": "HC3", "subset": row["source_subset"], "ground_truth": row["label"]},
            file_hash=f"hc3_{i}",
            file_name=None,
            model_version="ground_truth",
        )

    print(f"  Text samples saved: {len(all_rows)} (human={human_count}, ai={ai_count}) -> {out_file}")
    return len(all_rows)


MODALITY_MAP = {
    "image": acquire_images,
    "audio": acquire_audio,
    "text": acquire_text,
}


def main():
    parser = argparse.ArgumentParser(description="Download training datasets")
    parser.add_argument("--only", choices=list(MODALITY_MAP.keys()), help="Download a single modality")
    parser.add_argument("--max-samples", type=int, default=None, help="Cap per modality (for quick testing)")
    args = parser.parse_args()

    db.init_db()

    targets = [args.only] if args.only else list(MODALITY_MAP.keys())
    total = 0
    for mod in targets:
        print(f"\n{'='*60}")
        print(f"  Acquiring: {mod.upper()}")
        print(f"{'='*60}")
        total += MODALITY_MAP[mod](max_samples=args.max_samples)

    print(f"\n--- Done. {total} total samples registered in DB. ---")
    print(f"View data: python scripts/open_db.py")


if __name__ == "__main__":
    main()
