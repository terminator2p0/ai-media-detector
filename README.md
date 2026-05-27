# 🛡️ Multi-Modal Forensic AI Investigator

A production-ready deepfake detection suite designed to identify AI-generated artifacts across **Video, Image, Audio, and Text**. This project features a **Self-Supervised Feedback Loop**, allowing the system to ingest human corrections, deduplicate data via MD5 hashing, and retrain its neural engines locally.

---

## 🚀 Core Features

* **Video & Image Analysis:** Powered by **EfficientNet-B4** for spatial artifact detection.
* **Audio Analysis:** Utilizes **Wav2Vec2-base** to identify synthetic acoustic patterns.
* **Textual Analysis:** Employs a **RoBERTa-based** transformer (TMR-RoBERTa) to detect LLM-generated text.
* **Self-Supervised Learning:** Integrated UI to flag errors, which are then used for local model refinement.
* **Gemini Forensic Agent:** Generates detailed forensic analysis reports explaining *why* media is flagged. Multi-modal cross-referencing for videos (visual + audio).
* **Dataset Acquisition:** One-command download of training data from HuggingFace (CIFAKE, deepfake-audio, HC3).
* **SQLite Persistence:** Every scan and audit is logged to `data/forensic.db` — viewable in DB Browser for SQLite.
* **Hardware Optimized:** Native support for **Apple Silicon (MPS)** and NVIDIA (CUDA).

---

## 📂 Repository Structure

```text
ai-media-detector/
├── app.py                          # Streamlit Dashboard & Forensic UI
├── db.py                           # SQLite persistence (predictions + feedback)
├── train_feedback.py               # Fine-tune visual model on user feedback
├── train_all.py                    # Full training orchestrator (image/audio/text)
├── requirements.txt
├── agent/
│   └── forensic_agent.py           # Gemini-powered forensic reasoning + LangChain agent
├── configs/
│   └── training_configs.yaml
├── data/
│   ├── forensic.db                 # SQLite DB (auto-created)
│   ├── feedback_loop/{real,fake}/  # User-audited media for retraining
│   ├── training/{image,audio,text}/ # Acquired datasets
│   └── archive/                    # Archived training batches
├── data_pipeline/
│   ├── acquire_datasets.py         # Download CIFAKE + audio + HC3 from HuggingFace
│   ├── audio_dataloader.py
│   ├── video_dataloader.py
│   └── ...
├── models/
│   ├── inference_orchestrator.py   # Multi-modal model manager
│   ├── model.py                    # EfficientNet-B4 architecture
│   ├── text_detector.py            # TMR-RoBERTa wrapper
│   └── checkpoints/                # Model weights (.pth)
├── scripts/
│   └── open_db.py                  # Open forensic.db in DB Browser for SQLite
└── .env.example
```

---

## 🛠️ Setup Instructions

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/terminator2p0/ai-media-detector.git
cd ai-media-detector

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. API Keys

```bash
cp .env.example .env
# Edit .env and add your Google API key (for Gemini forensic reports)
```

### 3. Model Checkpoints

Place your trained weights in `models/checkpoints/` (auto-downloaded from Google Drive on first run):

* `efficientnet_b4_video_final.pth`
* `wav2vec2_audio_final.pth`

### 4. Download Training Datasets (optional)

```bash
python data_pipeline/acquire_datasets.py                 # all modalities
python data_pipeline/acquire_datasets.py --only image    # just CIFAKE
python data_pipeline/acquire_datasets.py --only audio    # deepfake audio
python data_pipeline/acquire_datasets.py --only text     # HC3 corpus
python data_pipeline/acquire_datasets.py --max-samples 500  # quick test run
```

### 5. Train Models (optional)

```bash
python train_all.py                  # train all three models
python train_all.py --only image     # just the visual model
python train_all.py --only audio     # just the audio model
python train_all.py --only text      # just the text model
python train_all.py --epochs 20      # override epoch count
```

---

## 🖥️ How to Use

### 1. Launch the Dashboard

```bash
streamlit run app.py
```

### 2. The Feedback Loop (Active Learning)

* **Run a Scan:** Upload a suspect file and click "Run Neural Scan".
* **Audit Result:** Click "✅ Yes, Correct" or "❌ No, Incorrect". Both signals are persisted.
* **Persistence:** Every scan is logged to a SQLite DB (`data/forensic.db`). Audited media is hashed (MD5), deduplicated, and copied into `data/feedback_loop/{real|fake}/`.
* **Refine the Agent:** Once you have a batch of audited samples, run:

```bash
python train_feedback.py
```

This pulls labeled samples from the DB, creates a model backup, fine-tunes the weights, and archives the training data.

### 3. Inspect the Backend

```bash
python -c "import db, json; print(json.dumps(db.stats(), indent=2))"
sqlite3 data/forensic.db "SELECT id, file_type, model_prediction, confidence, created_at FROM predictions ORDER BY id DESC LIMIT 10;"
```

---

## 🗄️ Backend Architecture

The project is currently **local-first** — no external services required.

### Current Setup

| Concern | Implementation |
| --- | --- |
| Prediction / feedback storage | **SQLite** (`data/forensic.db`) — see `db.py` |
| Media storage | Local filesystem under `data/feedback_loop/{real,fake}/` |
| Model weights | Local filesystem + Google Drive bootstrap via `gdown` |
| Web UI | Streamlit (`app.py`) |
| Agent (optional) | LangChain + Gemini 2.5 Flash (`agent/forensic_agent.py`) |

### Recommended Production Path

When you outgrow single-user, single-machine usage:

1. **Database — swap SQLite for PostgreSQL.**
   Keep the same schema (`predictions`, `feedback`). Use SQLAlchemy + Alembic for migrations. Hosted options: Neon, Supabase, RDS.

2. **Media storage — move off local disk to object storage.**
   S3-compatible (AWS S3, Cloudflare R2, Backblaze B2). Store the object key in the `stored_media_path` column instead of a local path. Cheap, durable, and survives container restarts.

3. **Split UI from inference — introduce a FastAPI service.**
   Streamlit becomes one client. The API exposes `/scan`, `/feedback`, `/stats` and owns DB + storage access. This unlocks: mobile clients, batch ingestion, an auth layer, and rate limiting.

4. **Move training off the user's box.**
   A scheduled job (Modal, Runpod, GitHub Actions + a GPU runner, or Airflow on K8s) pulls labeled samples from Postgres, fine-tunes, and writes a versioned checkpoint to object storage. The orchestrator pulls the latest checkpoint at boot.

5. **Auth + multi-tenancy** (when needed).
   Add a `users` table, a `user_id` foreign key on `predictions` / `feedback`, and put the FastAPI service behind an auth provider (Clerk, Auth0, or a JWT issuer of your choice).

For your current scale — solo use, single laptop — **SQLite + local filesystem is the right call**. Stay there until something forces the move.

---

## ⚙️ Technical Specifications

* **Deduplication:** Content-based hashing ensures training data integrity.
* **Refinement:** Uses `BCEWithLogitsLoss` with a `1e-6` learning rate to prevent "Catastrophic Forgetting."
* **Hardware:** Optimized for Mac M-series (MPS).

---

## 👤 Author

**Abhi Parimisetti**

* Data Engineer at Parallon
* GitHub: [@terminator2p0](https://github.com/terminator2p0)
