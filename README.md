# 🛡️ Multi-Modal Forensic AI Investigator

A production-ready deepfake detection suite designed to identify AI-generated artifacts across **Video, Image, Audio, and Text**. This project features a **Self-Supervised Feedback Loop**, allowing the system to ingest human corrections, deduplicate data via MD5 hashing, and retrain its neural engines locally.



## 🚀 Core Features
* **Video & Image Analysis:** Powered by **EfficientNet-B4** for spatial artifact detection.
* **Audio Analysis:** Utilizes **Wav2Vec2-base** to identify synthetic acoustic patterns.
* **Textual Analysis:** Employs a **RoBERTa-based** transformer (TMR-RoBERTa) to detect LLM-generated text.
* **Self-Supervised Learning:** Integrated UI to flag errors, which are then used for local model refinement.
* **Hardware Optimized:** Native support for **Apple Silicon (MPS)** and NVIDIA (CUDA).

---

## 📂 Repository Structure

```text
ai-media-detector/
├── app.py                     # Streamlit Dashboard & Forensic UI
├── train_feedback.py          # Self-supervised retraining engine
├── requirements.txt           # Project dependencies
├── agent/                     # Forensic reasoning logic
├── configs/                   # Global app & model configurations
├── data/                      # Data storage
│   ├── feedback_loop/         # Active training samples (Real/Fake)
│   └── archive/               # Processed training samples
├── models/                    
│   ├── inference_orchestrator.py # Multi-modal model manager
│   └── checkpoints/           # Model weight storage (.pth files)
└── .env.example               # Template for environment secrets

🛠️ Setup Instructions
1. Environment Setup
Bash
# Clone the repository
git clone [https://github.com/terminator2p0/ai-media-detector.git](https://github.com/terminator2p0/ai-media-detector.git)
cd ai-media-detector

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
2. Model Checkpoints
Place your trained weights in models/checkpoints/:

efficientnet_b4_video_final.pth

wav2vec2_audio_final.pth

🖥️ How to Use
1. Launch the Dashboard
Bash
streamlit run app.py
2. The Feedback Loop (Active Learning)
Run a Scan: Upload a suspect file and click "Run Neural Scan".

Audit Result: If the model makes a mistake, click "❌ No, Incorrect".

Deduplication: The system calculates an MD5 Hash to prevent duplicate storage.

Refine the Agent: Once you have a batch of errors, close the app and run:

Bash
python train_feedback.py
This creates a model backup, fine-tunes the weights, and archives the training data.

⚙️ Technical Specifications
Deduplication: Content-based hashing ensures training data integrity.

Refinement: Uses BCEWithLogitsLoss with a 1e-6 learning rate to prevent "Catastrophic Forgetting."

Hardware: Optimized for Mac M-series (MPS) and standard CUDA GPUs.

👤 Author
Abhi Parimisetti

Data Engineer at Parallon

GitHub: @terminator2p0
