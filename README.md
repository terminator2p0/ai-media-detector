🛡️ Multi-Modal Forensic AI Investigator
A production-ready deepfake detection suite designed to identify AI-generated artifacts across Video, Image, Audio, and Text. This project features a Self-Supervised Feedback Loop, allowing the system to ingest human corrections, deduplicate data via MD5 hashing, and retrain its neural engines locally.

🚀 Core Features
Video & Image Analysis: Powered by EfficientNet-B4 for spatial artifact detection.

Audio Analysis: Utilizes Wav2Vec2-base to identify synthetic acoustic patterns.

Textual Analysis: Employs a RoBERTa-based transformer (TMR-RoBERTa) to detect LLM-generated text.

Self-Supervised Learning: Integrated UI buttons to flag errors, which are then used for local model refinement.

Hardware Optimized: Native support for Apple Silicon (MPS) and NVIDIA (CUDA) for high-speed inference.

📂 Repository Structure
Based on the current project directory:

Plaintext
ai-media-detector/
├── app.py                     # Streamlit Dashboard & Forensic UI
├── train_feedback.py          # Self-supervised retraining engine
├── requirements.txt           # Project dependencies
├── agent/                     # Forensic reasoning logic
├── configs/                   # Global app & model configurations
├── data/                      # Feedback loop, archives, and processed media
├── models/                    
│   ├── inference_orchestrator.py # Multi-modal model manager
│   └── checkpoints/           # Model weight storage (.pth files)
├── wandb/                     # Weights & Biases experiment tracking
└── .env.example               # Template for environment secrets
🛠️ Setup Instructions
1. Environment Setup
Bash
# Clone the repository
git clone https://github.com/your-username/ai-media-detector.git
cd ai-media-detector

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
2. Model Checkpoints
Ensure your trained weights are placed in models/checkpoints/:

efficientnet_b4_video_final.pth

wav2vec2_audio_final.pth

🖥️ How to Use
1. Launch the Dashboard
Bash
streamlit run app.py
2. The Feedback Loop (Active Learning)
Run a Scan: Upload a suspect file and click "Run Neural Scan".

Audit Result: If the model makes a mistake, click "❌ No, Incorrect".

Deduplication: The system automatically calculates the file's MD5 Hash. If the file already exists in the feedback loop, it skips it to save space and maintain training integrity.

Refine the Agent: Once you have collected a batch of errors, run the retraining script:

Bash
python train_feedback.py
This script creates a backup of your current model.

It fine-tunes the model on the new "hard cases" using a micro-learning rate.

It archives the training data to keep the feedback loop clean.

⚙️ Technical Specifications
Deduplication: Content-based hashing ensures training data remains distinct regardless of filename changes.

Precision Refinement: Uses BCEWithLogitsLoss with a 1e-6 learning rate to prevent "Catastrophic Forgetting" during refinement sessions.

Frameworks: PyTorch, Streamlit, Hugging Face Transformers, OpenCV, Librosa
