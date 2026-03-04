import torch
import os
import librosa
import cv2
import gdown  # <--- NEW: The bulletproof Google Drive downloader
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

# Local Imports 
from models.model import AIMediaDetector
from models.text_detector import AITextDetector

def download_from_gdrive(file_id, destination):
    """
    Uses gdown to securely bypass Google Drive's anti-bot and 
    large-file warning screens.
    """
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    # gdown automatically constructs the correct API URL and handles tokens
    url = f"https://drive.google.com/uc?id={file_id}"
    
    # quiet=False ensures you see the actual MB progress bar in Streamlit logs
    gdown.download(url, destination, quiet=False)

class MediaForensicsOrchestrator:
    def __init__(self):
        # 1. Device Detection
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() 
            else "mps" if torch.backends.mps.is_available() 
            else "cpu"
        )
        print(f"--- Orchestrator initializing on: {self.device} ---")
        
        # 2. Define Weights Configuration
        self.checkpoints = {
            "visual": {
                "id": "1_1b7ng-ZjAY4ZBRukW4NUeQ_z7yOsMY9", 
                "path": "models/checkpoints/efficientnet_b4_video_final.pth"
            },
            "audio": {
                "id": "1238Ngl7hB2E6jzUcSVCX_ebQS1ZIHsA4",
                "path": "models/checkpoints/wav2vec2_audio_final.pth"
            }
        }

        # 3. Purge Corrupted Files & Download Real Weights
        for key, info in self.checkpoints.items():
            # Delete corrupted "dummy" files (smaller than 1MB)
            if os.path.exists(info["path"]) and os.path.getsize(info["path"]) < 1000000:
                print(f"⚠️ Removing corrupted {key} file...")
                os.remove(info["path"])

            # Download if missing
            if not os.path.exists(info["path"]):
                print(f"📦 {key} weights missing. Attempting cloud download...")
                download_from_gdrive(info["id"], info["path"])
            else:
                print(f"✅ {key} weights found locally.")

        # 4. Image Preprocessing
        self.img_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # 5. Load Visual Model (With PyTorch 2.6 weights_only Bypass)
        self.visual_model = AIMediaDetector(pretrained=False)
        self.visual_model.load_state_dict(
            torch.load(
                self.checkpoints["visual"]["path"], 
                map_location=self.device, 
                weights_only=False
            )
        )
        self.visual_model.to(self.device).eval()

        # 6. Load Text Model
        self.text_detector = AITextDetector()

        # 7. Load Audio Model (With PyTorch 2.6 weights_only Bypass)
        self.audio_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
        self.audio_model = AutoModelForAudioClassification.from_pretrained(
            "facebook/wav2vec2-base", 
            num_labels=2
        )
        self.audio_model.load_state_dict(
            torch.load(
                self.checkpoints["audio"]["path"], 
                map_location=self.device, 
                weights_only=False
            )
        )
        self.audio_model.to(self.device).eval()
        
        print("--- All Forensic Engines Successfully Loaded ---")

    def scan_image(self, path):
        img = Image.open(path).convert('RGB')
        img_t = self.img_transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.visual_model(img_t)
            prob = torch.sigmoid(output).item()
        return {
            "prediction": "AI-Generated" if prob > 0.5 else "Authentic", 
            "confidence": round(prob * 100, 2)
        }

    def scan_text(self, text):
        return self.text_detector.predict(text)

    def scan_audio(self, path):
        if not os.path.exists(path):
            return f"Error: File not found at {path}"
        try:
            speech, _ = librosa.load(path, sr=16000)
            inputs = self.audio_extractor(
                speech, 
                sampling_rate=16000, 
                return_tensors="pt", 
                padding=True
            ).to(self.device)
            with torch.no_grad():
                logits = self.audio_model(**inputs).logits
                prob = torch.softmax(logits, dim=-1).squeeze()[1].item()
            return {
                "prediction": "AI-Generated" if prob > 0.5 else "Authentic", 
                "confidence": round(prob * 100, 2)
            }
        except Exception as e:
            return f"Error processing audio: {str(e)}"

    def scan_video(self, path, sample_rate=1.0):
        if not os.path.exists(path):
            return f"Error: Video file {path} not found."
        
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if fps <= 0:
            fps = 30.0
            
        frame_count = 0
        scores = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % int(fps / sample_rate) == 0:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                img_t = self.img_transform(img).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    output = self.visual_model(img_t)
                    prob = torch.sigmoid(output).item()
                    scores.append(prob)
                    
            frame_count += 1
            
        cap.release()
        
        if not scores:
            return "Error: Could not extract frames from video."
            
        avg_prob = sum(scores) / len(scores)
        return {
            "prediction": "AI-Generated/Deepfake" if avg_prob > 0.5 else "Authentic",
            "average_confidence": round(avg_prob * 100, 2),
            "frames_analyzed": len(scores)
        }