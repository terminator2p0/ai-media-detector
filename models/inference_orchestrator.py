import torch
import os
import librosa
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
from models.model import AIMediaDetector
from models.text_detector import AITextDetector
import cv2

class MediaForensicsOrchestrator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"--- Orchestrator initializing on: {self.device} ---")
        
        # Image Preprocessing
        self.img_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Load Visual Model
        self.visual_model = AIMediaDetector(pretrained=False)
        self.visual_model.load_state_dict(torch.load("models/checkpoints/efficientnet_b4_video_final.pth", map_location=self.device))
        self.visual_model.to(self.device).eval()

        # Load Text Model
        self.text_detector = AITextDetector()

        # Load Audio Model
        self.audio_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
        self.audio_model = AutoModelForAudioClassification.from_pretrained("facebook/wav2vec2-base", num_labels=2)
        self.audio_model.load_state_dict(torch.load("models/checkpoints/wav2vec2_audio_final.pth", map_location=self.device))
        self.audio_model.to(self.device).eval()
        
        print("--- All Forensic Engines Loaded ---")

    def scan_image(self, path):
        img = Image.open(path).convert('RGB')
        img_t = self.img_transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.visual_model(img_t)
            prob = torch.sigmoid(output).item()
        return {"prediction": "AI-Generated" if prob > 0.5 else "Authentic", "confidence": round(prob * 100, 2)}

    def scan_text(self, text):
        return self.text_detector.predict(text)

    def scan_audio(self, path):
        # Safety check for file existence
        if not os.path.exists(path):
            return f"Error: File not found at {path}"
            
        try:
            speech, _ = librosa.load(path, sr=16000)
            inputs = self.audio_extractor(speech, sampling_rate=16000, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                logits = self.audio_model(**inputs).logits
                prob = torch.softmax(logits, dim=-1).squeeze()[1].item()
            return {"prediction": "AI-Generated" if prob > 0.5 else "Authentic", "confidence": round(prob * 100, 2)}
        except Exception as e:
            return f"Error processing audio: {str(e)}"
        
def scan_video(self, path, sample_rate=1.0):
    """Extracts frames from a video and runs the EfficientNet detector."""
    if not os.path.exists(path):
        return f"Error: Video file {path} not found."

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    scores = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Sample 1 frame per second to keep it fast
        if frame_count % int(fps) == 0:
            # Convert OpenCV BGR to PIL RGB
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
        "avg_confidence": round(avg_prob * 100, 2),
        "frames_analyzed": len(scores)
    }