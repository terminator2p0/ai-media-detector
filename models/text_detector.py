import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class AITextDetector:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Loading TMR-RoBERTa on: {self.device}...")
        
        # Using a fully fine-tuned model (no broken LoRA adapters needed!)
        model_id = "Oxidane/tmr-ai-text-detector"
        
        print("Fetching model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        
        self.model.to(self.device)
        self.model.eval() 
        print("Text Detector ready!\n")

    def predict(self, text: str) -> dict:
        if not text or len(text.strip()) == 0:
            return {"error": "Empty text provided."}

        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512, 
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).squeeze()
            
        # Label 0 = Human, Label 1 = AI
        human_prob = probs[0].item()
        ai_prob = probs[1].item()
        
        prediction = "AI-Generated" if ai_prob > 0.5 else "Human-Written"
        
        return {
            "prediction": prediction,
            "ai_probability": round(ai_prob * 100, 2),
            "human_probability": round(human_prob * 100, 2)
        }

if __name__ == "__main__":
    detector = AITextDetector()
    
    sample_text = "Right from the beginning we projected four to five weeks, but we have the capability to go far longer than that."
    
    print("Analyzing sample text...")
    print(f"Input: '{sample_text}'\n")
    
    results = detector.predict(sample_text)
    
    for key, value in results.items():
        print(f"{key.replace('_', ' ').title()}: {value}")