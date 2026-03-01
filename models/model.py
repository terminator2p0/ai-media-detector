import torch
import torch.nn as nn
from torchvision import models

class AIMediaDetector(nn.Module):
    def __init__(self, pretrained=True):
        super(AIMediaDetector, self).__init__()
        
        # Load the pre-trained EfficientNet-B4 backbone
        weights = models.EfficientNet_B4_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b4(weights=weights)
        
        # The original classifier has a Dropout and a Linear layer.
        # We need to replace the final Linear layer to output 1 value 
        # (for Binary Cross-Entropy loss).
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, 1)
        
    def forward(self, x):
        # Returns the raw logit score. 
        # A sigmoid function will be applied during loss calculation/inference.
        return self.backbone(x)

# Quick test block to ensure it initializes correctly
if __name__ == "__main__":
    model = AIMediaDetector()
    print("EfficientNet-B4 loaded successfully with modified classification head!")
    
    # Test with a dummy image tensor (Batch Size 1, 3 Channels, 224x224)
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape} (Expected: torch.Size([1, 1]))")