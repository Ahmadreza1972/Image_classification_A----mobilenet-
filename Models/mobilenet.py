import torchvision.models as models
import torch
import torch.nn as nn

class MobileNetV2ForCIFAR8M(nn.Module):
    def __init__(self, num_classes=5, num_layers_to_keep=10):
        super(MobileNetV2ForCIFAR8M, self).__init__()
        
        # Load the pre-trained MobileNetV2 model
        self.mobilenet_v2 = models.mobilenet_v2(pretrained=True)
        
        # Keep only the first `num_layers_to_keep` layers of the feature extractor
        self.mobilenet_v2.features = nn.Sequential(
            self.mobilenet_v2.features[0],  # First conv layer (Conv2d + BN + ReLU6)
            self.mobilenet_v2.features[1],  
            self.mobilenet_v2.features[2],  
            self.mobilenet_v2.features[3],  
            self.mobilenet_v2.features[4]
        )
        
        # Calculate the output size after the truncated feature extractor
        sample_input = torch.randn(1, 3, 32, 32)  # Example input size for MobileNetV2
        with torch.no_grad():
            output_shape = self.mobilenet_v2.features(sample_input).shape
            print("Output shape after truncation:", output_shape)
        flattened_features = output_shape[1] * output_shape[2] * output_shape[3]
        print("Flattened feature size:", flattened_features)
        # Replace the final classifier to match the new output size and number of classes
        self.mobilenet_v2.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_features, 512),  # Example: Intermediate layer size
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)  # Final layer matches the number of classes
        )

    def forward(self, x):
        x = self.mobilenet_v2.features(x)
        return self.mobilenet_v2.classifier(x)