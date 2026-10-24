import torch
import torch.nn as nn
import torchvision.models as models

class ImageEmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=256):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT
        base_resnet = models.resnet50(weights=weights)
        
        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(base_resnet.children())[:-1])
        
        # Non-linear projection head
        self.projection_head = nn.Sequential(
            nn.Linear(base_resnet.fc.in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, embedding_dim)
        )

    def forward(self, x):
        features = self.backbone(x)
        features = torch.flatten(features, 1)
        embeddings = self.projection_head(features)
        # Constrain to unit sphere for cosine similarity
        return nn.functional.normalize(embeddings, p=2, dim=1)