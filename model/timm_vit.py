import torch.nn as nn
import timm

class TimmViT(nn.Module):
    def __init__(self, pretrained=True, num_classes=10):
        super(TimmViT, self).__init__()
        self.model = timm.create_model("vit_base_patch16_224", pretrained=pretrained)
        self.model.head = nn.Linear(self.model.head.in_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x