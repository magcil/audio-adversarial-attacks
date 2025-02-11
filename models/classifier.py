import os
import sys
import yaml
from pathlib import Path

PROJECT_PATH = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import numpy as np
import torch
import torch.nn as nn

# Models
from models.beats.beats_modules.BEATs import BEATsConfig, BEATs
from hear21passt.base import get_basic_model


class FineTuneClassifier(nn.Module):

    def __init__(self,
                 model_name,
                 num_classes=5,
                 weight_path=None,
                 freeze_backbone=True,
                 device="cpu"):
        super(FineTuneClassifier, self).__init__()

        self.model_name = model_name
        self.device = device

        if model_name == "beats":
            checkpoint = torch.load(weight_path, weights_only=True)
            cfg = BEATsConfig(checkpoint['cfg'])
            self.backbone = BEATs(cfg)
            self.backbone.load_state_dict(checkpoint['model'])

            in_features = 768
        elif model_name == "passt":
            self.backbone = get_basic_model(mode="logits")
            self.backbone.net.head = nn.Identity()
            in_features = 768
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(nn.Linear(in_features, 512),
                                        nn.BatchNorm1d(512), nn.ReLU(),
                                        nn.Linear(512,
                                                  256), nn.BatchNorm1d(256),
                                        nn.ReLU(), nn.Linear(256, 128),
                                        nn.BatchNorm1d(128), nn.ReLU(),
                                        nn.Linear(128, num_classes))

    def forward(self, x):

        if self.model_name == "beats":
            # embeddings = self.get_embeddings_BEATs(x)
            with torch.no_grad():
                _, _, embeddings = self.backbone.extract_features(x)
                embeddings = embeddings.mean(dim=1)
        else:
            with torch.no_grad():
                embeddings = self.backbone(x)

        output = self.classifier(embeddings)
        return output