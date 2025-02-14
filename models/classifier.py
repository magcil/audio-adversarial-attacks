import os
import sys
import yaml
import json
from pathlib import Path

PROJECT_PATH = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from typing import Optional, Dict
import numpy as np
import torch
import torch.nn as nn

# Models
from models.beats.beats_modules.BEATs import BEATsConfig, BEATs
from hear21passt.base import get_basic_model
from datasets import ESC_CLASS_MAPPING, ESC_INV_CLASS_MAPPING


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
        
        # Move model components to the specified device
        self.backbone.to(self.device)
        self.classifier.to(self.device)

        self.map_to_hypercategories()


    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        x = x.to(self.device)
        if self.model_name == "beats":
            with torch.no_grad():
                _, _, embeddings = self.backbone.extract_features(x)
                embeddings = embeddings.mean(dim=1)
        else:
            with torch.no_grad():
                embeddings = self.backbone(x)

        output = self.classifier(embeddings)
        return output
    
    def make_inference_with_waveform(self, waveform: np.ndarray):
        """Method to make a prediction using a waveform

        waveform -- The audio waveform
        """
        self.backbone.eval()
        self.classifier.eval()

        # Load waveform
        waveform = torch.Tensor(waveform).unsqueeze(0).to(self.device)

        # Make prediction
        with torch.no_grad():
            probs = self.forward(waveform)[0]
            probs = probs.cpu().numpy()

        # Get Index and Class name of prediction
        max_idx = np.argmax(probs)

        label = ESC_INV_CLASS_MAPPING[max_idx]
        best_score = probs[max_idx]
        predicted_class_idx = max_idx
        
        return {"probs": probs, "predicted_class_idx": predicted_class_idx, "label": label, "best_score": best_score}
    
    def map_to_hypercategories(self):
        self.hypercategory_mapping = np.array(list(ESC_INV_CLASS_MAPPING.values()))
