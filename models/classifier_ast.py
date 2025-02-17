import os
import sys
import yaml
import json
from pathlib import Path
import torchaudio

PROJECT_PATH = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from typing import Optional, Dict
import numpy as np
import torch
import torch.nn as nn

# Models
from models.AST.ast_model.ast_models import ASTModel
from datasets import ESC_CLASS_MAPPING, ESC_INV_CLASS_MAPPING


class FineTuneClassifierAST(nn.Module):

    def __init__(self, model_name, num_classes=5, weight_path=None, freeze_backbone=True, device="cpu"):
        super(FineTuneClassifierAST, self).__init__()

        self.model_name = model_name
        self.device = device

        self.backbone = self._load_ast_model(weight_path).to(self.device)
        in_features = 768

        self.backbone.mlp_head = nn.Identity()
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(nn.Linear(in_features, 512), nn.BatchNorm1d(512), nn.ReLU(),
                                        nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Linear(256, 128),
                                        nn.BatchNorm1d(128), nn.ReLU(), nn.Linear(128, num_classes))

        # Move model components to the specified device
        self.backbone.to(self.device)
        self.classifier.to(self.device)

        self.map_to_hypercategories()

    def _load_ast_model(self, path_to_checkpoint):
        model = ASTModel(
            label_dim=527,  # AudioSet has 527 classes
            fstride=10,
            tstride=10,
            input_fdim=128,
            input_tdim=1024,
            imagenet_pretrain=True,
            audioset_pretrain=False,
            model_size='base384',
            verbose=False)

        checkpoint = torch.load(path_to_checkpoint, map_location='cuda')
        audio_model = torch.nn.DataParallel(model)
        audio_model.load_state_dict(checkpoint)
        audio_model = audio_model.to(self.device)

        return model

    def _make_features(self, waveform, mel_bins=128, target_length=1024, sr=16000):

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        batch_size = waveform.shape[0]
        features = []

        for i in range(batch_size):
            fbank = torchaudio.compliance.kaldi.fbank(waveform[i].unsqueeze(0),
                                                      htk_compat=True,
                                                      sample_frequency=sr,
                                                      use_energy=False,
                                                      window_type='hanning',
                                                      num_mel_bins=mel_bins,
                                                      dither=0.0,
                                                      frame_shift=10)

            n_frames = fbank.shape[0]

            p = target_length - n_frames
            if p > 0:
                m = torch.nn.ZeroPad2d((0, 0, 0, p))
                fbank = m(fbank)
            elif p < 0:
                fbank = fbank[0:target_length, :]

            fbank = (fbank - (-4.2677393)) / (4.5689974 * 2)

            features.append(fbank)

        output = torch.stack(features, dim=0)

        return output

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        x = x.to(self.device)

        feats = self._make_features(x, mel_bins=128)

        embeddings = self.backbone.forward(feats)

        output = self.classifier(embeddings)

        return output

    def make_inference_with_waveform(self, waveform: np.ndarray):
        """Method to make a prediction using a waveform

        waveform -- The audio waveform
        """
        self.backbone.eval()
        self.classifier.eval()

        # Load waveform
        waveform = torch.Tensor(waveform).to(self.device)

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
