import os
import sys
import yaml
from pathlib import Path
import argparse

PROJECT_PATH = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from typing import Optional, Dict
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, f1_score

from utils.init_utils import get_model
from models.callbacks import EarlyStopping
from datasets.datasets import ESC50Dataset

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
            pass
        elif model_name == "passt":
            self.backbone = get_basic_model(mode="logits")
            self.backbone.net.head = nn.Identity()
            in_features = 768
            pass
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



def training_loop(model,
                  train_dset,
                  val_dset,
                  batch_size,
                  epochs,
                  learning_rate,
                  patience,
                  pt_file,
                  num_workers,
                  weight_decay,
                  device="cpu"):
    """Training a deep neural network on the training dataset of AudioSet"""

    # Initialize loss function
    criterion = nn.CrossEntropyLoss().to(device)

    model.to(device)

    # Initialize dataloaders
    train_dloader = DataLoader(train_dset,
                               batch_size=batch_size,
                               shuffle=True,
                               num_workers=num_workers)
    val_dloader = DataLoader(val_dset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers)

    # Initialize optimizer, learning rate shedulers and early stopping.
    optim = Adam(model.classifier.parameters(),
                 lr=learning_rate,
                 weight_decay=weight_decay)

    lr_scheduler = CosineAnnealingLR(optimizer=optim,
                                     T_max=epochs,
                                     eta_min=1e-6)
    early_stopping = EarlyStopping(patience=patience,
                                   verbose=True,
                                   path=os.path.join(PROJECT_PATH,
                                                     "pretrained_models",
                                                     f"{pt_file}"))

    train_loss, val_loss = 0.0, 0.0
    _padding = len(str(epochs + 1))

    for epoch in range(1, epochs + 1):
        model.train()

        with tqdm(train_dloader,
                  unit="batch",
                  leave=False,
                  desc="Training set") as tbatch:
            for i, item in enumerate(tbatch, 1):
                # Forward pass
                X_wave = item['waveform'].to(device)

                label = item['label'].to(device)
                output = model.forward(X_wave)

                loss = criterion(output, label)

                train_loss += loss.item()

                # Backward pass
                optim.zero_grad()
                loss.backward()
                optim.step()
            train_loss /= len(train_dloader)
            lr_scheduler.step()

        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            with tqdm(val_dloader,
                      unit="batch",
                      leave=False,
                      desc="Validation set") as vbatch:
                for i, item in enumerate(vbatch, 1):
                    X_wave = item['waveform'].to(device)
                    label = item['label'].to(device)

                    output = model.forward(X_wave)

                    loss = criterion(output, label)

                    val_loss += loss.item()
                    y_true.append(label.cpu().numpy())
                    batch_preds = output.argmax(axis=1).detach().cpu().numpy()
                    y_pred.append(batch_preds)
        y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)
        val_loss /= len(val_dloader)
        print(
            f"Epoch {epoch:<{_padding}}/{epochs}. Train Loss: {train_loss:.3f}. Val Loss: {val_loss:.3f}. ",
            sep="")
        print(
            f"Val acc: {accuracy_score(y_true, y_pred):.3f}. Val F1: {f1_score(y_true, y_pred, average='macro'):.3f}"
        )
        early_stopping(val_loss, model, epoch)
        if early_stopping.early_stop:
            print("Early Stopping.")
            break
        train_loss, val_loss = 0.0, 0.0


def parse_args():
    parser = argparse.ArgumentParser(
        prog="run_attack_dev.py",
        description=
        "Perform an attack on a path of wav files on a pretrained model")
    parser.add_argument("-c",
                        "--config_file",
                        required=True,
                        help="Configuration file (.yaml) for the attack.")

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    # Get Config file
    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)

    json_hypercategory = config[
        "esc_hypercategories"]  # Hypercategory Mapping
    data_path = config["data_path"]  # Full Path to the Wav files of ESC-50
    metadata_csv = config["metadata_csv"]  # Full Path to Metadata csv

    train_folds = config["train_folds"]
    test_folds = config["test_folds"]

    train_dset = ESC50Dataset(data_path=data_path,
                              metadata_csv=metadata_csv,
                              hypercategory_mapping=json_hypercategory,
                              folds=train_folds)
    test_dset = ESC50Dataset(data_path=data_path,
                             metadata_csv=metadata_csv,
                             hypercategory_mapping=json_hypercategory,
                             folds=test_folds)

    model_name = config["model_architecture"]
    path_to_checkpoint = config["model_pretrained_weights"]
    device = config.get('device', 'cpu')

    model = FineTuneClassifier(model_name=model_name,
                               num_classes=5,
                               weight_path=path_to_checkpoint,
                               freeze_backbone=True,
                               device=device)

    # Trainning Parameters
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    learning_rate = float(config["learning_rate"])
    patience = config["patience"]
    pt_file = config["pt_file"]
    num_workers = config["num_workers"]
    weight_decay = config["weight_decay"]
    training_loop(model=model,
                  train_dset=train_dset,
                  val_dset=test_dset,
                  batch_size=batch_size,
                  epochs=epochs,
                  learning_rate=learning_rate,
                  patience=patience,
                  pt_file=pt_file,
                  num_workers=num_workers,
                  weight_decay=weight_decay,
                  device=device)
