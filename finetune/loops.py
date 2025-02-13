import os
import sys

PROJECT_PATH = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, PROJECT_PATH)

import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
from finetune.callbacks import EarlyStopping
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
    train_dloader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dloader = DataLoader(val_dset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Initialize optimizer, learning rate shedulers and early stopping.
    optim = Adam(model.classifier.parameters(), lr=learning_rate, weight_decay=weight_decay)

    lr_scheduler = CosineAnnealingLR(optimizer=optim, T_max=epochs, eta_min=1e-6)
    early_stopping = EarlyStopping(patience=patience,
                                   verbose=True,
                                   path=os.path.join(PROJECT_PATH, "pretrained_models", f"{pt_file}"))

    train_loss, val_loss = 0.0, 0.0
    _padding = len(str(epochs + 1))

    for epoch in range(1, epochs + 1):
        model.train()

        with tqdm(train_dloader, unit="batch", leave=False, desc="Training set") as tbatch:
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
            with tqdm(val_dloader, unit="batch", leave=False, desc="Validation set") as vbatch:
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
        print(f"Epoch {epoch:<{_padding}}/{epochs}. Train Loss: {train_loss:.3f}. Val Loss: {val_loss:.3f}. ", sep="")
        print(f"Val acc: {accuracy_score(y_true, y_pred):.3f}. Val F1: {f1_score(y_true, y_pred, average='macro'):.3f}")
        early_stopping(val_loss, model, epoch)
        if early_stopping.early_stop:
            print("Early Stopping.")
            break
        train_loss, val_loss = 0.0, 0.0


def validation_loop(model, path_to_pt_file, test_dset, batch_size, num_workers, device):
    """Validating the model on the validation set of AudioSet"""

    model.load_state_dict(torch.load(path_to_pt_file, weights_only=True))
    model = model.to(device)

    test_dloader = DataLoader(test_dset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model.eval()
    y_true, y_pred, posteriors = [], [], []

    with torch.no_grad():
        with tqdm(test_dloader, unit="batch", leave=False, desc="Test set") as vbatch:
            for i, item in enumerate(vbatch, 1):

                X_wave = item['waveform'].to(device)
                label = item['label'].to(device)

                output = model.forward(X_wave)
                
                probs = F.softmax(output, dim=1)
                y_true.append(label.cpu().numpy())
                y_pred.append(output.cpu().numpy())
                posteriors.append(probs.cpu().numpy())

    y_true, y_pred, posteriors = np.concatenate(y_true), np.concatenate(y_pred), np.concatenate(posteriors)

    return {"y_true": y_true, "y_pred": y_pred, "posteriors": posteriors}
