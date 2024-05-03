import os
import sys
import json
from typing import Optional, Dict

import torch
import librosa
import numpy as np

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'beats'))

from .beats_modules.BEATs import BEATsConfig, BEATs


class BEATs_Model:

    # Initialize BEATs Model
    def __init__(self,
                 path_to_checkpoint: str,
                 path_to_ontology: str = os.path.join(DIR_PATH, "ontology.json"),
                 device: Optional[str] = None,
                 hypercategory_mapping: Optional[Dict] = None):
        """
        Initalize the BEATs model

        Args:
            path_to_checkpoint (string): The path to the checkpont file that you downloaded.
            path_to_ontology (string): The path to the ontology file.

        Returns:
           
        """
        self.path_to_checkpoint = path_to_checkpoint

        # Parse Ontology
        self.id2name = parse_ontology(path_to_ontology)
        checkpoint = torch.load(self.path_to_checkpoint)
        self.label_dict = checkpoint['label_dict']

        cfg = BEATsConfig(checkpoint['cfg'])
        self.model = BEATs(cfg)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()

        if device == "cuda" and torch.cuda.is_available():
            self.device = device
            self.model.to(device)
        else:
            self.device = "cpu"

        if hypercategory_mapping is not None:
            self.map_to_hypercategories(hypercategory_mapping)
        else:
            self.hypercategory_mapping = np.array([])

    def make_inference_with_path(self, path_to_audio):
        """Method to make a prediction using a file path

        path_to_audio -- Path to the target audio file
        """

        # Load waveform
        audio, _ = librosa.load(path_to_audio, sr=16000)
        audio = torch.Tensor(audio).unsqueeze(0).to(self.device)

        # Make prediction
        with torch.no_grad():
            probs = self.model.extract_features(audio)[0][0]
            probs = probs.cpu().numpy()

        # Get Index and Class name of prediction
        max_idx = np.argmax(probs)

        label = self.label_dict[max_idx]
        best_score = probs[max_idx]
        label = self.id2name[label]
        predicted_class_idx = max_idx

        return probs, predicted_class_idx, label, best_score

    def make_inference_with_waveform(self, waveform: np.ndarray):
        """Method to make a prediction using a waveform

        waveform -- The audio waveform
        """

        # Load waveform
        waveform = torch.Tensor(waveform).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            probs = self.model.extract_features(waveform)[0][0]
            probs = probs.cpu().numpy()

        # Get Index and Class name of prediction
        max_idx = np.argmax(probs)

        label = self.label_dict[max_idx]
        best_score = probs[max_idx]
        label = self.id2name[label]
        predicted_class_idx = max_idx

        return probs, predicted_class_idx, label, best_score
    
    def map_to_hypercategories(self, hypercategory_mapping: Dict):
        self.hypercategory_mapping = np.array(
            [hypercategory_mapping[self.id2name[self.label_dict[i]]] for i in range(len(self.label_dict))])


def parse_ontology(path_to_ontology):

    with open(path_to_ontology, 'r') as f:
        ontology_lst = json.load(f)

    id2name = {elem['id']: elem['name'] for elem in ontology_lst}

    return id2name


