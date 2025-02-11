import os
import sys
import json
import csv
from typing import Optional, Dict

import torch
import librosa
import numpy as np

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from hear21passt.base import get_basic_model

class Passt_Model:

    # Initialize PaSST Model
    def __init__(self,
                 path_to_ontology: str = os.path.join(DIR_PATH,
                                                      "ontology.json"),
                 device: Optional[str] = "cpu",
                 hypercategory_mapping: Optional[Dict] = None):
        """
        
        Initalize the PaSST model

        Args:
            path_to_ontology (string): The path to the ontology file.

        Returns:
           
        """
        self.ontology = parse_ontology(path_to_ontology)

        self.model = get_basic_model(mode="logits")
        self.model.eval()

        if device.startswith("cuda") and torch.cuda.is_available():
            self.device = device
            self.model.to(device)

        else:
            self.device = "cpu"
            self.model.to(self.device)

        
        with open(hypercategory_mapping, 'r') as f:
            hypercategory_dict = json.load(f)

        self.map_to_hypercategories(hypercategory_dict)
    

    def make_inference_with_path(self, path_to_audio):
        """Method to make a prediction using a file path

        path_to_audio -- Path to the target audio file
        """

        # Load waveform
        audio, _ = librosa.load(path_to_audio, sr=16000)
        audio = torch.from_numpy(audio).unsqueeze(0).to(self.device)

        
        # Make prediction
        with torch.no_grad():
            probs = self.model(audio)
            probs = torch.nn.functional.softmax(probs, dim=1)
            probs=probs.squeeze(0).cpu().numpy()

        # Get Index and Class name of prediction
        max_idx = np.argmax(probs)

        best_score = probs[max_idx]
        label = self.ontology[max_idx]
        predicted_class_idx = max_idx

        return {"probs": probs, "predicted_class_idx": predicted_class_idx, "label": label, "best_score": best_score}

    def make_inference_with_waveform(self, waveform: np.ndarray):
        """Method to make a prediction using a waveform

        waveform -- The audio waveform
        """
        waveform = waveform.astype(np.float32)

        # Load waveform
        waveform = torch.from_numpy(waveform).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            probs = self.model(waveform)
            probs = torch.nn.functional.softmax(probs, dim=1)
            probs = probs.squeeze(0).cpu().numpy()

        # Get Index and Class name of prediction
        max_idx = np.argmax(probs)

        label = self.ontology[max_idx]
        best_score = probs[max_idx]
        predicted_class_idx = max_idx

        return {"probs": probs, "predicted_class_idx": predicted_class_idx, "label": label, "best_score": best_score}

    def map_to_hypercategories(self, hypercategory_mapping: Dict):
        self.hypercategory_mapping = np.array([hypercategory_mapping[name] for idx,name in self.ontology.items()])


def parse_ontology(path_to_ontology):

    with open(path_to_ontology, 'r') as f:
        ontology_list = json.load(f)
    
    # Convert keys to integers
    ontology_list = {int(k): v for k, v in ontology_list.items()}

    return ontology_list