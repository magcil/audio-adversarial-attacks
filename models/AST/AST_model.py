import os
import sys
import json
from typing import Optional, Dict

import torch
import torchaudio
import librosa
import numpy as np

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from ast_model.ast_models import ASTModel

class AST_Model:

    # Initialize AST Model
    def __init__(self,
                 path_to_checkpoint: str,
                 path_to_ontology: str = os.path.join(DIR_PATH, "ontology.json"), 
                 device: Optional[str] = None,
                 hypercategory_mapping: Optional[os.PathLike] = None):
        """
        Initalize the AST model

        Args:
            path_to_checkpoint (string): The path to the checkpont file that you downloaded.
            path_to_ontology (string): The path to the ontology file.

        Returns:
           
        """
        self.path_to_checkpoint = path_to_checkpoint

        # Dictionary containing 
        self.ontology = parse_ontology(path_to_ontology)

        if device.startswith("cuda") and torch.cuda.is_available():
            self.device = device
        else:
            self.device = "cpu"

        self.model = self._load_ast_model(path_to_checkpoint).to(self.device)

        with open(hypercategory_mapping, 'r') as f:
            hypercategory_dict = json.load(f)

        self.map_to_hypercategories(hypercategory_dict)


    def _load_ast_model(self, path_to_checkpoint):
        model = ASTModel(
            label_dim=527,  # AudioSet has 527 classes
            fstride=10,
            tstride=10,
            input_fdim=128,
            input_tdim=1024,
            imagenet_pretrain=True,
            audioset_pretrain=False,
            model_size='base384'
        )

        checkpoint = torch.load(path_to_checkpoint, map_location='cuda')
        audio_model = torch.nn.DataParallel(model, device_ids=[0])
        audio_model.load_state_dict(checkpoint)
        audio_model = audio_model.to(self.device)
      

        return model
    
    def _make_features(self, waveform, mel_bins = 128, target_length=1024, sr = 16000):
        # waveform, sr = torchaudio.load(wav_name)

        fbank = torchaudio.compliance.kaldi.fbank(
            waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
            window_type='hanning', num_mel_bins=mel_bins, dither=0.0,
            frame_shift=10)

        n_frames = fbank.shape[0]

        p = target_length - n_frames
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]

        fbank = (fbank - (-4.2677393)) / (4.5689974 * 2)
        return fbank


    def make_inference_with_path(self, path_to_audio):
        """Method to make a prediction using a file path

        path_to_audio -- Path to the target audio file
        """
        
        # Load wavefrom

        audio, _ = librosa.load(path_to_audio, sr = 16000, mono = True)
        audio = torch.from_numpy(audio).unsqueeze(0)

        
        feats = self._make_features(audio)

        input_tdim = feats.shape[0]
        
        feats_data = feats.expand(1, input_tdim, 128).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            output = self.model.forward(feats_data)
            output = torch.sigmoid(output)

        probs = output.data.cpu().numpy()[0]
        # Get Index and Class name of prediction
        max_idx = np.argmax(probs)

        label = self.ontology[max_idx]
        best_score = probs[max_idx]
        predicted_class_idx = max_idx

        return {"probs": probs, "predicted_class_idx": predicted_class_idx, "label": label, "best_score": best_score}

    def make_inference_with_waveform(self, waveform: np.ndarray):
        """Method to make a prediction using a waveform

        waveform -- The audio waveform
        """

        waveform = waveform.astype('float32')


        waveform = torch.from_numpy(waveform).unsqueeze(0)

        feats = self._make_features(waveform, mel_bins=128)

        input_tdim = feats.shape[0]
        
        feats_data = feats.expand(1, input_tdim, 128).to(self.device)

        # Make prediction
        with torch.no_grad():
            output = self.model.forward(feats_data)
            output = torch.sigmoid(output)

        probs = output.data.cpu().numpy()[0]

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

