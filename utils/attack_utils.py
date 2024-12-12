# Utility functions for attacking experiments
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from typing import List, Dict, Tuple


def filter_on_correct_predictions(model, wav_files: List[os.PathLike], true_labels: Dict[str,
                                                                                         str]) -> List[os.PathLike]:
    """Keep the correct predictions by the model
    
    Args:
        model: A model that implements the 'make_inference_with_path_method'
        wav_files: List containing the paths of the wav files
        true_labels: Correspondence wav_file -> class_name
    
    Returns:
        filtered_wavs: Wav files correctly classified by the model.
    """
    filtered_wavs = []
    for wav_file in wav_files:
        pred_results = model.make_inference_with_path(wav_file)
        # If prediction is correct then keep
        if pred_results['label'] == true_labels[os.path.basename(wav_file)]:
            filtered_wavs.append(wav_files)

    return filtered_wavs


def perform_single_attack(ATTACK_ALGORITHM, wav_file) -> Dict:
    """Perform an attack on a single wav file
    
    Args:
        ATTACK_ALGORITHM: An instance of an attack algorithm
        wav_file: Abs Path to of wav file
    
    Returns:
        Dictionary of attack results returned by ATTACK_ALGORITHM
    """
    attack_results = ATTACK_ALGORITHM.generate_adversarial_example(wav_file)

    return attack_results
