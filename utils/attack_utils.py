# Utility functions for attacking experiments
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from typing import List, Dict, Tuple

from sklearn.metrics import classification_report
import numpy as np
import json

def filter_on_correct_predictions(model, wav_files: List[os.PathLike],
                                  true_labels: Dict[str, str], hypercategory_mapping: List[os.PathLike]) -> Tuple[List[os.PathLike], str]:
    """Keep the correct predictions by the model
    
    Args:
        model: A model that implements the 'make_inference_with_path_method'
        wav_files: List containing the paths of the wav files
        true_labels: Correspondence wav_file -> class_name
    
    Returns:
        filtered_wavs: Wav files correctly classified by the model.
    """
    filtered_wavs = []
    y_true, y_pred = [], []

    with open(hypercategory_mapping, 'r') as f:
        hypercategory_dict = json.load(f)

    for wav_file in wav_files:
        pred_results = model.make_inference_with_path(wav_file)
        
        # print(hypercategory_dict[true_labels[os.path.basename(wav_file)[:-4]]])
        # y_true.append(hypercategory_dict[true_labels[os.path.basename(wav_file)[:-4]]])
        # y_pred.append(hypercategory_dict[pred_results['label']])

        y_true.append(true_labels[os.path.basename(wav_file)[:-4]])
        y_pred.append(hypercategory_dict[pred_results['label']])


        # # If prediction is correct, then keep
        # if hypercategory_dict[pred_results['label']] == hypercategory_dict[true_labels[os.path.basename(wav_file)[:-4]][0]]:
        #     filtered_wavs.append(wav_file)

        # If prediction is correct, then keep
        if hypercategory_dict[pred_results['label']] == true_labels[os.path.basename(wav_file)[:-4]]:
            filtered_wavs.append(wav_file)


    # unique_names = set(item for sublist in hypercategory_dict.values() for item in sublist)
    # unique_names = np.array(list(unique_names))
    unique_names = np.array(list(set(hypercategory_dict.values())))
    
    return {
        "filtered_wavs": filtered_wavs,
        "classification_report": classification_report(y_true=y_true,
                                                       y_pred=y_pred,
                                                       labels= unique_names)

    }


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
