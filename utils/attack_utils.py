import os
import sys
from typing import Optional, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from models.beats.beats_model import BEATs_Model
from algorithms.differential_evolution.Differential_Evolution import DifferentialEvolutionAttacker
from algorithms.pso.pso_attacker import PSO_Attacker


def get_model(model_str: str, model_pt_file: Optional[str] = None, hypercategory_mapping: Optional[Dict] = None):
    if model_str == 'beats':
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = BEATs_Model(path_to_checkpoint=model_pt_file,
                            device=device,
                            hypercategory_mapping=hypercategory_mapping)
    return model


def init_algorithm(algorithm: str,
                   model,
                   hyperparameters,
                   verbosity,
                   objective_function=None,
                   target_class=None,
                   hypercategory_target=None):
    if algorithm == 'de':
        ATTACK_ALGORITHM = DifferentialEvolutionAttacker(model=model,
                                                         de_hyperparameters=hyperparameters,
                                                         verbosity=verbosity,
                                                         objective_function=objective_function,
                                                         target_class=target_class,
                                                         hypercategory_target=hypercategory_target)
    elif algorithm == 'pso':
        ATTACK_ALGORITHM = PSO_Attacker(model=model,
                                        pso_hyperparameters=hyperparameters,
                                        verbosity=verbosity,
                                        objective_function=objective_function,
                                        target_class=target_class,
                                        hypercategory_target=hypercategory_target)

    return ATTACK_ALGORITHM


def get_model_pred(model, waveform):
    _, idx, inferred_class_name, confidence = model.make_inference_with_waveform(waveform)

    return inferred_class_name, idx, confidence


def is_true_pred(filename, true_labels, inferred_class_name) -> bool:
    return True if inferred_class_name in true_labels[filename] else False
