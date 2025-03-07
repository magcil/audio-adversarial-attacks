import os
import sys
from typing import Optional, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from models.beats.beats_model import BEATs_Model
from models.PaSST.passt_model import Passt_Model
from algorithms.differential_evolution.Differential_Evolution import DifferentialEvolutionAttacker
from algorithms.pso.pso_attacker import PSO_Attacker
from algorithms import default_pso_hyperparams, default_de_hyperparams


def get_model(model_str: str,
              model_pt_file: Optional[str] = None,
              hypercategory_mapping: Optional[Dict] = None,
              device: str = "cpu"):
    if model_str == 'beats':
        model = BEATs_Model(path_to_checkpoint=model_pt_file,
                            device=device,
                            hypercategory_mapping=hypercategory_mapping)
    elif model_str == "passt":
        model = Passt_Model(device=device, hypercategory_mapping=hypercategory_mapping)
    return model


def init_algorithm(algorithm: str,
                   model,
                   verbosity,
                   SNR_norm,
                   hyperparameters=None,
                   objective_function=None,
                   target_class=None,
                   hypercategory_target=None):

    if algorithm == 'de':
        if hyperparameters is None:
            hyperparameters = default_de_hyperparams
        ATTACK_ALGORITHM = DifferentialEvolutionAttacker(model=model,
                                                         verbosity=verbosity,
                                                         objective_function=objective_function,
                                                         SNR_norm=SNR_norm,
                                                         **hyperparameters)
    elif algorithm == 'pso':
        if hyperparameters is None:
            hyperparameters = default_pso_hyperparams
        ATTACK_ALGORITHM = PSO_Attacker(model=model,
                                        verbosity=verbosity,
                                        objective_function=objective_function,
                                        SNR_norm=SNR_norm,
                                        **hyperparameters)

    else:
        print("Enter Valid Algorithm")
        ATTACK_ALGORITHM = None

    return ATTACK_ALGORITHM
