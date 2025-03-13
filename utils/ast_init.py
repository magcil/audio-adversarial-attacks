import os
import sys
from typing import Optional, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from models.AST.AST_model import AST_Model
from algorithms.differential_evolution.Differential_Evolution import DifferentialEvolutionAttacker
from algorithms.pso.pso_attacker import PSO_Attacker


def get_model(model_str: str,
              model_pt_file: Optional[str] = None,
              hypercategory_mapping: Optional[Dict] = None,
              device: str = "cpu"):


    model = AST_Model(path_to_checkpoint=model_pt_file, device=device, hypercategory_mapping=hypercategory_mapping)
    return model


def init_algorithm(algorithm: str,
                   model,
                   hyperparameters,
                   verbosity,
                   SNR_norm,
                   objective_function=None):

    if algorithm == 'de':
        ATTACK_ALGORITHM = DifferentialEvolutionAttacker(model=model,
                                                         verbosity=verbosity,
                                                         objective_function=objective_function,
                                                         SNR_norm=SNR_norm,
                                                         **hyperparameters)
    elif algorithm == 'pso':
        ATTACK_ALGORITHM = PSO_Attacker(model=model,
                                        verbosity=verbosity,
                                        objective_function=objective_function,
                                        SNR_norm=SNR_norm,
                                        **hyperparameters)
        
    else:
        print("Enter Valid Algorithm")

    return ATTACK_ALGORITHM
