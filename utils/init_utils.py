import os
import sys
from typing import Optional, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from models.beats.beats_model import BEATs_Model
from models.PaSST.passt_model import Passt_Model
# from models.AST.AST_model import AST_Model
from algorithms.differential_evolution.Differential_Evolution import DifferentialEvolutionAttacker
from algorithms.pso.pso_attacker import PSO_Attacker


def get_model(model_str: str,
              model_pt_file: Optional[str] = None,
              hypercategory_mapping: Optional[Dict] = None):
    if model_str == 'beats':
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = BEATs_Model(path_to_checkpoint=model_pt_file,
                            device=device,
                            hypercategory_mapping=hypercategory_mapping)
    elif model_str == "passt":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = Passt_Model(device=device,
                            hypercategory_mapping=hypercategory_mapping)

    # else:
    #     device = "cuda" if torch.cuda.is_available() else "cpu"
    #     model = AST_Model(path_to_checkpoint=model_pt_file, device=device, hypercategory_mapping=hypercategory_mapping)
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
                                                         verbosity=verbosity,
                                                         objective_function=objective_function,
                                                         target_class=target_class,
                                                         hypercategory_target=hypercategory_target,
                                                         **hyperparameters)
    elif algorithm == 'pso':
        ATTACK_ALGORITHM = PSO_Attacker(model=model,
                                        verbosity=verbosity,
                                        objective_function=objective_function,
                                        target_class=target_class,
                                        hypercategory_target=hypercategory_target,
                                        **hyperparameters)
        
    else:
        print("Enter Valid Algorithm")

    return ATTACK_ALGORITHM
