import os
import sys

sys.path.insert(0, os.path.dirname((os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

def get_fitness(obj_func_name, **kwargs):

    if obj_func_name == "simple_minimization" or obj_func_name is None:
        return simple_minimization(kwargs["starting_idx"], kwargs["probs"])
    else:
        raise ValueError(f"{obj_func_name} not implemented")

#  ----- Untargeted -----
def simple_minimization(starting_idx, probs):
    """
        Simple minimization objective function.
        Defined as confidence of y_true minus the maximum confidence of all other classes except the starting one.
    """
    starting_class_score = np.max(probs[starting_idx])
    others_classes_scores = np.max(np.delete(probs, starting_idx))

    fitness = starting_class_score - others_classes_scores
    return fitness