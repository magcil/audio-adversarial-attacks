import os
import sys

sys.path.insert(0, os.path.dirname((os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from utils import utils


def get_obj_function(obj_func_name):

    if obj_func_name == "simple_minimization" or obj_func_name is None:
        return simple_minimization

    elif obj_func_name == "auditory_masking_minimization":
        return auditory_masking_minimization

    elif obj_func_name == "simple_minimization_targeted":
        return simple_minimization_targeted
    else:
        return L2_minimization


def get_fitness(obj_func_name, **kwargs):

    if obj_func_name == "simple_minimization" or obj_func_name is None:
        return simple_minimization(kwargs["starting_idx"], kwargs["probs"])

    elif obj_func_name == "auditory_masking_minimization":
        return auditory_masking_minimization(kwargs["raw_audio"], kwargs["noise"], kwargs["starting_idx"], kwargs["λ"])

    elif obj_func_name == "simple_minimization_targeted":
        return simple_minimization_targeted(kwargs["starting_idx"], kwargs["target_class_index"], kwargs["probs"])
    else:
        return L2_minimization(kwargs["raw_audio"], kwargs["raw_audio"] + kwargs["noise"], kwargs["starting_idx"],
                               kwargs["probs"], kwargs["λ"])


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


def auditory_masking_minimization(raw_audio, noise, starting_idx, probs, λ):
    """
        Use of regularization term based on auditory masking.

        Parameters:
            λ (int): Weight of the regularization term
    """

    Q0 = simple_minimization(starting_idx, probs)

    Q1 = noise / (np.abs(raw_audio) + 0.0000001)
    fitness = Q0 + λ * sum(Q1)

    return fitness


def L2_minimization(raw_audio, perturbed_audio, starting_idx, probs, λ):
    """
        Use of L2 distance regularization term.

        Parameters:
            λ (int): Weight of the regularization term.
    """

    Q0 = simple_minimization(starting_idx, probs)

    fitness = Q0 + λ * utils.calculate_euclidean_distance(raw_audio, perturbed_audio)

    return fitness


# ----- Targeted -----
def simple_minimization_targeted(starting_idx, target_class_idx, probs):

    starting_class_score = np.max(probs[starting_idx])
    target_classes_scores = np.max(probs[target_class_idx])

    fitness = starting_class_score - target_classes_scores
    return fitness
