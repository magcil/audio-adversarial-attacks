import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import random

import numpy as np
from utils import utils

from objective_functions import objective_functions


class Particle:

    def __init__(self,
                 model,
                 starting_class_index,
                 starting_class_label,
                 particle_position,
                 velocity,
                 raw_audio,
                 verbosity=True,
                 objective_function=None,
                 SNR_norm=None):
        """Instantiate Particle object

        model -- The model used for inference
        starting_class_index -- The model index of the starting class
        particle_position -- The waveform/position of the particle
        velocity -- The PSO velocity of the particle
        """

        self.model = model
        self.raw_audio = raw_audio
        self.objective_function = objective_function
        self.starting_class_index = starting_class_index
        self.starting_class_label = starting_class_label
        self.verbosity = verbosity

        self.velocity = velocity
        self.position = particle_position
        self.best_position = particle_position
        self.SNR_norm = SNR_norm

        self.fitness_results = self.calculate_fitness()
        self.best_fitness = self.fitness_results["fitness"]

    def calculate_fitness(self):
        """Calculate fitness of the particle based on position"""

        
        pred_audio = utils.add_normalized_noise(self.raw_audio, self.position - self.raw_audio,
                                                    self.SNR_norm)["adversary"]

        #---- Make inference ----#
        result = self.model.make_inference_with_waveform(pred_audio)
        scores, predicted_class_idx, label = result["probs"], result["predicted_class_idx"], result["label"]
        
        label = str(self.model.hypercategory_mapping[predicted_class_idx])

        self.starting_class_index = np.where(self.model.hypercategory_mapping == self.starting_class_label)[0]


        if (self.starting_class_label != label):
            if self.verbosity:
                print(f'Attack Succeded from {self.starting_class_label} to {label}')
            return {"fitness": float('-inf'), "inferred_class": label}

        adv_dict = utils.add_normalized_noise(self.raw_audio, self.position - self.raw_audio, self.SNR_norm)
        objective_function_kwargs = {
            "starting_idx": self.starting_class_index,
            "probs": scores,
            "raw_audio": adv_dict["clean_audio"],
            "noise": adv_dict["noise"]
        }
        fitness = objective_functions.get_fitness(self.objective_function, **objective_function_kwargs)

        return {"fitness": fitness, "inferred_class": label}

    def update_velocity_and_position(self, inertia_w, memory_w, information_w, sbp):
        """
        Calculate next velocity and position of particle
      
            inertia_w -- The PSO inertia weight
            memory_w -- The PSO memory weight
            memory_r -- The PSO memory random multiplier
            information_w -- The PSO information weight
            sbp -- The best position within the Swarm
        """

        inertia = inertia_w * self.velocity
        memory_r = random.uniform(0, 1)
        information_r = random.uniform(0, 1)

        memory = memory_w * memory_r * (self.best_position - self.position)
        information = information_w * information_r * (sbp - self.position)

        self.velocity = inertia + memory + information
        self.position = self.position + self.velocity
