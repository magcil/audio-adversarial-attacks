import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from .particle import Particle

from utils import utils


class Swarm:

    def __init__(self,
                 model,
                 initial_particles,
                 clean_audio,
                 perturbation_ratio,
                 starting_class_index,
                 starting_class_label,
                 verbosity=True,
                 objective_function=None,
                 target_class=None,
                 hypercategory_target=None,
                 SNR_norm=None):
        """Instantiate Swarm object
        
        model -- The model used for inference
        initial_particles -- The number of initial Swarm particles
        clean_audio -- Clean audio file
        perturbation_ratio -- The ratio of added perturbation
        starting_class_index -- The model index of the starting class
        """

        self.target_class = target_class
        self.hypercategory_target = hypercategory_target
        self.objective_function = objective_function
        self.sbf = float('inf')
        self.sbp = clean_audio  # SBP is initially equal to the raw audio file
        self.verbosity = verbosity
        self.perturbation_ratio = perturbation_ratio
        self.SNR_norm = SNR_norm
        self.particles = self.generate_initial_particles(model, initial_particles, starting_class_index,
                                                         starting_class_label, clean_audio)

    def generate_initial_particles(self, model, initial_particles, starting_class_index, starting_class_label,
                                   clean_audio):
        """Generate particles during swarm initialization
        
        Attributes:
        initial_particles -- Number of initial particles
        starting_class_index -- Index of starting class
        target_class_index -- Index of target class
        clean_audio -- Path of taregt wav file
        """

        particles = []
        target_waveform = clean_audio
        if self.verbosity:
            print("------ Particle's Initialization ------")
        for p in range(0, initial_particles):

            # TODO: Experiment with different initial velocity?
            initial_velocity = utils.generate_bounded_white_noise(target_waveform, self.perturbation_ratio)
            particle_position = target_waveform + initial_velocity
            if self.verbosity:
                print(f"------ Particle {p} ------")

            #---- Initialize particles ----#
            particle = Particle(model,
                                starting_class_index,
                                starting_class_label,
                                particle_position,
                                initial_velocity,
                                raw_audio=clean_audio,
                                verbosity=self.verbosity,
                                objective_function=self.objective_function,
                                target_class=self.target_class,
                                hypercategory_target=self.hypercategory_target,
                                SNR_norm=self.SNR_norm)
            particles.append(particle)

            # TODO: Termination condition here??
            # if(particle.best_fitness == float('-inf')):
            #     self.sbf = particle.best_fitness
            #     self.sbp = particle.best_position
            #     return

            #---- Update SBF and SBP, if better found ----#
            if particle.best_fitness < self.sbf:
                self.sbf = particle.best_fitness
                self.sbp = particle.best_position

        return particles
