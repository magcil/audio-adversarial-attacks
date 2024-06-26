import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from utils import generation_utils
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
                 hypercategory_target=None):
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
        self.sbf = float('-inf')
        self.sbf_history = [-1]
        self.sbp = clean_audio  # SBF is initially equal to the raw audio file
        self.verbosity = verbosity
        self.perturbation_ratio = perturbation_ratio
        self.particles = self.generate_initial_particles(model, initial_particles, starting_class_index,
                                                         starting_class_label, clean_audio)
        self.concurrent_stable_sbf_count = 0

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

            # TODO: Change initial velocity???
            initial_velocity = utils.generate_bounded_white_noise(target_waveform, self.perturbation_ratio)
            particle_position = np.clip(target_waveform + initial_velocity, -1.0, 1.0)
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
                                hypercategory_target=self.hypercategory_target)
            particles.append(particle)

            # TODO: Termination condition here??
            # if(particle.best_fitness == float('inf')):
            #     self.sbf = particle.best_fitness
            #     self.sbp = particle.best_position
            #     return

            #---- Update SBF and SBP, if better found ----#
            if particle.best_fitness > self.sbf:
                self.sbf = particle.best_fitness
                self.sbp = particle.best_position

        self.sbf_history.append(self.sbf)
        return particles

    def generate_additional_particles(self, num_of_temporary_particles, target_wav, perturbed_file):
        """Perform Temporary Particle Generation
        
        num_of_temporary_particles -- Number of particles to be generated
        target_wav -- Waveform of raw audio file
        perturbed_file -- 
        """

        for i in range(num_of_temporary_particles):
            particle_1, particle_2 = generation_utils.selection(0, len(self.particles), 2)
            # There are 2 childs generated from crossover. We choose 1.
            # TODO: Maybe chose random child1 or child2 or both?
            child_1, child_2 = generation_utils.crossover(self.particles[particle_1].particle_position,
                                                          self.particles[particle_2].particle_position)
            particle_position = generation_utils.mutate(child_1, 0.7, 0.3, target_wav)
            particle_position = np.clip(particle_position, -1.0, 1.0)  # Clip positions to stay into bounds
            velocity = utils.generate_bounded_white_noise(target_wav, self.perturbation_ratio)

            particle = Particle(self.model, perturbed_file, self.starting_class, self.target_class, particle_position,
                                velocity)
            self.particles.append(particle)

            self.generated_additional_particle = True

        return True

    def check_stable_sbf_stagnation(self):
        """Check if SBF is stable for more than 3 iterations"""

        previous_sbf = self.sbf_history[-2]
        current_sbf = self.sbf_history[-1]

        if self.verbosity:
            print(f'Previous SBF: {previous_sbf}, Current SBF: {current_sbf}')

        if previous_sbf == current_sbf:
            self.concurrent_stable_sbf_count += 1

            if (self.concurrent_stable_sbf_count == 3):
                if self.verbosity:
                    print("Stagnated SBF")
                return True
        else:
            if (self.concurrent_stable_sbf_count != 0):
                self.concurrent_stable_sbf_count -= 1

        return False
