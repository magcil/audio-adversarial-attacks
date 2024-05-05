import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import librosa
from .swarm import Swarm
from utils import utils
import numpy as np


class PSO_Attacker:

    def __init__(self,
                 model,
                 pso_hyperparameters,
                 verbosity=True,
                 objective_function=None,
                 target_class=None,
                 hypercategory_target=None):
        """Instantiate PSO attacker
        
        model -- The model used for inference
        clean_audio -- Clean audio file
        initial_particles -- The number of initial Swarm particles
        enable_particle_generation -- Switch for enabling particle generation
        additional_particles -- The number of generated particles
        max_iters -- Maximum number of iterations
        max_inertia_w -- Maximum PSO inertia weight
        min_inertia_w -- Minimum PSO inertia weight
        memory_w -- PSO memory weight
        information_w -- PSO iformation weight
        perturbation_ratio -- Ratio of added perturbation
        starting_class_index -- The model index of the starting class
        enabled_early_stopping -- Switch for enabling early stopping
        """

        self.target_class = target_class
        self.hypercategory_target = hypercategory_target
        self.model = model
        self.objective_function = objective_function
        self.swarm = None

        # ---- Unfold PSO Hyperparameters ----
        self.initial_particles = pso_hyperparameters["initial_particles"]
        self.added_particles = 0
        self.enable_particle_generation = pso_hyperparameters["enable_particle_generation"]
        self.enabled_early_stop = pso_hyperparameters["enabled_early_stopping"]
        self.additional_particles = pso_hyperparameters["additional_particles"]

        self.max_iters = pso_hyperparameters["max_iters"]
        self.max_inertia_w = pso_hyperparameters["max_inertia_w"]
        self.min_inertia_w = pso_hyperparameters["min_inertia_w"]
        self.inertia_w = None
        self.memory_w = pso_hyperparameters["memory_w"]
        self.information_w = pso_hyperparameters["information_w"]
        self.perturbation_ratio = pso_hyperparameters["perturbation_ratio"]
        self.verbosity = verbosity

    def initialization(self, starting_class_index, starting_class_label):
        """Stage 1: Initialize PSO"""

        # print('STAGE 1: PSO Attack Initialization.')
        if self.verbosity:
            print("\033[91m STAGE 1: PSO Attack Initialization. \033[0m")

        #---- Initialize Swarm ----#
        self.swarm = Swarm(self.model,
                           self.initial_particles,
                           self.clean_audio,
                           self.perturbation_ratio,
                           starting_class_index,
                           starting_class_label,
                           verbosity=self.verbosity,
                           objective_function=self.objective_function,
                           target_class=self.target_class,
                           hypercategory_target=self.hypercategory_target)

        #---- Keep count of model queries ----#
        self.queries = len(self.swarm.particles)

        if self.verbosity:
            print(f'SBF after initialization: {self.swarm.sbf}')

        return True

    def optimization(self):
        """Stage 2: Optimization"""

        # print('STAGE 2: Optimisation.')
        if self.verbosity:
            print("\033[91m STAGE 2: Optimisation. \033[0m")

        #---- Start iterations ----#
        for i in range(1, self.max_iters + 1):
            # Linearly decrease inertia w
            self.inertia_w = self.max_inertia_w - i * ((self.max_inertia_w - 0.0) / self.max_iters)
            if self.verbosity:
                print(f'----------- Iteration: {i} -----------')

            particles_counter = 0  # Initialise particles counter.
            #---- Optimize Particles ----#
            for p in self.swarm.particles:

                if self.verbosity:
                    print(f' ----------- Particle: {particles_counter} -----------')
                particles_counter += 1

                p.update_velocity_and_position(self.inertia_w, self.memory_w, self.information_w, self.swarm.sbp)
                fitness_results = p.calculate_fitness()
                self.queries += 1

                # Termination if attack succeded
                if (fitness_results["fitness"] == float('inf')):
                    self.swarm.sbf = float('inf')
                    self.swarm.sbp = p.position
                    return {
                        "noise": self.swarm.sbp - self.clean_audio,
                        "adversary": self.swarm.sbp,
                        "raw audio": self.clean_audio,
                        "iterations": i,
                        "success": True,
                        "queries": self.queries,
                        "inferred_class": fitness_results["inferred_class"]
                    }

                # Update partice BF and BP, if better found
                if (fitness_results["fitness"] > p.best_fitness):
                    if self.verbosity:
                        print("Better Personal Best found!")
                        print(f"Previous Best: {p.best_fitness}, Current Best {fitness_results['fitness']}")
                    p.best_fitness = fitness_results["fitness"]
                    p.best_position = p.position

                # Update SBF and SBP, if better found
                if (p.best_fitness > self.swarm.sbf):
                    if self.verbosity:
                        print("Better Global Best found!")
                        print(f"Previous Best: {self.swarm.sbf}, Current Best {p.best_fitness}")
                    self.swarm.sbf = p.best_fitness
                    self.swarm.sbp = p.best_position

            #---- Check for Stagnation ----#
            stagnated = self.swarm.check_stable_sbf_stagnation()
            if (stagnated):
                # Early stop, if enabled
                if (self.enabled_early_stop):
                    if self.verbosity:
                        print("Early Stopping")
                    break
                # Proceed to particles generation, if enabled
                if (self.enable_particle_generation):
                    if self.verbosity:
                        print('Proceeding to Temporary Particles Generation stage')

                    self.swarm.generate_additional_particles(self.additional_particles, self.target_wav,
                                                             self.perturbed_file)
                    self.added_particles += self.additional_particles

        return {
            "noise": self.swarm.sbp - self.clean_audio,
            "adversary": self.swarm.sbp,
            "raw audio": self.clean_audio,
            "iterations": i,
            "success": False,
            "queries": self.queries,
            "inferred_class": fitness_results["inferred_class"]
        }

    def generate_adversarial_example(self, source_audio):
        """Perform attack and return results"""

        # Parse source audio. Either wav file or numpy array
        if (os.path.isfile(source_audio)):
            self.clean_audio, _ = librosa.load(source_audio, sr=16000, mono=True)
        else:
            self.clean_audio = source_audio

        # Make inference to get index/label
        _, starting_class_index, starting_class_label, _ = self.model.make_inference_with_waveform(self.clean_audio)

        if len(self.model.hypercategory_mapping):
            starting_class_label = self.model.hypercategory_mapping[starting_class_index]

        if not self.initialization(starting_class_index=starting_class_index,
                                   starting_class_label=starting_class_label):
            return False

        results = self.optimization()

        # Make inference with perturbed waveform
        results["queries"] += 1
        probs, _, _, final_confidence = self.model.make_inference_with_waveform(results["adversary"])

        # Get final confidence of starting class
        if len(self.model.hypercategory_mapping):

            #Get indexes of all occurancies of the hyperclass
            hypercategory_idxs = np.where(self.model.hypercategory_mapping == starting_class_label)[0]

            # Get maximum probability
            max_prob = max(probs[hypercategory_idxs])
        else:
            max_prob = probs[starting_class_index]

        results["Final Starting Class Confidence"] = max_prob
        results["Final Confidence"] = final_confidence

        return results
