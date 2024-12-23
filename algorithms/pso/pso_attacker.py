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
                 initial_particles,
                 max_iters,
                 max_inertia_w,
                 min_inertia_w,
                 memory_w,
                 information_w,
                 perturbation_ratio,
                 verbosity=True,
                 objective_function=None,
                 target_class=None,
                 hypercategory_target=None,
                 SNR_norm=None):
        
        """Instantiate PSO attacker

           model (Object) -- A pretrained model used for inference.
           initial_particles (int) -- THe number of particels used in the Swarm (Population Size).
           max_iters (int) -- Maximum number of iterations.
           max_inertia_w (float) -- Maximum inertia weight, which controls the influence of particle's velocity.
           min_inertia_w (float) -- Minimum inertia weight.
           memory_w (float) -- Weight of particle's own memory (Personal Best).
           information_w (float) -- Weight for the global best information.
           
        """

        # Initialize PSO Hyperparameters
        self.pso_hyperparameters = {
            "initial_particles": initial_particles,
            "max_iters": max_iters,
            "max_inertia_w": max_inertia_w,
            "min_inertia_w": min_inertia_w,
            "memory_w": memory_w,
            "information_w": information_w,
            "perturbation_ratio": perturbation_ratio
        }

        self.target_class = target_class
        self.hypercategory_target = hypercategory_target
        self.model = model
        self.objective_function = objective_function
        self.swarm = None
        self.SNR_norm = SNR_norm

        # ---- Unfold PSO Hyperparameters ----
        self.verbosity = verbosity

    def initialization(self, starting_class_index, starting_class_label):
        """Stage 1: Initialize PSO"""

        # print('STAGE 1: PSO Attack Initialization.')
        if self.verbosity:
            print("\033[91m STAGE 1: PSO Attack Initialization. \033[0m")

        #---- Initialize Swarm ----#
        self.swarm = Swarm(self.model,
                           self.pso_hyperparameters['initial_particles'],
                           self.clean_audio,
                           self.pso_hyperparameters['perturbation_ratio'],
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

    def optimization(self):
        """Stage 2: Optimization"""

        # print('STAGE 2: Optimisation.')
        if self.verbosity:
            print("\033[91m STAGE 2: Optimisation. \033[0m")

        #---- Start iterations ----#
        for i in range(1, self.pso_hyperparameters["max_iters"] + 1):

            # Linearly decrease inertia w
            inertia_w = self.pso_hyperparameters["max_inertia_w"] - i * (
                self.pso_hyperparameters["max_inertia_w"] -
                self.pso_hyperparameters["min_inertia_w"]) / self.pso_hyperparameters["max_iters"]

            if self.verbosity:
                print(f'----------- Iteration: {i} -----------')

            particles_counter = 0  # Initialise particles counter.
            #---- Optimize Particles ----#
            for p in self.swarm.particles:

                if self.verbosity:
                    print(f' ----------- Particle: {particles_counter} -----------')
                particles_counter += 1

                p.update_velocity_and_position(inertia_w, self.pso_hyperparameters["memory_w"],
                                               self.pso_hyperparameters["information_w"], self.swarm.sbp)
                fitness_results = p.calculate_fitness()
                self.queries += 1

                # Termination if attack succeded
                if (fitness_results["fitness"] == float('-inf')):
                    self.swarm.sbf = float('-inf')
                    self.swarm.sbp = p.position
                    return {
                        "noise":
                        self.swarm.sbp - self.clean_audio,
                        "adversary":
                        self.swarm.sbp if self.SNR_norm is None else utils.add_normalized_noise(
                            self.clean_audio, self.swarm.sbp - self.clean_audio, self.SNR_norm),
                        "raw audio":
                        self.clean_audio,
                        "iterations":
                        i,
                        "success":
                        True,
                        "queries":
                        self.queries,
                        "inferred_class":
                        fitness_results["inferred_class"]
                    }

                # Update partice BF and BP, if better found
                if (fitness_results["fitness"] < p.best_fitness):
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

        return {
            "noise":
            self.swarm.sbp - self.clean_audio,
            "adversary":
            self.swarm.sbp if self.SNR_norm is None else utils.add_normalized_noise(self.clean_audio, self.swarm.sbp -
                                                                                    self.clean_audio, self.SNR_norm),
            "raw audio":
            self.clean_audio,
            "iterations":
            i,
            "success":
            False,
            "queries":
            self.queries,
            "inferred_class":
            fitness_results["inferred_class"]
        }

    def generate_adversarial_example(self, source_audio):
        """Perform attack and return results"""

        # Parse source audio. Either wav file or numpy array
        if (os.path.isfile(source_audio)):
            self.clean_audio, _ = librosa.load(source_audio, sr=16000, mono=True)
        else:
            self.clean_audio = source_audio

        # Make inference to get index/label
        prediction_results = self.model.make_inference_with_waveform(self.clean_audio)
        starting_class_index, starting_class_label = prediction_results["predicted_class_idx"], prediction_results["label"]

        if len(self.model.hypercategory_mapping):
            starting_class_label = self.model.hypercategory_mapping[starting_class_index]

        self.initialization(starting_class_index=starting_class_index, starting_class_label=starting_class_label)

        results = self.optimization()

        # Make inference with perturbed waveform
        results["queries"] += 1

        # Get probs and final confidence of adversarial sample.
        prediction_results = self.model.make_inference_with_waveform(results["adversary"])

        probs, final_confidence = prediction_results["probs"], prediction_results["best_score"]

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

        # Append starting class label to results
        results['starting_class'] = starting_class_label

        return results
