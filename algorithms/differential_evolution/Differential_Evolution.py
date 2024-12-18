import os
import sys

project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_path)

from numpy.random import rand, choice
from numpy import clip, argmin, min
import librosa
import numpy as np
from numpy import asarray
from utils import utils
from objective_functions import objective_functions


class DifferentialEvolutionAttacker:

    def __init__(self,
                 model,
                 pop_size,
                 iter,
                 F,
                 cr,
                 perturbation_ratio,
                 SNR_norm=None,
                 verbosity=True,
                 objective_function=None,
                 target_class=None,
                 hypercategory_target=None):
        """Instantiate DE attacker
        
        model -- The model used for inference
        clean_audio -- Clean audio file
        starting_class_index -- Index of starting class.
        starting_class_label -- Class label of input audio.
        bounds -- L infinity boundaries.
        """

        self.model = model
        self.verbosity = verbosity
        self.target_class = target_class
        self.hypercategory_target = hypercategory_target
        self.objective_function = objective_function

        self.de_hyperparameters = {
            "pop_size": pop_size,
            "iter": iter,
            "F": F,
            "cr": cr,
            "perturbation_ratio": perturbation_ratio
        }
        self.SNR_norm = SNR_norm

        # Get the indexes of targeted hypercategory
        if self.target_class and self.hypercategory_target:
            self.target_class_index = np.where(self.model.hypercategory_mapping == self.target_class)[0]
        
        # Get the index of targeted label
        elif self.target_class and not self.hypercategory_target:
            for k, v in self.model.ontology.items():
                if v ==self.target_class:
                    self.target_class_index = k
        else:
            self.target_class_index = None

    def obj(self, noise, starting_class_index, starting_class_label):

        if self.SNR_norm is not None:
            clipped_audio = utils.add_normalized_noise(self.clean_audio, noise, self.SNR_norm)

        else:
            clipped_audio = np.clip(self.clean_audio + noise, -1.0, 1.0)

        probs, predicted_class_idx, label, _ = self.model.make_inference_with_waveform(clipped_audio)

        if len(self.model.hypercategory_mapping):
            if self.hypercategory_target:
                label = self.model.hypercategory_mapping[predicted_class_idx]
            starting_class_index = np.where(self.model.hypercategory_mapping == starting_class_label)[0]

        # Termination Criteria
        if self.target_class:
            if (label == self.target_class):
                if self.verbosity:
                    print(f'Attack Succeded from {starting_class_label} to {label}')
                return {"fitness": float('-inf'), "inferred_class": label}

        else:
            if (starting_class_label != label):
                if self.verbosity:
                    print(f'Attack Succeded from {starting_class_label} to {label}')
                return {"fitness": float('-inf'), "inferred_class": label}

        objective_function_kwargs = {
            "starting_idx": starting_class_index,
            "target_class_index": self.target_class_index,
            "probs": probs,
            "raw_audio": self.clean_audio,
            "noise": noise
        }

        fitness = objective_functions.get_fitness(self.objective_function, **objective_function_kwargs)
        if self.verbosity:
            print("Fitness:", fitness)

        return {"fitness": fitness, "inferred_class": label}

    def mutation(self, x):
        a, b, c = x
        return a + self.de_hyperparameters["F"] * (b - c)

    def check_bounds(self, mutated):
        mutated_bound = [clip(mutated[i], self.bounds[i, 0], self.bounds[i, 1]) for i in range(len(self.bounds))]
        return mutated_bound

    def crossover(self, mutated, target):
        p = rand(len(self.bounds))
        trial = [mutated[i] if p[i] < self.de_hyperparameters["cr"] else target[i] for i in range(len(self.bounds))]
        return np.array(trial)

    def optimization(self, starting_class_index: int, starting_class_label: str):
        if self.verbosity:
            print("----------- Attack Started -----------")
            print("----------- Initialise Population -----------")

        # pop = self.bounds[:, 0] + (rand(self.pop_size, len(self.bounds)) * (self.bounds[:, 1] - self.bounds[:, 0]))

        pop = []
        for _ in range(self.de_hyperparameters["pop_size"]):
            random_values = utils.generate_bounded_white_noise(self.clean_audio, self.de_hyperparameters["perturbation_ratio"])
            pop.append(random_values)

        pop = np.array(pop)

        all_fitness_results = [self.obj(ind, starting_class_index, starting_class_label) for ind in pop]

        obj_all = [x['fitness'] for x in all_fitness_results]
        labels_all = [x['inferred_class'] for x in all_fitness_results]
        self.queries += self.de_hyperparameters["pop_size"]

        best_vector = pop[argmin(obj_all)]
        best_obj = min(obj_all)
        prev_obj = best_obj

        if self.verbosity:
            print(f"Init Best Obj: {best_obj}")
        # Early Stopping if attack succeds during initialisation
        if (best_obj == float('-inf')):
            if self.verbosity:
                print("----------- Attack Succeded from Initialisation -----------")
            return {
                "noise":
                best_vector,
                "adversary":
                np.clip(self.clean_audio + best_vector, -1.0, 1.0)
                if self.SNR_norm is None else utils.add_normalized_noise(self.clean_audio, best_vector, self.SNR_norm),
                "raw audio":
                self.clean_audio,
                "iterations":
                0,
                "success":
                True,
                "queries":
                self.queries,
                "inferred_class":
                labels_all[argmin(obj_all)]
            }
        if self.verbosity:
            print("----------- Iterations Loop Started -----------")
        # Loop for iterations.
        for i in range(1, self.de_hyperparameters["iter"] + 1):
            print(f'----------- Iteration: {i} -----------')
            #  Loop for population.
            for j in range(self.de_hyperparameters["pop_size"]):
                if self.verbosity:
                    print(f' ----------- Candidate Solution: {j} -----------')
                candidates = [candidate for candidate in range(self.de_hyperparameters["pop_size"]) if candidate != j]
                a, b, c = pop[choice(candidates, 3, replace=False)]

                mutated = self.mutation([a, b, c])
                # mutated = self.check_bounds(mutated)
                trial = self.crossover(mutated, pop[j])

                fitness_results_target = self.obj(pop[j], starting_class_index, starting_class_label)
                obj_target = fitness_results_target["fitness"]

                fitness_results_trial = self.obj(trial, starting_class_index, starting_class_label)
                obj_trial = fitness_results_trial["fitness"]
                self.queries += 2

                # Early stop if the trial vector succeds
                if (obj_trial == float('-inf')):
                    if self.verbosity:
                        print("----------- Attack Succeded -----------")
                    return {
                        "noise":
                        trial,
                        "adversary":
                        np.clip(self.clean_audio + trial, -1.0, 1.0) if self.SNR_norm is None else
                        utils.add_normalized_noise(self.clean_audio, trial, self.SNR_norm),
                        "raw audio":
                        self.clean_audio,
                        "iterations":
                        i,
                        "success":
                        True,
                        "queries":
                        self.queries,
                        "inferred_class":
                        fitness_results_trial["inferred_class"]
                    }

                if obj_trial < obj_target:
                    if self.verbosity:
                        print("Better candidate found!")
                        print(f"Previous Obj: {obj_target}, Current Obj {obj_trial}")
                        print("Replace selected candidate with the trial vector")
                    pop[j] = trial
                    obj_all[j] = obj_trial

            best_obj = min(obj_all)

            if best_obj < prev_obj:
                if self.verbosity:
                    print(f"Better Global Fitness found! {round(best_obj, 5)}")
                    print("Replace best vector and vest fitness")
                best_vector = pop[argmin(obj_all)]
                prev_obj = best_obj

        return {
            "noise":
            best_vector,
            "adversary":
            np.clip(self.clean_audio + best_vector, -1.0, 1.0) if self.SNR_norm is None else utils.add_normalized_noise(
                self.clean_audio, best_vector, self.SNR_norm),
            "raw audio":
            self.clean_audio,
            "iterations":
            i,
            "success":
            False,
            "queries":
            self.queries,
            "inferred_class":
            labels_all[argmin(obj_all)]
        }

    def generate_adversarial_example(self, source_audio):

        # Parse source audio. Either wav file or numpy array
        if (os.path.isfile(source_audio)):
            self.clean_audio, _ = librosa.load(source_audio, sr=16000, mono=True)
        else:
            self.clean_audio = source_audio

        self.bounds = asarray([(-self.de_hyperparameters["rangeOfBounds"], self.de_hyperparameters["rangeOfBounds"])
                               for _ in range(len(self.clean_audio))])

        # Make inference to get index/label
        _, starting_class_index, starting_class_label, _ = self.model.make_inference_with_waveform(self.clean_audio)

        if len(self.model.hypercategory_mapping):
            starting_class_label = self.model.hypercategory_mapping[starting_class_index]

        # Initialize queries counter
        self.queries = 0

        results = self.optimization(starting_class_index=starting_class_index,
                                    starting_class_label=starting_class_label)

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

        # Append starting class label to results
        results['starting_class'] = starting_class_label

        return results
