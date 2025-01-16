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
        
        model (Object) -- A pretrained model used for inference.
        pop_size (int) -- The population size.
        iter (int) -- Number of iterations.
        F (float) -- Mutation Factor that scales the difference vector during mutation.
        cr (float) -- Crossover probability in the range [0,1]. Defines the likelihogg of crossover.
        
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
        else:
            self.target_class_index = None

    def obj(self, noise, starting_class_index, starting_class_label):

        clipped_audio = utils.add_normalized_noise(self.clean_audio, noise, self.SNR_norm)["adversary"]

        inference_results = self.model.make_inference_with_waveform(clipped_audio)
        probs, predicted_class_idx, label = inference_results["probs"], inference_results[
            "predicted_class_idx"], inference_results["label"]

        label = str(self.model.hypercategory_mapping[predicted_class_idx])
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

        adv_dict = utils.add_normalized_noise(self.clean_audio, noise, self.SNR_norm)
        objective_function_kwargs = {
            "starting_idx": starting_class_index,
            "target_class_index": self.target_class_index,
            "probs": probs,
            "raw_audio": adv_dict["clean_audio"],
            "noise": adv_dict["noise"]
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

        pop = []
        for _ in range(self.de_hyperparameters["pop_size"]):
            random_values = utils.generate_bounded_white_noise(self.clean_audio,
                                                               self.de_hyperparameters["perturbation_ratio"])
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
            adv_dict = utils.add_normalized_noise(self.clean_audio, best_vector, self.SNR_norm)

            if self.verbosity:
                print("----------- Attack Succeded from Initialisation -----------")
            return {
                "noise":
                adv_dict["noise"],
                "adversary":
                adv_dict["adversary"],
                "raw audio":
                adv_dict["clean_audio"],
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
            if self.verbosity:
                print(f'----------- Iteration: {i} -----------')
            #  Loop for population.
            for j in range(self.de_hyperparameters["pop_size"]):
                if self.verbosity:
                    print(f' ----------- Candidate Solution: {j} -----------')
                candidates = [candidate for candidate in range(self.de_hyperparameters["pop_size"]) if candidate != j]
                a, b, c = pop[choice(candidates, 3, replace=False)]

                mutated = self.mutation([a, b, c])
                trial = self.crossover(mutated, pop[j])

                fitness_results_target = self.obj(pop[j], starting_class_index, starting_class_label)
                obj_target = fitness_results_target["fitness"]

                fitness_results_trial = self.obj(trial, starting_class_index, starting_class_label)
                obj_trial = fitness_results_trial["fitness"]
                self.queries += 2

                # Early stop if the trial vector succeds
                if (obj_trial == float('-inf')):
                    adv_dict = utils.add_normalized_noise(self.clean_audio, trial, self.SNR_norm)

                    if self.verbosity:
                        print("----------- Attack Succeded -----------")
                    return {
                        "noise":
                        adv_dict["noise"],
                        "adversary":
                        adv_dict["adversary"],
                        "raw audio":
                        adv_dict["clean_audio"],
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


        adv_dict = utils.add_normalized_noise(self.clean_audio, best_vector, self.SNR_norm)

        return {
            "noise":
            adv_dict["noise"],
            "adversary":
            adv_dict["adversary"],
            "raw audio":
            adv_dict["clean_audio"],
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

        self.bounds = asarray([(-1, 1) for _ in range(len(self.clean_audio))])

        # Make inference to get index/label
        inference_results = self.model.make_inference_with_waveform(self.clean_audio)
        starting_class_index, starting_class_label = inference_results["predicted_class_idx"], inference_results[
            "label"]

        
        starting_class_label = str(self.model.hypercategory_mapping[starting_class_index])

        # Initialize queries counter
        self.queries = 0

        results = self.optimization(starting_class_index=starting_class_index,
                                    starting_class_label=starting_class_label)

        # Make inference with perturbed waveform
        results["queries"] += 1

        inference_results = self.model.make_inference_with_waveform(results["adversary"])
        probs, final_confidence = inference_results["probs"], inference_results["best_score"]

        # Get final confidence of starting class
        #Get indexes of all occurancies of the hyperclass
        hypercategory_idxs = np.where(self.model.hypercategory_mapping == starting_class_label)[0]

        # Get maximum probability
        max_prob = max(probs[hypercategory_idxs])


        results["Final Starting Class Confidence"] = max_prob
        results["Final Confidence"] = final_confidence

        # Append starting class label to results
        results['starting_class'] = starting_class_label

        return results
