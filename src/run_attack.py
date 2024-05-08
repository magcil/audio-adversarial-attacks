import os
import csv
import sys
import json
import argparse
from datetime import datetime
from collections import Counter

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import plotly.graph_objects as go
import numpy as np
import librosa
import tabulate
from tqdm import tqdm
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import utils
from utils.attack_utils import get_model_pred, get_model, is_true_pred, init_algorithm

# Number of samples to save from successful/non_successful attacks
NUM_SAMPLES_STORE = 20

# Results folder's path
PROJECT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_PATH = os.path.join(PROJECT_PATH, "results")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', required=True, help='The experiment json configuration file.')

    return parser.parse_args()


def audio_files_attack(model,
                       wav_files,
                       true_labels,
                       attack_algorithm,
                       experiment_path,
                       hypercategory_mapping,
                       store_samples=True):

    # Sample counter
    sample_counter = Counter({"successful": 0, "non_successful": 0})
    (successes, total_attacks, miss_predictions, skipped, success_queries, success_snr,
     success_iterations) = 0, 0, 0, 0, 0, 0, 0
    if len(model.hypercategory_mapping):
        hypercategory_attack_accuracy_dict = {
            hypercategory: Counter()
            for hypercategory in list(set(model.hypercategory_mapping))
        }
        hypercategory_attack_acuracy_headers = ['hypercategory', 'successes', 'num_trials', 'accuracy_rate']
        with open(hypercategory_attack_accuracy_csv_path, 'a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(hypercategory_attack_acuracy_headers)

    for wav_file in tqdm(wav_files, desc="Processing wav files", total=len(wav_files)):
        # Load waveform
        waveform, sr = librosa.load(wav_file, sr=16000)
        # Check if dur is at least 1 sec long
        dur = waveform.size // sr
        if not dur:
            skipped += 1
            continue
        # Get filename and remove suffix .wav
        filename = os.path.basename(wav_file)[:-4]

        # Get prediction label
        starting_class, true_idx, starting_confidence = get_model_pred(model, waveform)

        # Check if prediction is correct
        if true_labels:
            # Check on hypercategory level
            if hypercategory_mapping:
                inferred_hypercategory = model.hypercategory_mapping[true_idx]
                file_hypercategories = [hypercategory_mapping[x] for x in true_labels[filename]]
                if inferred_hypercategory not in file_hypercategories:
                    attack_results_row = [
                        filename, False, starting_class, starting_confidence, "Miss Prediction", starting_confidence, 0,
                        0, 0, 0
                    ]
                    with open(results_csv_path, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(attack_results_row)
                    miss_predictions += 1
                    skipped += 1
                    continue
            # Check on label level
            else:
                if not is_true_pred(filename=filename, true_labels=true_labels, inferred_class_name=starting_class):
                    attack_results_row = [
                        filename, False, starting_class, starting_confidence, "Miss Prediction", starting_confidence, 0,
                        0, 0, 0
                    ]
                    with open(results_csv_path, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(attack_results_row)
                    # Write csv row / skip attack
                    miss_predictions += 1
                    skipped += 1
                    continue

        # Perform Attack
        attack_results = attack_algorithm.generate_adversarial_example(source_audio=waveform)
        snr = utils.calculate_snr(attack_results["raw audio"], attack_results["noise"])

        # Update success rate's counters
        if attack_results['success']:
            successes += 1
            success_snr += snr
            success_queries += attack_results["queries"]
            success_iterations += attack_results["iterations"]
        total_attacks += 1

        # Store samples for further inspection
        if store_samples:
            save_path = ""
            if attack_results["success"] and sample_counter["successful"] < NUM_SAMPLES_STORE:
                save_path = os.path.join(experiment_path, "successful", filename + "_perturbated.wav")
                sample_counter["successful"] += 1
            elif not attack_results["success"] and sample_counter["non_sucessful"] < NUM_SAMPLES_STORE:
                save_path = os.path.join(experiment_path, "non_successful", filename + "_perturbated.wav")
                sample_counter["non_sucessful"] += 1
            if save_path:
                sf.write(file=save_path, data=attack_results["adversary"], samplerate=16000, subtype="FLOAT")

        if len(model.hypercategory_mapping):
            starting_class = model.hypercategory_mapping[true_idx]
            # Attack succeded
            if starting_class != attack_results['inferred_class']:
                hypercategory_attack_accuracy_dict[starting_class]['successes'] += 1
            # Always update num_iters
            hypercategory_attack_accuracy_dict[starting_class]['num_trials'] += 1

        # Row info to write on csv
        attack_results_row = [
            filename, attack_results["success"], starting_class, starting_confidence, attack_results["inferred_class"],
            attack_results["Final Starting Class Confidence"], attack_results["Final Confidence"],
            attack_results["iterations"], attack_results["queries"], snr
        ]

        with open(results_csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(attack_results_row)
    if len(model.hypercategory_mapping):
        with open(hypercategory_attack_accuracy_csv_path, 'a', newline="") as csv_file:
            writer = csv.writer(csv_file)
            for key, value in hypercategory_attack_accuracy_dict.items():
                if value['num_trials'] == 0:
                    attack_accuracy = 0
                else:
                    attack_accuracy = value['successes'] / value['num_trials']
                writer.writerow([key, value['successes'], value['num_trials'], attack_accuracy])

    return (successes, total_attacks, miss_predictions, skipped, success_queries, success_snr, success_iterations)


def create_graphs(pt_list, print_dict, experiment_base_path):
    fig = go.Figure([go.Bar(x=pt_list, y=print_dict["Successes"])])
    # Customize layout, if desired
    fig.update_layout(
        title='Success Ratio per Perturbation Ratio',
        xaxis_title='PT_Ratio',
        yaxis_title='Successes',
        bargap=0.2,  # gap between bars of adjacent location coordinates
    )
    # save the figure
    fig.write_image(os.path.join(experiment_base_path, 'success_perturbation.png'))

    fig = go.Figure([go.Bar(x=pt_list, y=print_dict["Success avg Queries"])])
    # Customize layout, if desired
    fig.update_layout(
        title='Queries per Perturbation Ratio',
        xaxis_title='PT_Ratio',
        yaxis_title='Queries',
        bargap=0.2,  # gap between bars of adjacent location coordinates
    )
    # save the figure
    fig.write_image(os.path.join(experiment_base_path, 'queries_perturbation.png'))

    fig = go.Figure([go.Bar(x=pt_list, y=print_dict["Success avg SNR"])])
    # Customize layout, if desired
    fig.update_layout(
        title='SNR per Perturbation Ratio',
        xaxis_title='PT_Ratio',
        yaxis_title='SNR',
        bargap=0.2,  # gap between bars of adjacent location coordinates
    )
    # save the figure
    fig.write_image(os.path.join(experiment_base_path, 'snr_perturbation.png'))


if __name__ == '__main__':

    args = parse_args()
    with open(args.config_file, "r") as f:
        config_file = json.load(f)

    if "objective_function" not in config_file.keys():
        objective_function = None
    else:
        objective_function = config_file["objective_function"]

    if "hypercategory_target" not in config_file.keys():
        hypercategory_target = None
    else:
        hypercategory_target = config_file["hypercategory_target"]

    if "target_class" not in config_file.keys():
        target_class = None
    else:
        target_class = config_file["target_class"]

    # Get hypercategories if given
    if "hypercategory_mapping" in config_file.keys() and config_file["hypercategory_mapping"] is not None:
        with open(os.path.join(PROJECT_PATH, config_file["hypercategory_mapping"]), "r") as f:
            hypercategory_mapping = json.load(f)
        # Get balanced subset if given
        if 'balanced_subset' in config_file.keys() and config_file['balanced_subset'] is not None:
            with open(os.path.join(PROJECT_PATH, config_file['balanced_subset']), 'r') as f:
                balanced_subset = json.load(f)
        else:
            balanced_subset = None
    else:
        hypercategory_mapping = None

    # Get model
    model = get_model(model_str=config_file["model_name"],
                      model_pt_file=os.path.join(PROJECT_PATH, config_file["model_pt_file"]),
                      hypercategory_mapping=hypercategory_mapping)

    # Initialiation of algorithm
    ATTACK_ALGORITHM = init_algorithm(algorithm=config_file['algorithm'],
                                      model=model,
                                      hyperparameters=config_file['algorithm_hyperparameters'],
                                      verbosity=False,
                                      objective_function=objective_function,
                                      target_class=target_class,
                                      hypercategory_target=hypercategory_target)

    # Check if true labels are given
    if "true_labels" in config_file.keys() and config_file["true_labels"] is not None:
        with open(os.path.join(PROJECT_PATH, config_file["true_labels"]), "r") as f:
            true_labels = json.load(f)
    else:
        true_labels = None

    # Results subfolder
    EXPERIMENT_FOLDER_NAME = f"Experiment_{datetime.today().date()}_{datetime.today().hour}_{datetime.today().minute}" +\
        f"_{datetime.today().second}"
    experiment_base_path = os.path.join(RESULTS_PATH, EXPERIMENT_FOLDER_NAME)
    os.mkdir(experiment_base_path)
    num_files_total = config_file["num_samples"] if "num_samples" in config_file else 0

    if balanced_subset:
        wav_files = []
        wav_names = list(balanced_subset.keys())
        for wav_name in wav_names:
            wav_file_dir = os.path.join(PROJECT_PATH, config_file['input_path'] + "/" + wav_name)
            wav_files.append(wav_file_dir)
    else:
        # Get input wav files
        wav_files = utils.crawl_directory(directory=os.path.join(PROJECT_PATH, config_file["input_path"]),
                                          extension="wav",
                                          num_files=num_files_total)

    # configure perturbation ratios to use, if config is given
    gs_config = config_file["grid_search"] if "grid_search" in config_file else {}
    pt_list = [ATTACK_ALGORITHM.perturbation_ratio]
    if gs_config:
        pt_start = gs_config["perturbation_ratio_start"]
        pt_stop = gs_config["perturbation_ratio_stop"]
        pt_step = gs_config["perturbation_ratio_step"]
        pt_list = \
            np.arange(pt_start, pt_stop, pt_step)
        print(f'\nStarting grid search on perturbation ratios: '
              f'\t\nrange: [{pt_start}, {pt_stop}], step: {pt_step}]')

    # whether to store audio files from attack
    store_samples = config_file["store_samples"] if "store_samples" in config_file else True
    header_row = [
        'File', 'Status', 'Starting Class', 'Starting Class Confidence', 'Inferred Class',
        'Final Confidence of Starting Class', 'Final Confidence of Inferred Class', 'Iterations', 'Queries', 'SNR'
    ]
    print_dict = {"Perturbation ratio": pt_list}

    for pt_ratio in pt_list:
        pt_ratio = round(pt_ratio, 3)
        ATTACK_ALGORITHM.perturbation_ratio = pt_ratio
        experiment_path = os.path.join(experiment_base_path, f"pt_{pt_ratio}")
        os.makedirs(os.path.join(experiment_path, "successful"))
        os.makedirs(os.path.join(experiment_path, "non_successful"))

        results_csv_path = os.path.join(experiment_path, "results.csv")
        hypercategory_attack_accuracy_csv_path = os.path.join(experiment_path, "hypercategory_attack_accuracy.csv")
        with open(results_csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header_row)

        (successes, total_attacks, miss_predictions, skipped, success_queries,
         success_snr, success_iterations) = \
            audio_files_attack(model=model, wav_files=wav_files, true_labels=true_labels,
                               attack_algorithm=ATTACK_ALGORITHM,
                               experiment_path=experiment_path,
                               store_samples=store_samples,
                               hypercategory_mapping=hypercategory_mapping)

        # calculate metrics for logging
        success_rate = successes / total_attacks if total_attacks > 0 else 0
        miss_pred_rate = miss_predictions / (total_attacks + miss_predictions)
        skipped_rate = skipped / len(wav_files)
        avg_success_queries = success_queries / successes if successes > 0 else 0
        avg_success_snr = success_snr / successes if successes > 0 else 0
        avg_success_iterations = success_iterations / successes if successes > 0 else 0
        print_dict.setdefault("Successes", []).append(successes)
        print_dict.setdefault("Total Attacks", []).append(total_attacks)
        print_dict.setdefault("Missed", []).append(miss_predictions)
        print_dict.setdefault("Skipped", []).append(skipped)
        print_dict.setdefault("Success avg SNR", []).append(avg_success_snr)
        print_dict.setdefault("Success avg Queries", []).append(avg_success_queries)
        print_dict.setdefault("Success avg Iterations", []).append(avg_success_iterations)
        print_dict.setdefault("Success rate (%)", []).append(100 * success_rate)
        print_dict.setdefault("Miss pred rate (%)", []).append(100 * miss_pred_rate)
        print_dict.setdefault("Skipped rate (%)", []).append(100 * skipped_rate)
    print(tabulate.tabulate(print_dict, headers='keys', tablefmt='rst'))

    # save success_ratio/perturbation plot
    if gs_config:
        create_graphs(pt_list, print_dict, experiment_base_path)
