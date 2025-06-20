import os
import sys
import yaml
import logging
import argparse
import json
import re
import numpy as np

# Initialize Project Path & Append to sys
PROJECT_PATH = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, PROJECT_PATH)

from tqdm import tqdm
import soundfile as sf
from prettytable import PrettyTable
from datetime import datetime

from utils.ast_init import get_model, init_algorithm
from utils.utils import crawl_directory, calculate_snr
from utils.attack_utils import filter_on_correct_predictions, perform_single_attack


def parse_args():
    parser = argparse.ArgumentParser(prog="run_attack_dev.py",
                                     description="Perform an attack on a path of wav files on a pretrained model")
    parser.add_argument("-c", "--config_file", required=True, help="Configuration file (.yaml) for the attack.")

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    # Get Config file
    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)

    # Initialize logger
    logging.basicConfig(filename=os.path.join(
        PROJECT_PATH, config['log_path'],
        config['log_file'] + f'_{datetime.now().date()}.log'),
                        encoding='utf-8',
                        level=logging.INFO,
                        force=True,
                        filemode='w',
                        format='%(asctime)s %(message)s')

    # Log configuration params
    for k, v in config.items():
        logging.info(f"{k}:{v}")
        
    # Number of samples to store for inspection (if specified)
    num_examples_to_store = config.get("num_examples_to_store", None)

    # Initialize Model
    model = get_model(model_str=config['model_architecture'],
                      model_pt_file=config['model_pretrained_weights'],
                      hypercategory_mapping=config['hypercategory_mapping'],
                      device=config.get('device', 'cpu'))

    # Get Wav Data
    wav_files = crawl_directory(directory=config['data_path'],
                                extension=".wav")

    # Get True Labels
    with open(config['true_labels'], 'r') as f:
        true_labels = json.load(f)

    # Filter Wavs and Keep those with 1 Hypercategory
    filtered_wav_files = [
        wav_file for wav_file in wav_files
        if os.path.basename(wav_file)[:-4] in list(true_labels.keys())
    ]

    # Log the number of filtered wav files
    logging.info(
        f"Total Wav files: {len(wav_files)} | Belong to 1 Hypercategory: {len(filtered_wav_files)}"
    )

    # Filter wav files
    filter_results = filter_on_correct_predictions(
        model=model,
        wav_files=filtered_wav_files,
        true_labels=true_labels,
        hypercategory_mapping=config['hypercategory_mapping'])
    correct_wav_files = filter_results['filtered_wavs']

    # Log the number of filtered wav files
    logging.info(
        f"Total Filtered Wav files: {len(filtered_wav_files)} | Belong to 1 Hypercategory: {len(filtered_wav_files)}"
    )

    logging.info(f"{filter_results['classification_report']}\n")
    logging.info(
        f"Correct: {len(correct_wav_files)} ({100*len(correct_wav_files)/len(filtered_wav_files):.2f} %)"
    )

    # Get Unique Class Names
    CLASS_NAMES = sorted(list(set(true_labels.values())))
    CLASS_MAPPING = {class_name: i for i, class_name in enumerate(CLASS_NAMES)}

    # SNR / Success Rate table
    snr_succes_rate_table = PrettyTable()
    snr_succes_rate_table.field_names = ['SNR', 'Success Rate']


    # Create directory to store examples.
    if num_examples_to_store:
        os.makedirs(os.path.join(PROJECT_PATH, "Examples"), exist_ok=True)

    for SNR_norm in config["SNR_norm"]:
        logging.info(f"SNR: {SNR_norm}")

        # Create directory based on SNR
        if num_examples_to_store:
            snr_dict = os.path.join(PROJECT_PATH, "Examples", f"{config['model_architecture']}",f"snr_{SNR_norm}")
            os.makedirs(snr_dict, exist_ok=True)
            files_counter = 0

        # Initialize Algorithm
        ATTACK_ALGORITHM = init_algorithm(
            algorithm=config['algorithm'],
            model=model,
            hyperparameters=config['algorithm_hyperparameters'],
            objective_function=config['objective_function'],
            SNR_norm=SNR_norm,
            verbosity=config.get("verbosity", None))

        # Initialize Table to Calculate Aggregated Results
        results_table = PrettyTable()
        results_table.field_names = [' ', *CLASS_NAMES, 'Success Rate (%)']

        # Aggregation Results
        aggregated_table_results = PrettyTable()
        aggregated_table_results.field_names = [
            'Total Attacks', 'Success Rate', 'Avg. SNR', 'Std. SNR', 'Queries'
        ]

        # Initialize Class Dict / Information of Attack Distribution Per Class
        CLASSES_DICT = {
            classname: len(CLASS_NAMES) * [0]
            for classname in CLASS_NAMES
        }

        aggregated_successes, aggregated_SNR, aggregated_queries = 0, [], 0

        # Loop through wav files
        for wav_file in tqdm(correct_wav_files,
                             desc="Processing wav files",
                             total=len(correct_wav_files)):

            attack_results = perform_single_attack(
                ATTACK_ALGORITHM=ATTACK_ALGORITHM, wav_file=wav_file)

            # Update Aggregate successes & Queries
            if attack_results["success"]:
                aggregated_successes += 1
                aggregated_queries += attack_results["queries"]

            # Append SNR of adversarial example
            aggregated_SNR.append(
                calculate_snr(signal=attack_results['raw audio'],
                              noise=attack_results["noise"]))

            starting_class, predicted_class = attack_results[
                "starting_class"], attack_results["inferred_class"]

            predicted_class_idx = CLASS_MAPPING[predicted_class]

            # Update Class Dict
            CLASSES_DICT[starting_class][predicted_class_idx] += 1
            
            if num_examples_to_store and files_counter != num_examples_to_store and attack_results["success"]:
                filename = os.path.splitext(os.path.basename(wav_file))[0]
                starting_class_store = re.sub(r"[ /,]", "_", starting_class)
                pred_class_store = re.sub(r"[ /,]", "_", predicted_class)

                sf.write(os.path.join(snr_dict, f"{filename}_{starting_class_store}.wav"), attack_results["raw audio"],
                         16000, subtype="FLOAT")
                sf.write(os.path.join(snr_dict, f"{filename}_{pred_class_store}_adversarial.wav"),
                         attack_results["adversary"], 16000, subtype="FLOAT")
                files_counter += 1

        # Calculate Results Table
        for classname, class_distribution in CLASSES_DICT.items():
            class_idx = CLASS_MAPPING[classname]

            # Prevent Division by Zero
            if sum(class_distribution) == 0:
                continue

            succ_rate = (
                sum(class_distribution) -
                class_distribution[class_idx]) / sum(class_distribution)

            results_table.add_row(
                [classname, *class_distribution, f"{100 * succ_rate:.2f}"])

        aggregated_table_results.add_row([
            len(correct_wav_files),
            f"{100 * aggregated_successes / len(correct_wav_files):.2f}",
            f"{np.nanmean(aggregated_SNR):.2f}", f"{np.nanstd(aggregated_SNR):.2f}",
            f"{100*aggregated_queries/aggregated_successes:.2f}"
        ])

        snr_succes_rate_table.add_row([
            f"{SNR_norm}",
            f"{100 * aggregated_successes / len(correct_wav_files):.2f}"
        ])

        # Log Class Results Table
        logging.info(f"Attack Distribution Table\n{results_table}")

        # Log Aggregated Table
        logging.info(f"Aggregated Result Table\n{aggregated_table_results}")

    # Log SNR / Success rate table
    logging.info(f"Aggregated Result Table\n{snr_succes_rate_table}")
