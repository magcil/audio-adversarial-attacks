import os
import sys
import yaml
import logging
import datetime
import argparse
import json

# Initialize Project Path & Append to sys
PROJECT_PATH = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, PROJECT_PATH)

from tqdm import tqdm
from prettytable import PrettyTable

from utils.init_utils import get_model, init_algorithm
from utils.utils import crawl_directory
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
    logging.basicConfig(filename=os.path.join(PROJECT_PATH, config['log_path'],
                                              config['log_file'] + f'_{datetime.now().date()}.log'),
                        encoding='utf-8',
                        level=logging.INFO,
                        force=True,
                        filemode='w',
                        format='%(asctime)s %(message)s')

    # Log configuration params
    for k, v in config.keys():
        logging.info(f"{k}:{v}")

    # Initialize Model
    model = get_model(model_str=config['model_architecture'],
                      model_pt_file=config['model_pretrained_weights'],
                      hypercategory_mapping=config['hypercategory_mapping'])
    # Initialize Algorithm
    ATTACK_ALGORITHM = init_algorithm(algorithm=config['algorithm'],
                                      hyperparameters=config['algorithm_hyperparameters'],
                                      objective_function=config['objective_function'],
                                      target_class=config.get("target_class", None),
                                      hypercategory_target=config.get("hypercategory_target", None),
                                      verbosity=config.get("verbosity", None))

    # Get Wav Data
    wav_files = crawl_directory(directory=config['data_path'], extension=".wav")

    # Get True Labels
    with open(config['true_labels'], 'r') as f:
        true_labels = json.load(f)

    # Filter Wavs and Keep those with 1 Hypercategory
    filtered_wav_files = [wav_file for wav_file in wav_files if wav_file in true_labels.keys()]
    # Log the number of filtered wav files
    logging.info(f"Total Wav files: {len(wav_files)} | Belong to 1 Hypercategory: {len(filtered_wav_files)}")

    # Filter wav files
    filter_results = filter_on_correct_predictions(model=model, wav_files=filtered_wav_files, true_labels=true_labels)
    correct_wav_files = filter_results['filtered_wavs']
    logging.info(f"{filter_results['classification_report']}\n")
    logging.info(f"Correct: {len(correct_wav_files)} ({100*len(correct_wav_files)/len(filtered_wav_files):.2f} %)")

    # Get Unique Class Names
    CLASS_NAMES = sorted(list(set(true_labels.values())))
    CLASS_MAPPING = {class_name: i for i, class_name in enumerate(CLASS_NAMES)}

    # Initialize Table to Calculate Aggregated Results
    results_table = PrettyTable()
    results_table.field_names = [' ', *CLASS_NAMES, 'Success Rate (%)']

    # Initialize Class Dict / Information of Attack Distribution Per Class
    CLASSES_DICT = {classname: len(CLASS_NAMES) * [0] for classname in CLASS_NAMES}

    # Loop through wav files
    for wav_file in tqdm(correct_wav_files, desc="Processing wav files", total=len(correct_wav_files)):

        attack_results = perform_single_attack(ATTACK_ALGORITHM=ATTACK_ALGORITHM, wav_file=wav_file)

        starting_class, predicted_class = attack_results["starting_class"], attack_results["inferred_class"]

        predicted_class_idx = CLASS_MAPPING[predicted_class]

        # Update Class Dict
        CLASSES_DICT[starting_class][predicted_class_idx] += 1

    # Calculate Results Table
    for classname, class_distribution in CLASSES_DICT.items():
        class_idx = CLASS_MAPPING[classname]
        succ_rate = (sum(class_distribution) - class_distribution[class_idx]) / sum(class_distribution)

        results_table.add_row([classname, *class_distribution, format(100 * succ_rate, str=":.2f")])
        
    
    # Log Table
    logging.info(f"{results_table}")
