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

from utils.init_utils import get_model, init_algorithm
from utils.utils import crawl_directory
from utils.attack_utils import filter_on_correct_predictions


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

    # Filter wav files
    wav_files_filtered = filter_on_correct_predictions(model=model, wav_files=wav_files, true_labels=true_labels)
    logging.info(
        f"Kept: {len(wav_files_filtered)} | Total: {len(wav_files)} ({100*len(wav_files_filtered)/len(wav_files):.2f})")

    