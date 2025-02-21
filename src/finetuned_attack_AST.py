import os
import sys
import yaml
import logging
import argparse
import numpy as np
import torch
import re

# Initialize Project Path & Append to sys
PROJECT_PATH = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, PROJECT_PATH)

from tqdm import tqdm
import soundfile as sf
from datetime import datetime
from prettytable import PrettyTable

from utils.ast_init import init_algorithm
from utils.utils import calculate_snr
from utils.attack_utils import perform_single_attack

from models.classifier_ast import FineTuneClassifierAST
from datasets.datasets import ESC50Dataset
from finetune.loops import training_loop
from torch.utils.data import DataLoader
from datasets import ESC_INV_CLASS_MAPPING

from sklearn.metrics import classification_report


def parse_args():
    parser = argparse.ArgumentParser(prog="finetuned_attack.py",
                                     description="Perform an attack on a path of wav files on a pretrained model")
    parser.add_argument("-c", "--config_file", required=True, help="Configuration file (.yaml) for the attack.")

    return parser.parse_args()


def filter_predictions(model, d_set, num_workers, device):
    filtered_wavs = []
    y_true, y_pred = [], []

    d_set = DataLoader(d_set, batch_size=1, shuffle=False, num_workers=num_workers)

    model.eval()
    with torch.no_grad():
        for item in d_set:
            waveform = item["waveform"].to(device)
            logits = model.forward(waveform)
            max_idx = torch.argmax(logits, dim=1).item()

            y_pred.append(ESC_INV_CLASS_MAPPING[max_idx])
            y_true.append(item["hypercategory"][0])

            if ESC_INV_CLASS_MAPPING[max_idx] == item["hypercategory"][0]:
                filtered_wavs.append(item["filename"][0])

    unique_names = np.array(list(ESC_INV_CLASS_MAPPING.values()))

    return {
        "filtered_wavs": filtered_wavs,
        "classification_report": classification_report(y_true=y_true, y_pred=y_pred, labels=unique_names)
    }


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
    for k, v in config.items():
        logging.info(f"{k}:{v}")

    # Initialize Model

    model = FineTuneClassifierAST(model_name=config["model_architecture"],
                                  num_classes=config["num_classes"],
                                  weight_path=config.get('model_pretrained_weights', 'None'),
                                  freeze_backbone=True,
                                  device=config.get('device', 'cpu'))

    # Initialize datasets
    data_path, metadata_csv, json_hypercategory = config["data_path"], config["metadata_csv"], config[
        "esc_hypercategories"]
    train_folds, val_folds = config["train_folds"][:-1], config["train_folds"][-1:]
    train_dset = ESC50Dataset(data_path=data_path,
                              metadata_csv=metadata_csv,
                              hypercategory_mapping=json_hypercategory,
                              folds=train_folds)
    val_dset = ESC50Dataset(data_path=data_path,
                            metadata_csv=metadata_csv,
                            hypercategory_mapping=json_hypercategory,
                            folds=val_folds)
    test_dset = ESC50Dataset(data_path=data_path,
                             metadata_csv=metadata_csv,
                             hypercategory_mapping=json_hypercategory,
                             folds=config["test_folds"])

    logging.info(
        f"Train Wav files: {len(train_dset)} | Validation Wav files: {len(val_dset)} | Test Wav files: {len(test_dset)} "
    )

    # Trainning Loop
    training_loop(model=model,
                  train_dset=train_dset,
                  val_dset=val_dset,
                  batch_size=config["batch_size"],
                  epochs=config["epochs"],
                  learning_rate=float(config["learning_rate"]),
                  patience=config["patience"],
                  pt_file=config["pt_file"] + ".pt",
                  num_workers=config["num_workers"],
                  weight_decay=config["weight_decay"],
                  device=config.get('device', 'cpu'))

    # Test Loop
    path_to_pt_file = os.path.join(PROJECT_PATH, "pretrained_models", config['pt_file'] + ".pt")

    # Load model
    model.load_state_dict(torch.load(path_to_pt_file, weights_only=True))
    # Move to device
    model = model.to(config.get('device', 'cpu'))

    # Filter wav files / Report Classification scores
    filter_results = filter_predictions(model,
                                        test_dset,
                                        num_workers=config["num_workers"],
                                        device=config.get('device', 'cpu'))
    correct_wav_files = filter_results['filtered_wavs']

    logging.info(f"{filter_results['classification_report']}\n")
    logging.info(f"Correct: {len(correct_wav_files)} ({100*len(correct_wav_files)/len(test_dset):.2f} %)")

    # Get Unique Class Names
    CLASS_NAMES = list(ESC_INV_CLASS_MAPPING.values())
    CLASS_MAPPING = {str(class_name): i for i, class_name in enumerate(CLASS_NAMES)}

    # SNR / Success Rate table
    snr_succes_rate_table = PrettyTable()
    snr_succes_rate_table.field_names = ['SNR', 'Success Rate']

    # Create directory to store examples.
    if config["num_examples_to_store"]:
        os.makedirs(os.path.join(PROJECT_PATH, "Examples"), exist_ok=True)

    for SNR_norm in config["SNR_norm"]:
        logging.info(f"SNR: {SNR_norm}")

        # Create directory based on SNR
        if config["num_examples_to_store"]:
            snr_dict = os.path.join(PROJECT_PATH, "Examples", f"snr_{SNR_norm}")
            os.makedirs(snr_dict, exist_ok=True)
            files_counter = 0

        # Initialize Algorithm
        ATTACK_ALGORITHM = init_algorithm(algorithm=config['algorithm'],
                                          model=model,
                                          hyperparameters=config['algorithm_hyperparameters'],
                                          objective_function=config['objective_function'],
                                          SNR_norm=SNR_norm,
                                          target_class=config.get("target_class", None),
                                          hypercategory_target=config.get("hypercategory_target", None),
                                          verbosity=config.get("verbosity", None))

        # Initialize Table to Calculate Aggregated Results
        results_table = PrettyTable()
        results_table.field_names = [' ', *CLASS_NAMES, 'Success Rate (%)']

        # Aggregation Results
        aggregated_table_results = PrettyTable()
        aggregated_table_results.field_names = ['Total Attacks', 'Success Rate', 'Avg. SNR', 'Std. SNR', 'Queries']

        # Initialize Class Dict / Information of Attack Distribution Per Class
        CLASSES_DICT = {classname: len(CLASS_NAMES) * [0] for classname in CLASS_NAMES}

        aggregated_successes, aggregated_SNR, aggregated_queries = 0, [], 0

        # Loop through wav files
        for wav_file in tqdm(correct_wav_files, desc="Processing wav files", total=len(correct_wav_files)):

            attack_results = perform_single_attack(ATTACK_ALGORITHM=ATTACK_ALGORITHM, wav_file=wav_file)

            # Update Aggregate successes & Queries
            if attack_results["success"]:
                aggregated_successes += 1
                aggregated_queries += attack_results["queries"]

            # Append SNR of adversarial example
            aggregated_SNR.append(calculate_snr(signal=attack_results['raw audio'], noise=attack_results["noise"]))

            starting_class, predicted_class = attack_results["starting_class"], attack_results["inferred_class"]

            predicted_class_idx = CLASS_MAPPING[predicted_class]

            # Update Class Dict
            CLASSES_DICT[starting_class][predicted_class_idx] += 1

            if config["num_examples_to_store"] and files_counter != int(
                    config["num_examples_to_store"]) and attack_results["success"]:
                filename = os.path.splitext(os.path.basename(wav_file))[0]
                starting_class_store = re.sub(r"[ /,]", "_", starting_class)
                pred_class_store = re.sub(r"[ /,]", "_", predicted_class)

                sf.write(os.path.join(snr_dict, f"{filename}_{starting_class_store}.wav"), attack_results["raw audio"],
                         16000)
                sf.write(os.path.join(snr_dict, f"{filename}_{pred_class_store}_adversarial.wav"),
                         attack_results["adversary"], 16000)
                files_counter += 1

        # Calculate Results Table
        for classname, class_distribution in CLASSES_DICT.items():
            class_idx = CLASS_MAPPING[classname]

            # Prevent Division by Zero
            if sum(class_distribution) == 0:
                continue

            succ_rate = (sum(class_distribution) - class_distribution[class_idx]) / sum(class_distribution)

            results_table.add_row([classname, *class_distribution, f"{100 * succ_rate:.2f}"])

        aggregated_table_results.add_row([
            len(correct_wav_files), f"{100 * aggregated_successes / len(correct_wav_files):.2f}",
            f"{np.nanmean(aggregated_SNR):.2f}", f"{np.nanstd(aggregated_SNR):.2f}",
            f"{100*aggregated_queries/aggregated_successes:.2f}"
        ])

        snr_succes_rate_table.add_row([f"{SNR_norm}", f"{100 * aggregated_successes / len(correct_wav_files):.2f}"])

        # Log Class Results Table
        logging.info(f"Attack Distribution Table\n{results_table}")

        # Log Aggregated Table
        logging.info(f"Aggregated Result Table\n{aggregated_table_results}")

    # Log SNR / Success rate table
    logging.info(f"Aggregated Result Table\n{snr_succes_rate_table}")
