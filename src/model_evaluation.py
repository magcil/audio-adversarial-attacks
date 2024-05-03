import os
import sys
import json
import argparse

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm
import librosa

from config.config import FIXED_PATHS, HYPERCATEGORIES
from utils.attack_utils import get_model, get_model_pred

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from tabulate import tabulate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_files", required=True, help="Full path of input wav files for AudioSet val")
    parser.add_argument("-m",
                        "--model",
                        required=True,
                        choices=["yamnet", "beats"],
                        help="Model to be used for inference")
    return parser.parse_args()

if __name__ == '__main__':

    # Parse fixed paths
    HYPERCATEGORY_PATH = os.path.join(PROJECT_PATH, FIXED_PATHS["hypercategory_path"])
    BEATs_WEIGHTS_PATH = os.path.join(PROJECT_PATH, FIXED_PATHS["beats_weights"])
    PATH_TO_SAMPLES = os.path.join(PROJECT_PATH, FIXED_PATHS["path_to_samples"])

    # Parse path to input wav files / model str
    args = parse_args()
    MODEL_NAME = args.model
    INPUT_FILES = args.input_files

    with open(HYPERCATEGORY_PATH, "r") as f:
        hypercategory_mapping = json.load(f)

    # Get model
    model = get_model(model_str=MODEL_NAME,
                      model_pt_file=BEATs_WEIGHTS_PATH,
                      hypercategory_mapping=hypercategory_mapping)

    with open(PATH_TO_SAMPLES, "r") as f:
        samples = json.load(f)

    correct_predictions = 0
    y_pred, y_true = [], []

    for wav_file in tqdm(samples.keys(), desc="Processing wav files", total=len(samples.keys())):

        wav_file_path = os.path.join(INPUT_FILES, wav_file)

        # Load waveform
        waveform, sr = librosa.load(wav_file_path, sr=16000)

        # Check if dur is at least 1 sec long
        dur = waveform.size // sr
        if not dur:
            continue

        # Get filename and remove suffix .wav
        filename = os.path.basename(wav_file_path)[:-4]

        # Get prediction label
        starting_class, true_idx,_ = get_model_pred(model, waveform)
        hypercategory_label = model.hypercategory_mapping[true_idx]

        full_filename = os.path.basename(wav_file_path)

        if (hypercategory_label == samples[full_filename]):
            correct_predictions += 1

        y_pred.append(hypercategory_label)
        y_true.append(samples[full_filename])

    total_accuracy_rate = correct_predictions / len(samples.keys())

    print(f"\n Evaluation of the Model {MODEL_NAME}\n")
    print(f"Accuracy: [{correct_predictions} | {len(samples.keys())}] ({100*total_accuracy_rate:.2f}  %)")
    print("Micro Precision: {:.2f}".format(precision_score(y_true, y_pred, average="micro")))
    print("Micro Recall: {:.2f}".format(recall_score(y_true, y_pred, average="micro")))
    print("Micro F1-score: {:.2f}\n".format(f1_score(y_true, y_pred, average="micro")))

    f1_scores = f1_score(y_true, y_pred, average=None)
    table = [["Class", "F1 Score"]]
    for class_index, f1_score_value in enumerate(f1_scores):
        class_name = HYPERCATEGORIES[class_index]
        table.append([class_name, round(100 * f1_score_value, 2)])

    print(tabulate(table))

    cm = confusion_matrix(y_true, y_pred, labels=HYPERCATEGORIES)

    df_cm = pd.DataFrame(cm, HYPERCATEGORIES, HYPERCATEGORIES)
    plt.figure(figsize=(8, 6))
    ax = sn.heatmap(df_cm, annot=True, annot_kws={"size": 18}, cmap="Blues", square=True, cbar=True, fmt='g')
    plt.title("Classes Heatmap")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=12)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()
