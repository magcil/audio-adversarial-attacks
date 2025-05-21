import os
import json
import sys
import argparse

import pandas as pd

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_PATH)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-hc", "--hypercategory_mapping", required=True, type=str, help='Hypercategory mapping.')
    parser.add_argument("-tl", "--true_labels", required=True, type=str, help="Audioset true labels.")
    parser.add_argument("-n",
                        "--number_of_samples",
                        required=True,
                        type=int,
                        help='Number of samples per hypercategory')
    parser.add_argument("-t", "--target_path", required=True, type=str, help="Target path and name to store json output file.")

    return parser.parse_args()


def create_data_subsets(hypercategory_json_path,
                        audioset_val_true_labels_path,
                        num_of_samples,
                        hypercategory_list=None,
                        random_seed=42):

    # Load hypercategory mapping from JSON
    with open(hypercategory_json_path) as f:
        hypercategory_mapping = json.load(f)

    # Load true labels from JSON
    with open(audioset_val_true_labels_path) as g:
        wavs_true_labels = json.load(g)

    subset_rows = []
    for wav, list_of_labels in wavs_true_labels.items():
        hypercategories = {hypercategory_mapping[label] for label in list_of_labels}

        # Filter wavs that belong to more than one hypercategory
        if len(hypercategories) != 1:
            continue

        # Handle custom given hypercategories
        if hypercategory_list and next(iter(hypercategories)) not in hypercategory_list:
            continue

        # wav = wav + ".wav"
        row = {"wav_file": wav, 'hypercategory': next(iter(hypercategories))}
        subset_rows.append(row)

    subset_df_all_rows = pd.DataFrame(subset_rows).reset_index(drop=True)

    sampled_df = subset_df_all_rows.groupby(
        'hypercategory',
        group_keys=False).apply(lambda group: group.sample(min(len(group), num_of_samples), random_state=random_seed))
    sampled_df.reset_index(drop=True, inplace=True)
    return sampled_df


def write_subset_to_json(subset_df, writepath):
    subset_dict = {}
    for _, row in subset_df.iterrows():
        wav_name = row.iloc[0]
        hypercategory = row.iloc[1]
        subset_dict[wav_name] = hypercategory

    with open(writepath, 'w') as json_file:
        json.dump(subset_dict, json_file, indent=2)


if __name__ == '__main__':
    args = parse_args()

    df = create_data_subsets(hypercategory_json_path=args.hypercategory_mapping,
                             audioset_val_true_labels_path=args.true_labels,
                             num_of_samples=args.number_of_samples)
    
    write_subset_to_json(subset_df=df, writepath=args.target_path)