import json
import argparse
import pandas as pd


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', '-i', type=str, required=True, help='Input result csv from run_attack.py')
    parser.add_argument('--output_file', '-o', type=str, default="snr.csv", help='Output csv')
    parser.add_argument('--hypercategory_mapping',
                        '-hm',
                        type=str,
                        default='../data/audioset_hypercategory_mapping.json')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    with open(args.hypercategory_mapping, "r") as f:
        hypercategory_mapping = json.load(f)
    df = pd.read_csv(args.input_file)
    groups = df.groupby('Starting Class')
    snr_df = pd.DataFrame(columns=[
        'Name', 'num_files', 'Most_Common_Inferred_Class', 'Most_Common_Inferred_Class_Count', 'Mean_Iterations_Succ',
        'Mean_Queries_Succ', 'Mean_Start_Cl_Confidence_Start_Succ', 'Mean_Start_Cl_Confidence_End_Succ',
        'Mean_Inferred_Cl_Confidence_Succ', 'SNR_Mean_Succ', 'SNR_STD_Succ', 'Mean_Iterations_Fail',
        'Mean_Queries_Fail', 'Mean_Start_Cl_Confidence_Start_Fail', 'Mean_Start_Cl_Confidence_End_Fail',
        'Mean_Inferred_Cl_Confidence_Fail', 'SNR_Mean_Fail', 'SNR_STD_Fail'
    ])
    for name, group in groups:
        if name in hypercategory_mapping:
            continue
        success_df = group[group['Status'] == True]
        num_files = group['File'].size
        val_counts = success_df['Inferred Class'].value_counts()
        most_common_inferred_class = val_counts.idxmax()
        most_common_inferred_class_count = val_counts.max()
        mean_iter_succ = round(success_df['Iterations'].mean(), 3)
        mean_queries_succ = round(success_df['Queries'].mean(), 3)
        mean_start_class_st_conf_succ = round(success_df['Starting Class Confidence'].mean(), 3)
        mean_start_class_end_conf_succ = round(success_df['Final Confidence of Starting Class'].mean(), 3)
        mean_inferred_class_conf_succ = round(success_df['Final Confidence of Inferred Class'].mean(), 3)
        mean_snr_succ = round(success_df['SNR'].mean(), 3)
        std_snr_succ = success_df['SNR'].std()

        false_df = group[group['Status'] == False]

        mean_iter_false = round(false_df['Iterations'].mean(), 3)
        mean_queries_false = round(false_df['Queries'].mean(), 3)
        mean_start_class_st_conf_false = round(false_df['Starting Class Confidence'].mean(), 3)
        mean_start_class_end_conf_false = round(false_df['Final Confidence of Starting Class'].mean(), 3)
        mean_inferred_class_conf_false = round(false_df['Final Confidence of Inferred Class'].mean(), 3)
        mean_snr_false = false_df['SNR'].mean()
        std_snr_false = false_df['SNR'].std()
        snr_df.loc[len(snr_df)] = ([
            name, num_files, most_common_inferred_class, most_common_inferred_class_count, mean_iter_succ,
            mean_queries_succ, mean_start_class_st_conf_succ, mean_start_class_end_conf_succ,
            mean_inferred_class_conf_succ, mean_snr_succ, std_snr_succ, mean_iter_false, mean_queries_false,
            mean_start_class_st_conf_false, mean_start_class_end_conf_false, mean_inferred_class_conf_false,
            mean_snr_false, std_snr_false
        ])
    print(snr_df)
    snr_df.to_csv(args.output_file, index=False)
