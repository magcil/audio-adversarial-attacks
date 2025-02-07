import os
import sys
from typing import Optional, List
import json

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from torch.utils.data import Dataset
import pandas as pd
import librosa

from utils.utils import crawl_directory
from datasets import ESC_CLASS_MAPPING, ESC_INV_CLASS_MAPPING


class ESC50Dataset(Dataset):
    """Dataset Class for ESC-50 Hypercategory Parsing"""

    def __init__(self,
                 data_path: os.PathLike,
                 metadata_csv: os.PathLike,
                 hypercategory_mapping: os.PathLike,
                 folds: Optional[List[int]] = None):
        """
        Args:
            data_path (os.PathLike): Abs Path to wav files of ESC-50.
            metadata_csv (os.PathLike): Abs Path to metadata csv.
            hypercategory_mapping (os.PathLike): Abs Path to hypercategory mapping json.
            folds (List[int]): List of folds to filter wav files.
        """

        # Parse wav files
        self.wav_files = crawl_directory(directory=data_path, extension=".wav")
        self.metadata_df = pd.read_csv(metadata_csv)
        self.hypercategory_mapping = self.parse_hypercategory_mapping(hypercategory_mapping)

        # Filter on folds
        if folds:
            self.metadata_df = self.metadata_df[self.metadata_df['fold'].isin(folds)]

        # Create List[wav_path, label, hypercategory, label int]
        wavs_to_keep = self.metadata_df['filename'].tolist()
        self.items = []
        for wav_file in self.wav_files:
            filename = os.path.basename(wav_file)
            if filename in wavs_to_keep:
                
                # Get waveform
                y, sr = librosa.load(wav_file, sr=16_000)
                
                row_dict = self.metadata_df[self.metadata_df['filename'] == filename].iloc[0].to_dict()
                sample_hypercategory = self.hypercategory_mapping[row_dict['category']]
                
                self.items.append({
                    "filename": wav_file,
                    "class": row_dict['category'],
                    "fold": row_dict['fold'],
                    "hypercategory": sample_hypercategory,
                    "label": ESC_CLASS_MAPPING[sample_hypercategory],
                    "waveform": y
                })
    
    def __getitem__(self, idx):
        return self.items[idx]
    
    def __len__(self):
        return len(self.items)

    def parse_hypercategory_mapping(self, json_file: os.PathLike):
        """Parse hypercategory mapping as dict.
        
        Args:
            json_file (os.PathLike): Abs Path to hypercategory mapping
        Returns:
            Dict[str, str]
        """

        with open(json_file, "r") as f:
            hypercategory_mapping = json.load(f)
        return hypercategory_mapping
