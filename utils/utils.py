import os
import math
import random
import wave

import numpy as np
import librosa
import pyaudio


# ----- Load Audio File -----
def is_audio_file(file_path):
    # Define a list of audio file extensions
    audio_extensions = ['.mp3', '.wav', '.ogg', '.flac', '.aac', '.wma']

    # Get the file extension from the file path
    file_extension = file_path[file_path.rfind('.'):].lower()

    # Check if the file extension is in the list of audio extensions
    if file_extension in audio_extensions:
        return True
    else:
        return False


def load_wav_16k_mono(file_path):
    """Load a wav file as mono and 16K sample rate

    Attributes:
    file_path -- The target file path
    """

    waveform, _ = librosa.load(file_path, sr=16000, mono=True)

    return waveform


# ----- Constraints -----
# TODO: Add L0 constraints.
# Apply L0 norm  constraints
def apply_l0_norm_constraint(audio, k):
    vector = []
    # Get indices of k largest elements in magnitude
    indices = np.argsort(np.abs(audio))[-k:]

    # Set all elements to zero
    vector[:] = 0

    # Set only the k selected elements to their original values
    vector[indices] = audio[indices]

    return vector


# ---- Files Management ----


def sample_random_file(files_dir):
    """Sample random file from test audio files

    Attributes:
    files_dir -- Path of files
    """

    wav_files = [f for f in os.listdir(files_dir) if f.endswith('.wav')]
    random_wav_name = random.choice(wav_files)
    random_wav_path = os.path.join(files_dir, random_wav_name)

    return random_wav_path


# ----- Imperceptibility Evaluation -----


def calculate_snr(signal, noise):
    """Method to measure SNR of signal
    
    Attributes:
    raw_signal -- raw audio
    perturbation -- perturbation
    """

    # Calculate power of signal and noise
    power_signal = np.mean(signal**2)
    power_noise = np.mean(noise**2)

    # Calculate SNR in decibels
    snr = 10 * np.log10(power_signal / power_noise)

    return snr


def calculate_euclidean_distance(vector1, vector2):
    """
        Method to calculate L2 distance
    """
    if len(vector1) != len(vector2):
        raise ValueError("Vectors must have the same length")

    distance = np.linalg.norm(vector1 - vector2)

    return distance


def generate_white_noise(waveform, perturbation_ratio, bound):
    """
        Generate white noise.
    
    Attributes:
        waveform -- Raw waveform.
        perturbation_ratio -- The ratio of added noise.
        bound -- L infinity norm constraint.
    """

    # noise_range = perturbation_ratio * np.abs(target_waveform)
    perturbation = np.random.uniform(-bound, bound, len(waveform))

    noise = perturbation_ratio * perturbation
    return noise


def generate_bounded_white_noise(target_waveform, perturbation_ratio):
    """
        Generate bounded white noise that follows the distribution of the original waveform.
    
    Attributes:
        target_waveform -- Raw waveform.
        perturbation_ratio -- The ratio of added noise.
    """

    noise_range = perturbation_ratio * np.abs(target_waveform)
    perturbation = np.random.uniform(-noise_range, noise_range, len(target_waveform))

    return perturbation


# Generate White Noise based on SNR
def SNR_based_white_noise(signal, SNR):
    """
        Generate noise based on a certain SNR level.

        Attributes:
            signal : original waveform.
            SNR : Desired SNR value.
    """

    RMS_s = math.sqrt(np.mean(signal**2))

    RMS_n = math.sqrt(RMS_s**2 / (pow(10, SNR / 10)))
    #Additive white gausian noise. Thereore mean=0
    #Because sample length is large (typically > 40000)
    #we can use the population formula for standard daviation.
    #because mean=0 STD=RMS
    STD_n = RMS_n
    noise = np.random.normal(0, STD_n, signal.shape[0])
    return noise


def crawl_directory(directory: str, extension: str = None, num_files: int = 0) -> list:
    """Crawling data directory
    Args:
        directory (str) : The directory to crawl
        extension (str) : file extension to look for
        num_files (int) : number of files to return
    Returns:
        tree (list)     : A list with all the filepaths
    """
    tree = []
    subdirs = [folder[0] for folder in os.walk(directory)]

    for subdir in subdirs:
        files = next(os.walk(subdir))[2]
        for _file in files:
            if extension is not None:
                if _file.endswith(extension):
                    tree.append(os.path.join(subdir, _file))
            else:
                tree.append(os.path.join(subdir, _file))
            if 0 < num_files <= len(tree):
                break
    return tree


def play_audio(audio_file: str):
    """Play an audio file
    Args:
        audio_file (str): The full path to the audio wav file
    """
    CHUNK_SIZE = 1024

    with wave.open(audio_file, "rb") as wf:
        
        p = pyaudio.PyAudio()
        
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)
        data = wf.readframes(CHUNK_SIZE)

        while data:
            stream.write(data)
            data = wf.readframes(CHUNK_SIZE)
        
        stream.close()
        p.terminate()
