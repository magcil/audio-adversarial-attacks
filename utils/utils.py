import os
import math
import numpy as np

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

def add_normalized_noise(y: np.ndarray, y_noise: np.ndarray, SNR: float) -> np.ndarray:
    """Apply the background noise y_noise to y with a given SNR
    
    Args:
        y (np.ndarray): The original signal
        y_noise (np.ndarray): The noisy signal
        SNR (float): Signal to Noise ratio (in dB)
        
    Returns:
        np.ndarray: The original signal with the noise added.
    """
    if y.size < y_noise.size:
        y_noise = y_noise[:y.size]
    else:
        y_noise = np.resize(y_noise, y.shape)
    snr = 10**(SNR / 10)
    E_y, E_n = np.sum(y**2), np.sum(y_noise**2)

    z = np.sqrt((E_n / E_y) * snr) * y + y_noise

    return z / z.max()


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