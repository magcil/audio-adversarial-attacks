# audio-adversarial-examples

Repository for the paper titled:

<i>On the Robustness of State-of-the-Art Transformers for Sound Event Classification against Black Box Adversarial Attacks</i>


## Table of Contents
- [audio-adversarial-examples](#audio-adversarial-examples)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [1. Environment Setup](#1-environment-setup)
  - [2. Generating audio adversarial examples](#2-generating-audio-adversarial-examples)
    - [2.1 Model Initialization](#21-model-initialization)
    - [2.2 Noise Control](#22-noise-control)
    - [2.3 Particle Swarm Optimization](#23-particle-swarm-optimization)
    - [2.4 Differential Evolution](#24-differential-evolution)
    - [2.5 Inspecting the adversarial example](#25-inspecting-the-adversarial-example)
  - [3. Reproducing the Experiments](#3-reproducing-the-experiments)
    - [3.1 Experiments on AudioSet](#31-experiments-on-audioset)
    - [3.2 Experiments on ESC-50:](#32-experiments-on-esc-50)
  - [References](#references)


## Overview
This project represents an effort to evaluate the robustness of state-of-the-art transformer-based models for sound event classification against adversarial attacks. The attacks are performed using two evolutionary algorithms: Particle Swarm Optimization (PSO) and Differential Evolution (DE). We conduct experiments utilizing three deep learning models (BEATs, PaSST, AST) and two benchmark datasets (AudioSet, ESC-50).

## 1. Environment Setup

To reproduce the experiments or to perform attacks using any of the algorithms, first create a conda environment with python 3.9 by typing
```bash
conda create -n adversarial_audio python=3.9
```
Then activate the environment
```bash
conda activate adversarial_audio
```
and install the requirements
```bash
pip install -r requirements.txt
```

**Note**:
Due to dependency conflicts between some packages, this project uses two separate conda environments. The instructions above set up an environment for running experiments and adversarial attacks on BEATs or PaSST. If
you want to run any AST-related scripts you need to install requirements as:

```bash
pip install -r requirements_ast.txt
```

Now you ready to go!

## 2. Generating audio adversarial examples


We use the pre-trained models to generate audio adversarial attacks by utilizing two optimization algoriths: Particle Swarm Optimization [4], and Differential Evolution [5]. We operate in a black-box setting where the architecture and weights of the model are unknown to the attacker.

### 2.1 Model Initialization

To conduct experiments utilizing the **BEATs** model, please follow these steps:

1. <b>Download Model Weights: </b>Acquire the model weights that have been fine-tuned on the Audioset. These weights can be downloaded from the following [link](https://github.com/microsoft/unilm/tree/master/beats). In our scenario, we conduct experiments using the <b>Fine-tuned BEATs_iter3+ (AS2M) (cpt2)</b> pt file. To conduct experiments with various weights, you must configure the path to the appropriate file within the get_model function, as well as ensure that all corresponding configuration files are properly set for each specific case.

2. <b> Add Weights to Pretrained Models Folder:</b> After downloading, place the weights into the <b>'pretrained_models directory'</b> within your project.


In order to utilize **AST** model for attacks, you need to proceed with the following steps:
1. <b>Download Model Weights: </b> Obtain the model weights for AudioSet from [link](https://github.com/YuanGongND/ast/tree/master/pretrained_models). In our demonstrations, we use the  
<b>Full AudioSet, 10 tstride, 10 fstride, without Weight Averaging, Model 3 (0.448 mAP)</b>.

2. <b> Add Weights to Pretrained Models Folder:</b> After downloading, place the weights into the <b>'pretrained_models directory'</b> within your project.


To run experiments using the **PaSST** model, you do not have to download any weights they are already installed and automatically downloaded.


### 2.2 Noise Control

**Perturbation Ratio** To regulate the amount of perturbation added, it is necessary to adjust the perturbation ratio parameter within the algorithm's parameters dictionary. The perturbation ratio serves as a weight used in our noise initialization method.

**SNR Control** To generate attacks with a fixed Signal-to-Noise-Ratio (SNR), you need add the desired SNR values to the SNR_norm parameter in the form of a list. In this way, the generated adversarial example will have the specified signal-to-noise ratios.


### 2.3 Particle Swarm Optimization

To generate an adversarial example using PSO you'll need to first initialize the class responsible for making the attack. For example, to produce an adversarial example for a given `example.wav` file:

```python
from utils.init_utils import init_algorithm, get_model

# Define Algorithm Parameters.
algorithm_hyperparameters = {
    "initial_particles": 25,
    "max_iters": 15, 
    "max_inertia_w": 0.9, 
    "min_inertia_w": 0.1, 
    "memory_w": 1.2, 
    "information_w": 1.2, 
    "perturbation_ratio": 0.5}

# Define hypercategory mapping path.
hypercategory_mapping = "ontologies/hypercategory_from_ontology.json"

# Load the pre-trained model.
model = get_model(model_str="beats",
                  model_pt_file="pretrained_models/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt",
                  hypercategory_mapping=hypercategory_mapping)

# Initialize PSO Attacker
PSO_ATTACKER = init_algorithm(algorithm="pso",
                              model=model,
                              verbosity=False,
                              SNR_norm= [5],
                              hyperparameters=algorithm_hyperparameters,
                              objective_function="simple_minimization_targeted")


# Start the attack / Generate adversarial example
attack_results = PSO_ATTACKER.generate_adversarial_example("example.wav")
```

The variable `attack_results` is a python dictionary, containing the keys: 

- `noise`: The waveform of the perturbation.

- `adversary`:  The waveform of the generated adversarial example.

- `raw_audio`:  The original waveform.

- `iterations`: Number of total iterations performed on the attack.

- `success`: If the attack succedeed.

- `queries`: Number of queries to the model.

- `inferred_class`: The inferred class.

- `Final Starting Class Confidence`: The confidence of the starting class.

- `Final Confidence`: Confidence of the inferred class.

- `starting_class`: The predicted class before attack.

### 2.4 Differential Evolution

In similar manner you can use the Differential Evolution as an optimization algorithm to generate an adversarial example:

```python
from utils.init_utils import init_algorithm, get_model

# Define Algorithm Parameters.
algorithm_hyperparameters = {
  "pop_size": 20,
  "iter": 10,
  "F": 1.2,
  "cr": 0.9, 
  "perturbation_ratio": 0.5}

# Define hypercategory mapping path.
hypercategory_mapping = "ontologies/hypercategory_from_ontology.json"

# Load the pre-trained model.
model = get_model(model_str="beats",
                  model_pt_file="pretrained_models/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt",
                  hypercategory_mapping=hypercategory_mapping)

# Initialize PSO Attacker
DE_ATTACKER = init_algorithm(algorithm="de",
                              model=model,
                              verbosity=False,
                              SNR_norm= [5],
                              hyperparameters=algorithm_hyperparameters,
                              objective_function="simple_minimization_targeted")


attack_results = DE_ATTACKER.generate_adversarial_example("example.wav")
```

**Reminder**:
If you want to run attack on the AST model, you need to install other dependencies.

### 2.5 Inspecting the adversarial example

To hear the generated example you can use `soundfile` to store the wav file:

```python
import soundfile as sf

sf.write(file="adversary_example.wav", data=attack_results["adversary"], samplerate=16000, subtype="FLOAT")
```

## 3. Reproducing the Experiments

### 3.1 Experiments on AudioSet

To reproduce the experiments using the AudioSet dataset first download the validation subset of AudioSet from the following link: <a href="https://www.kaggle.com/datasets/zfturbo/audioset-valid">https://www.kaggle.com/datasets/zfturbo/audioset-valid</a>. Store all the wav files in a folder named `valid_wav` and place it inside `data` folder.

```bash
python src/run_attack.py --config_file config/attack_config.yaml
```

**Note**:
To run the experiments on audioset you need to create an ontology that maps .wav filenames to hypercategories.
This can be achieved by running the `create_subset_audioset.py` script. This script returns a json file containing the required format.

Parameters of script:
1. -hc : Hypercategory mapping found in `ontologies/hypercategory_from_ontology.json`
2. -tl : True labels ontology in `ontologies/audioset_val_true_labels.json`
3. -n : Number of desired samples.
4. -t : Target path to store the new ontology.

### 3.2 Experiments on ESC-50:

To run the experiments on the ESC-50 dataset, first download the dataset from <a href="https://www.kaggle.com/datasets/mmoreaux/environmental-sound-classification-50">https://www.kaggle.com/datasets/mmoreaux/environmental-sound-classification-50</a>. The models need to be finetuned on this dataset, thus run the script:

```bash
python src/finetuned_attack.py --config_file config/finetune.yaml
```

## References
[1] *<a href="https://arxiv.org/abs/2212.09058">BEATs: Audio Pre-Training with Acoustic Tokenizers</a>*

[2] *<a href="https://arxiv.org/abs/2104.01778"> AST: Audio Spectrogram Transformer</a>*

[3] *<a href="https://arxiv.org/abs/2110.05069"> Efficient Training of Audio Transformers with Patchout</a>*

[4] *<a href="https://link.springer.com/article/10.1023/A:1008202821328">Differential Evolution – A Simple and Efficient Heuristic for global Optimization over Continuous Spaces</a>*

[5] *<a href="https://link.springer.com/article/10.1007/s11831-021-09694-4">Particle Swarm Optimization Algorithm and Its Applications: A Systematic Review</a>*
