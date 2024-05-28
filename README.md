# audio-adversarial-examples
Black Box Adversarial Attacks in Surveillance Sound Event Classification Scenarios


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

Now you ready to go!

## 2. Generating audio adversarial examples

The model under attack is BEATs [1], a transformer-based deep neural network for detecting audio events. We use this model to generate audio adversarial attacks by utilizing two optimization algoriths: Particle Swarm Optimization [2], and Differential Evolution [3]. We operate in a black-box setting where the architecture and weights of the model are unknown to the attacker. To run the experiments with the BEATs model, download the weights of the pretrained model [here](https://github.com/microsoft/unilm/tree/master/beats) and add them on the pretrained_models folder.

### 2.1 Particle Swarm Optimization

To generate an adversarial example using PSO you'll need to first initialize the class responsible for making the attack. For example, to generate an adversarial example that misclassifies the `example.wav` as "Silence" use:

```python
import json

from utils.attack_utils import init_algorithm, get_model

# PSO Hyperparameters
algorithm_hyperparameters = {
    "initial_particles": 35,
    "additional_particles": 0,
    "max_iters": 10,
    "max_inertia_w": 0.9,
    "min_inertia_w": 0.1,
    "memory_w": 0.3,
    "information_w": 1.2,
    "perturbation_ratio": 0.01,
    "enable_particle_generation": False,
    "enabled_early_stopping": False
}

# Load the hypercategory mapping
with open("ontologies/hypercategory_from_ontology.json", "r") as f:
    hypercategory_mapping = json.load(f)

# Load the model under attack
model = get_model(model_str="beats",
                  model_pt_file="pretrained_models/BEATs_iter3_plus_AS2M.pt",
                  hypercategory_mapping=d)

# Initialize PSO Attacker
PSO_ATTACKER = init_algorithm(algorithm="pso",
                              model=model,
                              hyperparameters=algorithm_hyperparameters,
                              objective_function="simple_minimization_targeted",
                              target_class="Silence",
                              hypercategory_target=False,
                              verbosity=False)

# Start the attack / Generate adversarial example

attack_results = PSO_ATTACKER.generate_adversarial_example("example.wav")
```

The variable `attack_results` is a python dictionary, containing the keys: 

- `noise`: The waveform of the perturbation.

- `adversary`:  The waveform of the generated adversarial example.

- `iteration`: Number of total iterations performed on the attack.

- `success`: If the attack succedeed.

- `queries`: Number of queries to the model.

- `inferred_class`: The inferred class.

- `Final Starting Class Confidence`: The confidence of the starting class.

- `Final Confidence`: Confidence of the inferred class.

### 2.2 Differential Evolution

In similar manner you can use the Differential Evolution as an optimization algorithm to generate an adversarial example:

```python
import json

from utils.attack_utils import init_algorithm, get_model

# Load the hypercategory mapping
with open("ontologies/hypercategory_from_ontology.json", "r") as f:
    hypercategory_mapping = json.load(f)

# DE hyperparameters
de_hyperparameters = {
    "pop_size": 10,
    "iter": 10,  # Number of iterations
    "F": 1.2,  # Mutation Rate
    "cr": 0.9,  # Crossover Rate
    "λ": 0.005,  # Regularisation Parameter weight
    "perturbation_ratio": 0.01,
    "rangeOfBounds":0.01
}

# Initialize model
model = get_model(model_str="beats",
                  model_pt_file="pretrained_models/BEATs_iter3_plus_AS2M.pt",
                  hypercategory_mapping=d)

DE_ATTACKER = init_algorithm(algorithm="de",
                             model=model,
                             hyperparameters=de_hyperparameters,
                             verbosity=False,
                             objective_function="simple_minimization_targeted",
                             target_class="Silence",
                             hypercategory_target=False)

attack_results = DE_ATTACKER.generate_adversarial_example("example.wav")
```

### 2.3 Inspecting the adversarial example

To hear the generated example you can `soundfile` to store the wav file:

```python
import soundfile as sf

sf.write(file="adversary_example.wav", data=attack_results["adversary"], samplerate=16000, subtype="FLOAT")
```

## 3. Reproducing the Experiments

To reproduce the experiments for the three scenarios first download the validation subset of AudioSet from the following link: <a href="https://www.kaggle.com/datasets/zfturbo/audioset-valid">https://www.kaggle.com/datasets/zfturbo/audioset-valid</a>. Store all the wav files in a folder named `valid_wav` and place it inside `data` folder.

### 3.1 Untargeted Scenario

To run the experiment for the untargeted scenario using PSO use the command

```bash
python src/run_attack.py --config_file config/pso_untargeted_config.json
```

To run the experiment using DE use the command

```bash
python src/run_attack.py --config_file config/de_untargeted_config.json
```

### 3.2 Alarming to non-Alarming

To run the experiment for the "alarming to non-alarming" case using PSO use the command

```bash
python src/run_attack.py --config_file config/pso_alerting_config.json
```

Similarly, for DE use

```bash
python src/run_attack.py --config_file config/de_alerting_config.json
```

### 3.3 Everything to Silence

To reproduce the experiments for the scenario "everything to silence" use

```bash
python src/run_attack.py --config_file config/pso_targeted_attack_config.json
```

To use DE use the command

```bash
python src/run_attack.py --config_file config/de_targeted_attack_config.json
```


## References
[1] *<a href="https://arxiv.org/abs/2212.09058">BEATs: Audio Pre-Training with Acoustic Tokenizers</a>*

[2] *<a href="https://link.springer.com/article/10.1023/A:1008202821328">Differential Evolution – A Simple and Efficient Heuristic for global Optimization over Continuous Spaces</a>*

[3] *<a href="https://link.springer.com/article/10.1007/s11831-021-09694-4">Particle Swarm Optimization Algorithm and Its Applications:
A Systematic Review</a>*
