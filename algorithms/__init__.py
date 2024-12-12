# Default Hyperparameters for PSO & DE these will be used if not specified

default_pso_hyperparams = {
    "initial_particles": 35,
    "max_iters": 20,
    "max_inertia_w": 0.9,
    "min_inertia_w": 0.1,
    "memory_w": 0.3,
    "information_w": 1.2,
    "perturbation_ratio": 0.5
}

default_de_hyperparams = {
    "target_class": None,
    "rangeOfBounds": 0.01,
    "pop_size": 15,
    "iter": 10,
    "F": 1.2,
    "cr": 0.9,
    "λ": 0.005,
    "perturbation_ratio": 0.5
}
