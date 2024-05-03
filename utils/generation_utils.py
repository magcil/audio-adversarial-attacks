import numpy as np

#---- Functions for Generation of New particles----#

# Function to perform crossover between two parents
def crossover(parent1, parent2):
    
    # Choose a random crossover point
    crossover_point = np.random.randint(1, len(parent1) - 1)
    
    # Create two children by combining the parents' vectors
    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    
    return child1, child2

# ------------------------------------ Mutation Function ------------------------------------

# Function to perform mutation on an individual
def mutate(individual, mutation_rate, perturbation_ratio, target_wav):
    mutated_individual = individual.copy()

    raw_audio = target_wav
    noise_range = perturbation_ratio * np.abs(raw_audio)

    for i in range(len(individual)):
        # mutation rate: hyperparameter that determines the probability of adding mutation
        if np.random.rand() < mutation_rate:
            # Add a small random value to the element if the random number is less than the mutation rate

            mutated_individual[i] += np.random.uniform(-noise_range[i], noise_range[i])
    return mutated_individual

# ------------------------------------ Random Selection ------------------------------------

def selection(start_range, end_range, num_of_particles):
    # Generate num_of_particles distinct random integers from the specified range
    random_integers = np.random.randint(start_range, end_range, size=num_of_particles)
    
    # Ensure that all selected integers are distinct
    while len(set(random_integers)) < num_of_particles:
        random_integers = np.random.randint(start_range, end_range, size=num_of_particles)

    return random_integers