"""
Innovation Module: Evolutionary Neural Networks

This module implements innovation mechanisms through evolutionary algorithms,
demonstrating how neural networks can evolve and adapt to find novel solutions.
"""

import numpy as np
from typing import List, Tuple, Callable
import copy


class NeuralGenotype:
    """
    Represents the genetic encoding of a neural network.
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Initialize a neural network genotype.
        
        Args:
            input_size: Number of input neurons
            hidden_size: Number of hidden neurons
            output_size: Number of output neurons
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights (genes)
        self.w1 = np.random.randn(input_size, hidden_size) * 0.5
        self.w2 = np.random.randn(hidden_size, output_size) * 0.5
        self.fitness = 0.0
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the network.
        
        Args:
            x: Input array
            
        Returns:
            Output array
        """
        hidden = np.tanh(np.dot(x, self.w1))
        output = np.tanh(np.dot(hidden, self.w2))
        return output
    
    def mutate(self, mutation_rate: float = 0.1, mutation_strength: float = 0.3):
        """
        Mutate the network weights (introduce innovation).
        
        Args:
            mutation_rate: Probability of mutating each weight
            mutation_strength: Magnitude of mutations
        """
        # Mutate first layer weights
        mask1 = np.random.random(self.w1.shape) < mutation_rate
        self.w1[mask1] += np.random.randn(np.sum(mask1)) * mutation_strength
        
        # Mutate second layer weights
        mask2 = np.random.random(self.w2.shape) < mutation_rate
        self.w2[mask2] += np.random.randn(np.sum(mask2)) * mutation_strength
        
    def copy(self) -> 'NeuralGenotype':
        """Create a deep copy of this genotype."""
        return copy.deepcopy(self)


class NeuroEvolution:
    """
    Implements neuroevolution: evolution of neural networks.
    Demonstrates how innovation emerges through selection and mutation.
    """
    
    def __init__(
        self,
        population_size: int,
        input_size: int,
        hidden_size: int,
        output_size: int,
        mutation_rate: float = 0.1,
        elite_ratio: float = 0.2
    ):
        """
        Initialize neuroevolution system.
        
        Args:
            population_size: Number of individuals in population
            input_size: Input dimension
            hidden_size: Hidden layer size
            output_size: Output dimension
            mutation_rate: Rate of mutation
            elite_ratio: Proportion of top performers to keep
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.elite_count = max(1, int(population_size * elite_ratio))
        
        # Initialize population
        self.population = [
            NeuralGenotype(input_size, hidden_size, output_size)
            for _ in range(population_size)
        ]
        
        self.generation = 0
        self.best_fitness_history = []
        self.avg_fitness_history = []
        
    def evaluate_fitness(self, fitness_function: Callable[[NeuralGenotype], float]):
        """
        Evaluate fitness of all individuals in population.
        
        Args:
            fitness_function: Function that evaluates a genotype and returns fitness
        """
        for individual in self.population:
            individual.fitness = fitness_function(individual)
            
    def select_and_reproduce(self):
        """
        Select best individuals and create next generation through reproduction and mutation.
        This is where innovation happens!
        """
        # Sort by fitness (descending)
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Track statistics
        best_fitness = self.population[0].fitness
        avg_fitness = np.mean([ind.fitness for ind in self.population])
        self.best_fitness_history.append(best_fitness)
        self.avg_fitness_history.append(avg_fitness)
        
        # Keep elite individuals (reproduction without change)
        next_generation = [ind.copy() for ind in self.population[:self.elite_count]]
        
        # Create offspring through mutation (innovation)
        while len(next_generation) < self.population_size:
            # Tournament selection: pick best of random subset
            tournament = np.random.choice(self.population[:self.population_size//2], size=2, replace=False)
            parent = max(tournament, key=lambda x: x.fitness)
            
            # Create offspring with mutation
            offspring = parent.copy()
            offspring.mutate(mutation_rate=self.mutation_rate)
            next_generation.append(offspring)
        
        self.population = next_generation
        self.generation += 1
        
    def evolve(
        self,
        fitness_function: Callable[[NeuralGenotype], float],
        generations: int,
        verbose: bool = True
    ) -> NeuralGenotype:
        """
        Evolve population for specified number of generations.
        
        Args:
            fitness_function: Function to evaluate fitness
            generations: Number of generations to evolve
            verbose: Whether to print progress
            
        Returns:
            Best individual from final generation
        """
        for gen in range(generations):
            self.evaluate_fitness(fitness_function)
            self.select_and_reproduce()
            
            if verbose and gen % 10 == 0:
                print(f"Generation {gen}: Best Fitness = {self.best_fitness_history[-1]:.4f}, "
                      f"Avg Fitness = {self.avg_fitness_history[-1]:.4f}")
        
        # Final evaluation
        self.evaluate_fitness(fitness_function)
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        return self.population[0]


def xor_fitness(individual: NeuralGenotype) -> float:
    """
    Fitness function for XOR problem.
    XOR is a classic problem requiring innovation to solve.
    """
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    targets = np.array([[0], [1], [1], [0]])
    
    total_error = 0.0
    for x, y in zip(inputs, targets):
        output = individual.forward(x)
        error = (output[0] - y[0]) ** 2
        total_error += error
    
    # Fitness is inverse of error (higher is better)
    fitness = 1.0 / (1.0 + total_error)
    return fitness


def target_function_fitness(individual: NeuralGenotype, target_func: Callable) -> float:
    """
    Fitness function for approximating a target function.
    
    Args:
        individual: Neural network to evaluate
        target_func: Target function to approximate
        
    Returns:
        Fitness score
    """
    # Test on random points
    n_samples = 20
    x_vals = np.random.uniform(-1, 1, (n_samples, individual.input_size))
    
    total_error = 0.0
    for x in x_vals:
        output = individual.forward(x)[0]
        target = target_func(x)
        error = (output - target) ** 2
        total_error += error
    
    fitness = 1.0 / (1.0 + total_error / n_samples)
    return fitness


def demonstrate_innovation():
    """Demonstrate evolutionary innovation in neural networks."""
    print("=" * 60)
    print("NEUROEVOLUTION: SOLVING XOR THROUGH INNOVATION")
    print("=" * 60)
    print("\nXOR is a classic problem that requires innovation to solve.")
    print("Networks must discover the hidden representation through evolution.\n")
    
    # Create neuroevolution system
    neuroevo = NeuroEvolution(
        population_size=50,
        input_size=2,
        hidden_size=4,
        output_size=1,
        mutation_rate=0.15,
        elite_ratio=0.2
    )
    
    # Evolve to solve XOR
    print("Evolving neural networks...")
    best_individual = neuroevo.evolve(
        fitness_function=xor_fitness,
        generations=100,
        verbose=True
    )
    
    print(f"\nEvolution complete! Best fitness: {best_individual.fitness:.4f}")
    
    # Test the best network
    print("\nTesting best evolved network on XOR:")
    test_cases = [
        ([0, 0], 0),
        ([0, 1], 1),
        ([1, 0], 1),
        ([1, 1], 0)
    ]
    
    for inputs, expected in test_cases:
        output = best_individual.forward(np.array(inputs))[0]
        print(f"Input: {inputs} -> Output: {output:.4f} (Expected: {expected})")
    
    print("\n" + "=" * 60)
    print("INNOVATION THROUGH MUTATION")
    print("=" * 60)
    print("\nDemonstrating how mutation introduces innovation...")
    
    # Show effect of mutation
    original = NeuralGenotype(2, 4, 1)
    mutated = original.copy()
    
    print("\nOriginal network output:")
    for inputs, _ in test_cases:
        output = original.forward(np.array(inputs))[0]
        print(f"Input: {inputs} -> Output: {output:.4f}")
    
    # Apply several mutations
    for i in range(5):
        mutated.mutate(mutation_rate=0.3)
    
    print("\nAfter 5 mutations (innovation):")
    for inputs, _ in test_cases:
        output = mutated.forward(np.array(inputs))[0]
        print(f"Input: {inputs} -> Output: {output:.4f}")
    
    return neuroevo


def demonstrate_function_approximation():
    """Demonstrate innovation in learning complex functions."""
    print("\n" + "=" * 60)
    print("EVOLVING NETWORKS TO APPROXIMATE COMPLEX FUNCTIONS")
    print("=" * 60)
    
    # Define a complex target function
    def target_func(x):
        return np.sin(x[0] * np.pi) * np.cos(x[1] * np.pi)
    
    print("\nTarget function: f(x, y) = sin(πx) * cos(πy)")
    print("Networks must innovate to approximate this function...\n")
    
    neuroevo = NeuroEvolution(
        population_size=30,
        input_size=2,
        hidden_size=6,
        output_size=1,
        mutation_rate=0.1,
        elite_ratio=0.25
    )
    
    # Create fitness function with target
    def fitness_func(ind):
        return target_function_fitness(ind, target_func)
    
    best = neuroevo.evolve(fitness_func, generations=50, verbose=True)
    
    print(f"\nFinal best fitness: {best.fitness:.4f}")
    
    # Test on specific points
    print("\nTesting on sample points:")
    test_points = [
        [0.0, 0.0],
        [0.5, 0.5],
        [1.0, 0.0],
        [-0.5, 0.5]
    ]
    
    for point in test_points:
        predicted = best.forward(np.array(point))[0]
        actual = target_func(np.array(point))
        error = abs(predicted - actual)
        print(f"Point {point}: Predicted={predicted:.4f}, Actual={actual:.4f}, Error={error:.4f}")


if __name__ == "__main__":
    demonstrate_innovation()
    demonstrate_function_approximation()
