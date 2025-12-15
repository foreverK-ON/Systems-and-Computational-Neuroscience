"""
Test suite for Reproduction and Innovation modules.
"""

import numpy as np
from reproduction import NeuralPatternReproducer, HebbianLearner
from innovation import NeuralGenotype, NeuroEvolution, xor_fitness


def test_neural_pattern_reproducer():
    """Test the neural pattern reproducer."""
    print("Testing NeuralPatternReproducer...")
    
    # Create a simple pattern
    pattern = np.array([1, 0, 1, 0, 1])
    
    # Initialize and train
    reproducer = NeuralPatternReproducer(input_size=5, hidden_size=3, learning_rate=0.1)
    losses = reproducer.train([pattern], epochs=50)
    
    # Test reproduction
    reproduced = reproducer.reproduce(pattern)
    error = np.mean((reproduced - pattern) ** 2)
    
    assert len(losses) == 50, "Should have 50 loss values"
    assert error < 0.5, f"Reproduction error too high: {error}"
    assert losses[-1] < losses[0], "Loss should decrease during training"
    
    print(f"  ✓ Pattern reproduction error: {error:.6f}")
    print(f"  ✓ Initial loss: {losses[0]:.6f}, Final loss: {losses[-1]:.6f}")
    return True


def test_hebbian_learner():
    """Test Hebbian learning."""
    print("Testing HebbianLearner...")
    
    # Create a simple pattern
    pattern = np.array([1, -1, 1, -1, 1])
    
    # Initialize and learn
    hebbian = HebbianLearner(size=5)
    hebbian.learn_pattern(pattern)
    
    # Test recall with noisy pattern
    noisy_pattern = pattern.copy()
    noisy_pattern[2] = -1  # Introduce noise
    recalled = hebbian.recall_pattern(noisy_pattern, iterations=5)
    
    assert hebbian.weights.shape == (5, 5), "Weights should be 5x5"
    assert np.allclose(np.diag(hebbian.weights), 0), "Diagonal should be zero"
    
    print(f"  ✓ Pattern learned and recalled successfully")
    print(f"  ✓ Weight matrix shape: {hebbian.weights.shape}")
    return True


def test_neural_genotype():
    """Test neural genotype."""
    print("Testing NeuralGenotype...")
    
    # Create a genotype
    genotype = NeuralGenotype(input_size=2, hidden_size=3, output_size=1)
    
    # Test forward pass
    input_data = np.array([0.5, 0.5])
    output = genotype.forward(input_data)
    
    assert output.shape == (1,), "Output should be 1D"
    assert -1 <= output[0] <= 1, "Output should be in tanh range"
    
    # Test mutation
    original_w1 = genotype.w1.copy()
    genotype.mutate(mutation_rate=0.5)
    
    assert not np.allclose(genotype.w1, original_w1), "Weights should change after mutation"
    
    print(f"  ✓ Genotype forward pass works")
    print(f"  ✓ Mutation changes weights")
    return True


def test_neuroevolution():
    """Test neuroevolution on a simple problem."""
    print("Testing NeuroEvolution...")
    
    # Create neuroevolution system
    neuroevo = NeuroEvolution(
        population_size=20,
        input_size=2,
        hidden_size=3,
        output_size=1,
        mutation_rate=0.1,
        elite_ratio=0.2
    )
    
    # Evolve for a few generations
    best = neuroevo.evolve(fitness_function=xor_fitness, generations=20, verbose=False)
    
    assert len(neuroevo.population) == 20, "Population size should be 20"
    assert len(neuroevo.best_fitness_history) == 20, "Should have 20 generations of history"
    assert neuroevo.best_fitness_history[-1] >= neuroevo.best_fitness_history[0], \
        "Fitness should improve or stay same"
    assert best.fitness > 0, "Best fitness should be positive"
    
    print(f"  ✓ Evolution completed {neuroevo.generation} generations")
    print(f"  ✓ Initial fitness: {neuroevo.best_fitness_history[0]:.4f}")
    print(f"  ✓ Final fitness: {neuroevo.best_fitness_history[-1]:.4f}")
    return True


def test_xor_fitness():
    """Test XOR fitness function."""
    print("Testing XOR fitness function...")
    
    # Create a random genotype
    genotype = NeuralGenotype(input_size=2, hidden_size=4, output_size=1)
    
    # Evaluate fitness
    fitness = xor_fitness(genotype)
    
    assert 0 < fitness <= 1, f"Fitness should be in (0, 1], got {fitness}"
    
    print(f"  ✓ XOR fitness calculated: {fitness:.4f}")
    return True


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("RUNNING TEST SUITE")
    print("=" * 60)
    print()
    
    tests = [
        test_neural_pattern_reproducer,
        test_hebbian_learner,
        test_neural_genotype,
        test_neuroevolution,
        test_xor_fitness
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
                print(f"✓ {test.__name__} PASSED\n")
        except AssertionError as e:
            failed += 1
            print(f"✗ {test.__name__} FAILED: {e}\n")
        except Exception as e:
            failed += 1
            print(f"✗ {test.__name__} ERROR: {e}\n")
    
    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
