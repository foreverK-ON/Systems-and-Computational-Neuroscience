"""
Main Demonstration: Reproduction and Innovation in Computational Neuroscience

This script demonstrates both reproduction (pattern replication) and 
innovation (evolutionary adaptation) in neural systems.
"""

import numpy as np
import matplotlib.pyplot as plt
from reproduction import demonstrate_reproduction
from innovation import demonstrate_innovation, demonstrate_function_approximation


def plot_evolution_progress(neuroevo):
    """Plot the progress of evolution over generations."""
    plt.figure(figsize=(10, 6))
    
    generations = range(len(neuroevo.best_fitness_history))
    plt.plot(generations, neuroevo.best_fitness_history, 'b-', label='Best Fitness', linewidth=2)
    plt.plot(generations, neuroevo.avg_fitness_history, 'r--', label='Average Fitness', linewidth=2)
    
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Fitness', fontsize=12)
    plt.title('Evolution of Neural Networks: Innovation Over Time', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('evolution_progress.png', dpi=150)
    print("\nEvolution progress plot saved as 'evolution_progress.png'")
    plt.close()


def plot_learning_curve(losses):
    """Plot the learning curve for reproduction."""
    plt.figure(figsize=(10, 6))
    
    plt.plot(losses, 'g-', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('Neural Pattern Reproduction: Learning Curve', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('reproduction_learning.png', dpi=150)
    print("\nReproduction learning curve saved as 'reproduction_learning.png'")
    plt.close()


def compare_reproduction_and_innovation():
    """
    Compare and contrast reproduction and innovation mechanisms.
    """
    print("\n" + "=" * 70)
    print(" " * 15 + "REPRODUCTION vs INNOVATION")
    print("=" * 70)
    
    print("\n┌─────────────────────┬──────────────────────┬──────────────────────┐")
    print("│ Aspect              │ Reproduction         │ Innovation           │")
    print("├─────────────────────┼──────────────────────┼──────────────────────┤")
    print("│ Goal                │ Preserve patterns    │ Create new solutions │")
    print("│ Mechanism           │ Learning & memory    │ Mutation & selection │")
    print("│ Biological basis    │ Synaptic plasticity  │ Genetic variation    │")
    print("│ Time scale          │ Fast (learning)      │ Slow (evolution)     │")
    print("│ Accuracy            │ High fidelity        │ Variable, exploratory│")
    print("│ Application         │ Pattern recognition  │ Optimization         │")
    print("└─────────────────────┴──────────────────────┴──────────────────────┘")
    
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print("""
1. REPRODUCTION ensures stability and faithful transmission of learned patterns
   - Enables organisms to preserve successful behaviors
   - Implements through Hebbian learning and backpropagation
   - Critical for memory and skill retention

2. INNOVATION drives adaptation and discovery of novel solutions
   - Enables populations to solve new problems
   - Implements through mutation, recombination, and selection
   - Critical for evolution and optimization

3. BALANCE between reproduction and innovation is crucial
   - Too much reproduction → lack of adaptability (overfitting)
   - Too much innovation → loss of proven solutions (instability)
   - Natural systems balance both through multi-level mechanisms
    """)


def main():
    """Main demonstration of reproduction and innovation."""
    print("=" * 70)
    print(" " * 10 + "REPRODUCTION AND INNOVATION IN")
    print(" " * 12 + "COMPUTATIONAL NEUROSCIENCE")
    print("=" * 70)
    print("\nThis demonstration explores two fundamental mechanisms in")
    print("biological and artificial neural systems:\n")
    print("1. REPRODUCTION - How patterns are learned and replicated")
    print("2. INNOVATION - How new solutions emerge through evolution\n")
    
    # Run reproduction demonstration
    print("\n" + "=" * 70)
    print("PART 1: REPRODUCTION")
    print("=" * 70)
    losses = demonstrate_reproduction()
    
    # Run innovation demonstrations
    print("\n\n" + "=" * 70)
    print("PART 2: INNOVATION")
    print("=" * 70)
    neuroevo = demonstrate_innovation()
    demonstrate_function_approximation()
    
    # Create visualizations
    print("\n\n" + "=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)
    try:
        plot_learning_curve(losses)
        plot_evolution_progress(neuroevo)
    except Exception as e:
        print(f"Note: Visualization skipped (matplotlib not available): {e}")
    
    # Compare mechanisms
    compare_reproduction_and_innovation()
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nKey files generated:")
    print("  - reproduction.py: Pattern replication mechanisms")
    print("  - innovation.py: Evolutionary algorithms")
    print("  - main.py: This integrated demonstration")
    if losses:
        print("  - reproduction_learning.png: Learning curves")
        print("  - evolution_progress.png: Evolution progress")
    print("\nThese implementations showcase how biological neural systems")
    print("balance preservation of learned patterns with discovery of")
    print("novel solutions - a fundamental principle in both neuroscience")
    print("and artificial intelligence.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
