"""
Reproduction Module: Neural Pattern Replication

This module implements neural pattern reproduction mechanisms,
demonstrating how neural networks can learn and replicate patterns.
"""

import numpy as np
from typing import List, Tuple


class NeuralPatternReproducer:
    """
    A simple neural network that learns to reproduce input patterns.
    Uses Hebbian learning principles: neurons that fire together, wire together.
    """
    
    def __init__(self, input_size: int, hidden_size: int = 10, learning_rate: float = 0.01):
        """
        Initialize the neural pattern reproducer.
        
        Args:
            input_size: Dimension of input patterns
            hidden_size: Number of hidden neurons
            learning_rate: Learning rate for pattern reproduction
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        
        # Initialize weights with small random values
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.1
        self.weights_hidden_output = np.random.randn(hidden_size, input_size) * 0.1
        
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through the network.
        
        Args:
            x: Input pattern
            
        Returns:
            Tuple of (hidden activations, output pattern)
        """
        hidden = self.sigmoid(np.dot(x, self.weights_input_hidden))
        output = self.sigmoid(np.dot(hidden, self.weights_hidden_output))
        return hidden, output
    
    def train(self, patterns: List[np.ndarray], epochs: int = 100) -> List[float]:
        """
        Train the network to reproduce patterns using backpropagation.
        
        Args:
            patterns: List of patterns to learn
            epochs: Number of training epochs
            
        Returns:
            List of loss values over epochs
        """
        losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for pattern in patterns:
                # Forward pass
                hidden, output = self.forward(pattern)
                
                # Calculate loss (mean squared error)
                loss = np.mean((output - pattern) ** 2)
                epoch_loss += loss
                
                # Backpropagation
                output_error = output - pattern
                hidden_error = np.dot(output_error, self.weights_hidden_output.T) * hidden * (1 - hidden)
                
                # Update weights
                self.weights_hidden_output -= self.learning_rate * np.outer(hidden, output_error)
                self.weights_input_hidden -= self.learning_rate * np.outer(pattern, hidden_error)
            
            avg_loss = epoch_loss / len(patterns)
            losses.append(avg_loss)
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
        
        return losses
    
    def reproduce(self, pattern: np.ndarray) -> np.ndarray:
        """
        Reproduce a learned pattern.
        
        Args:
            pattern: Input pattern
            
        Returns:
            Reproduced pattern
        """
        _, output = self.forward(pattern)
        return output


class HebbianLearner:
    """
    Implements Hebbian learning: "Cells that fire together, wire together."
    Demonstrates basic synaptic plasticity and pattern association.
    """
    
    def __init__(self, size: int):
        """
        Initialize Hebbian learner.
        
        Args:
            size: Size of the network (number of neurons)
        """
        self.size = size
        self.weights = np.zeros((size, size))
        
    def learn_pattern(self, pattern: np.ndarray, learning_rate: float = 0.1):
        """
        Learn a pattern using Hebbian rule.
        
        Args:
            pattern: Binary or continuous pattern
            learning_rate: Hebbian learning rate
        """
        # Hebbian update: Δw_ij = η * x_i * x_j
        pattern = pattern.reshape(-1, 1)
        self.weights += learning_rate * np.dot(pattern, pattern.T)
        np.fill_diagonal(self.weights, 0)  # No self-connections
        
    def recall_pattern(self, partial_pattern: np.ndarray, iterations: int = 10) -> np.ndarray:
        """
        Recall a complete pattern from partial input.
        
        Args:
            partial_pattern: Partial or noisy version of learned pattern
            iterations: Number of recall iterations
            
        Returns:
            Recalled pattern
        """
        pattern = partial_pattern.copy()
        
        for _ in range(iterations):
            activation = np.dot(self.weights, pattern)
            pattern = np.sign(activation)
            pattern[pattern == 0] = 1  # Handle zero values
            
        return pattern


def demonstrate_reproduction():
    """Demonstrate neural pattern reproduction."""
    print("=" * 60)
    print("NEURAL PATTERN REPRODUCTION DEMONSTRATION")
    print("=" * 60)
    
    # Create simple patterns (3x3 binary images)
    patterns = [
        np.array([1, 1, 1, 0, 0, 0, 0, 0, 0]),  # Horizontal line
        np.array([1, 0, 0, 1, 0, 0, 1, 0, 0]),  # Vertical line
        np.array([1, 0, 1, 0, 1, 0, 1, 0, 1]),  # Diagonal pattern
    ]
    
    print("\nTraining neural network to reproduce patterns...")
    reproducer = NeuralPatternReproducer(input_size=9, hidden_size=5, learning_rate=0.1)
    losses = reproducer.train(patterns, epochs=100)
    
    print("\nTesting pattern reproduction:")
    for i, pattern in enumerate(patterns):
        reproduced = reproducer.reproduce(pattern)
        print(f"\nPattern {i + 1}:")
        print(f"Original:    {pattern}")
        print(f"Reproduced:  {reproduced.round(2)}")
        print(f"Error:       {np.mean((reproduced - pattern) ** 2):.6f}")
    
    # Demonstrate Hebbian learning
    print("\n" + "=" * 60)
    print("HEBBIAN LEARNING DEMONSTRATION")
    print("=" * 60)
    
    hebbian = HebbianLearner(size=9)
    
    # Learn patterns
    for pattern in patterns:
        binary_pattern = np.where(pattern > 0.5, 1, -1)
        hebbian.learn_pattern(binary_pattern)
    
    # Test recall with noisy pattern
    print("\nRecalling patterns from noisy input...")
    test_pattern = np.array([1, 1, 1, 0, -1, 0, -1, 0, 0])  # Noisy version of pattern 1
    recalled = hebbian.recall_pattern(test_pattern)
    
    print(f"Noisy input:  {test_pattern}")
    print(f"Recalled:     {recalled}")
    
    return losses


if __name__ == "__main__":
    demonstrate_reproduction()
