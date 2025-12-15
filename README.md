# Systems and Computational Neuroscience
## Reproduction and Innovation

小组大作业 / Group Project

### Overview

This project explores two fundamental mechanisms in biological and artificial neural systems:

1. **Reproduction** - How neural systems learn and replicate patterns
2. **Innovation** - How neural systems evolve and discover novel solutions

### Project Structure

```
.
├── reproduction.py      # Neural pattern reproduction mechanisms
├── innovation.py        # Evolutionary neural network algorithms
├── main.py             # Integrated demonstration
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

### Key Concepts

#### Reproduction
- **Hebbian Learning**: "Neurons that fire together, wire together"
- **Pattern Replication**: Neural networks learning to reproduce input patterns
- **Synaptic Plasticity**: How connections strengthen based on activity
- **Memory Formation**: Storage and retrieval of learned patterns

#### Innovation
- **Neuroevolution**: Evolution of neural network structures and weights
- **Genetic Algorithms**: Selection, mutation, and reproduction
- **Fitness-Based Selection**: Survival of the most adapted solutions
- **Emergent Complexity**: How complex behaviors arise from simple rules

### Installation

```bash
# Clone the repository
git clone https://github.com/foreverK-ON/Systems-and-Computational-Neuroscience.git
cd Systems-and-Computational-Neuroscience

# Install dependencies
pip install -r requirements.txt
```

### Usage

Run the complete demonstration:
```bash
python main.py
```

Run individual modules:
```bash
# Reproduction mechanisms only
python reproduction.py

# Innovation mechanisms only
python innovation.py
```

### What the Code Demonstrates

#### 1. Reproduction Module (`reproduction.py`)
- **NeuralPatternReproducer**: A simple neural network using backpropagation to learn and reproduce patterns
- **HebbianLearner**: Implementation of Hebbian learning for pattern association
- Demonstrates how neural systems preserve and replicate learned information

#### 2. Innovation Module (`innovation.py`)
- **NeuralGenotype**: Genetic representation of neural networks
- **NeuroEvolution**: Evolutionary algorithm for neural network optimization
- Solves the XOR problem through evolution (a classic non-linear problem)
- Demonstrates function approximation through innovative adaptation

#### 3. Main Demonstration (`main.py`)
- Integrates both reproduction and innovation
- Creates visualizations of learning and evolution
- Compares and contrasts the two mechanisms
- Shows how biological systems balance both approaches

### Examples

#### Pattern Reproduction
The system learns to reproduce patterns like:
```
Original:    [1 1 1 0 0 0 0 0 0]  (horizontal line)
Reproduced:  [0.98 0.97 0.99 0.02 0.01 0.03 0.01 0.02 0.01]
```

#### Evolutionary Innovation
Networks evolve to solve XOR:
```
Generation 0:  Best Fitness = 0.3421
Generation 50: Best Fitness = 0.8932
Generation 100: Best Fitness = 0.9876

Final XOR results:
Input: [0, 0] -> Output: 0.0234 (Expected: 0)
Input: [0, 1] -> Output: 0.9821 (Expected: 1)
Input: [1, 0] -> Output: 0.9765 (Expected: 1)
Input: [1, 1] -> Output: 0.0189 (Expected: 0)
```

### Biological Relevance

This project illustrates key principles from neuroscience and evolutionary biology:

- **Learning (Reproduction)**: Similar to how synapses strengthen during learning
- **Evolution (Innovation)**: Similar to how populations adapt over generations
- **Balance**: Natural systems use both mechanisms at different timescales
  - Fast learning for within-lifetime adaptation
  - Slow evolution for cross-generational optimization

### Applications

These concepts apply to:
- **Artificial Intelligence**: Neural network training and architecture search
- **Optimization**: Solving complex problems through evolutionary algorithms
- **Robotics**: Adaptive control systems
- **Neuroscience Research**: Understanding biological learning and evolution
- **Cognitive Science**: Modeling memory and adaptation

### Technical Details

#### Reproduction Mechanisms
- Uses backpropagation for supervised learning
- Implements Hebbian learning for unsupervised pattern association
- Sigmoid activation functions for smooth gradients
- Mean Squared Error (MSE) loss function

#### Innovation Mechanisms
- Tournament selection for parent selection
- Gaussian mutation for weight perturbation
- Elitism to preserve best solutions
- Fitness-based ranking for selection pressure

### Further Reading

- Hebb, D.O. (1949). "The Organization of Behavior"
- Stanley, K.O. & Miikkulainen, R. (2002). "Evolving Neural Networks through Augmenting Topologies"
- Floreano, D. et al. (2008). "Neuroevolution: from architectures to learning"

### License

MIT License - Free for educational and research purposes

### Contributors

Systems and Computational Neuroscience Group

### Acknowledgments

This project is part of a course on Systems and Computational Neuroscience, exploring the intersection of neuroscience, evolution, and artificial intelligence.
