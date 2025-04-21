# tsCSHMM: Time Shadow Continuous-State Hidden Markov Model for scRNA-seq Trajectories

**tsCSHMM** is a modular, performance-oriented framework for learning continuous, branching trajectories from single-cell RNA-seq data using Expectation-Maximization (EM).

This repository presents a reimplementation of the Continuous-State Hidden Markov Model (CSHMM) originally introduced by Lin and Bar-Joseph (2019), replacing their variational inference approach with a streamlined EM algorithm that is faster, more interpretable, and amenable to large-scale datasets.

While inspired by the CSHMM formulation, tsCSHMM introduces novel structural and computational modifications that extend its capabilities beyond the original framework. The model is built around a graph-based latent space, flexible emission modeling, and efficient reassignment of cells via direct optimization.

**Note**: The "Time Shadow" aspect of the model refers to a new conceptual layer that guides latent trajectory construction; additional methodological details will be released in a forthcoming preprint.

## Features

- Continuous-state cell assignments along latent trajectory graphs
- EM-based updates for emission means, kinetic decay, and variance
- Modular pruning and restructuring for trajectory refinement
- GPU-accelerated PyTorch components and parallelized optimization

## Background

Original method:  
Lin, C. & Bar-Joseph, Z. (2019). Continuous-state HMMs for modeling time-series single-cell RNA-Seq data. *Bioinformatics*, 35(22), 4707–4715.  
DOI: [10.1093/bioinformatics/btz296](https://doi.org/10.1093/bioinformatics/btz296)

We acknowledge the foundational contributions of Lin and Bar-Joseph and build directly on the principles they introduced, while significantly advancing the practical performance and flexibility of the method.

## Getting Started

1. Install dependencies:

pip install torch scanpy anndata numpy networkx
Prepare your .h5ad dataset with clustering (e.g., via Leiden) and connectivity (e.g., via PAGA).

Construct a trajectory and initialize the model:

from models import initialize_trajectory
traj, assignments = initialize_trajectory(adata)
Fit the model using the EMTrainer:

trainer = EMTrainer(traj, assignments)
trainer.run_em(n_iterations=5)

## Directory Structure

tsCSHMM/
├── models/
│   └── trajectory.py       # Core model logic
├── viz/
│   └── trajectory.py       # Visualization utilities
└── Readme.md

## License and Citation

Preprint and formal citation will be provided upon release. Please contact the authors for early collaboration or usage inquiries.