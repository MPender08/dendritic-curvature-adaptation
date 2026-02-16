# Dynamic Curvature Adaptation

This repository contains the official simulation suite for the **Curvature Adaptation Hypothesis** (CAH), a unified theory of cortical state that bridges cellular-level dendritic gating with macroscopic functional geometry, and the **Metabolic Phase Transition**, a theoretical framework that reframes consciousness as a thermodynamic necessity for high-dimensional biological systems.

The Dynamic Curvature Adaptation manuscript is available here: [https://doi.org/10.5281/zenodo.18615180](https://doi.org/10.5281/zenodo.18615180)

The Metabolic Phase Transition manuscript is available here: [https://doi.org/10.5281/zenodo.18655523](https://doi.org/10.5281/zenodo.18655523)

## Overview

The **Curvature Adaptation Hypothesis** (CAH) proposes that the brain does not reside in a fixed geometric manifold. Instead, it dynamically "warps" its functional space to match the hierarchical depth of incoming data. We identify Somatostatin (SST) interneuron-mediated dendritic shunting as a biophysical actuator that regulates the apical-somatic conductance ratio (γ) to serve as a geometric switch.

By modulating this switch, the cortex can transition from a stable Euclidean regime (κ≈0) to a deep Hyperbolic regime (κ<0), unlocking a global "signaling tax haven" for efficient hierarchical inference.

### Key Theoretical Findings

**Topological Robustness:** The hyperbolic phase transition is driven by local synaptic density rather than global architectural order. It survives degree-preserving scrambling but collapses under synaptic loss.

**Geometric Trilogy:** We model cognitive health and disease as distinct functional states:

        Healthy: Tunable flexibility between flat and hyperbolic manifolds.

        Manic: "Geometric Inelasticity" (forced hyperbolicity) via VIP-like hub nodes.

        Demented: "Geometric Collapse" (trapped Euclidean) via stochastic pruning.

## Installation

The script energy_ROI_tracker.py depends on the physics engine in run_CAH_scaling_analysis.py. Please ensure both files are downloaded to the same directory before running.

This project requires Python 3.8+ and the following scientific libraries:
```
pip install networkx numpy matplotlib pot tqdm joblib
```

### Simulation Suite


**1. Finite-Size Scaling and Robustness** (run_CAH_scaling_analysis.py)

  Reproduces Figure 1 from the manuscript. It tests the scale-invariance of the phase transition across depths (N=3,5,7) and compares the hierarchy against a scrambled null model.
  Optimization: Utilizes a Sparse Neighborhood Transport algorithm to reduce computational complexity for large graphs (N=7, ~8500 nodes), ignoring zero-mass entries in the distance matrix to accelerate the OT solver.

**2. Pathological Hubs (Manic State)** (run_CAH_with_Hubs.py)

  Reproduces Figure 2. Introduces high-centrality "VIP-like" hub nodes to demonstrate how hyper-connectivity abolishes the Euclidean "rest" state.

**3. Synaptic Pruning (Geometric Collapse)** (run_CAH_Pruning.py)

  Reproduces Figure 3. Simulates the 30% stochastic spine loss characteristic of Alzheimer’s disease to demonstrate the loss of geometric depth.

**4. Metabolic ROI Tracker** (energy_ROI_tracker.py)

  Reproduces Figure 4. Models the metabolic trade-off between the 'tax' of maintaining SST gating and the 'profit' of hyperbolic signaling, demonstrating the 'Landauer Deficit' accessible to healthy networks but lost in pruned (Alzheimer's-like) topologies.
  
## Citation

If you use this code or the CAH framework in your research, please cite the manuscript:

```
@article{pender2026dynamic,
  title={Dynamic Curvature Adaptation: A Unified Geometric Theory of Cortical State and Pathological Collapse},
  author={Pender, Matthew A.},
  year={2026},
  journal={Zenodo Pre-print},
  doi={10.5281/zenodo.18653519}
}
```

## Acknowledgments

This research was assisted by Gemini 3 Flash and Gemini 3 Pro for drafting text, generating simulation code, and structural argument analysis.
