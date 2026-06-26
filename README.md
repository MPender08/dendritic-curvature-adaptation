# Dynamic Curvature Adaptation

This repository contains the official simulation suite for the **Curvature Adaptation Hypothesis** (CAH).

The Dynamic Curvature Adaptation manuscript is available here: [https://doi.org/10.5281/zenodo.18615180](https://doi.org/10.5281/zenodo.18615180)

<figure>
        <img src="Metabolic_Phase_Transition_Proof.png">
</figure>        

**Figure 5: Thermodynamic Return on Curvature-Sensitive Control.** Simulation of total metabolic cost (C_Total) as a function of SST gating intensity (γ). **Red Line:** Healthy hierarchical networks exhibit a transition near γ ≈ 0.8, beyond which local control expenditure is offset by a marked reduction in distributed signaling burden. This indicates that the architecture can convert gating cost into a real energetic return by gaining access to a more transport-efficient routing regime. **Grey Line:** Pathological networks with 30% synaptic pruning fail to realize the same return. Although maintenance cost is still incurred, pruning raises resistance to the transition and prevents the architecture from securing a comparable reduction in signaling cost, leaving the system trapped in a more metabolically expensive regime. **Green Region:** The macroscopic thermodynamic discount—the energetic advantage obtained when local control successfully purchases more efficient large-scale routing. In later refinements of CAH, this advantage is interpreted less as a simple reward for maximal global hyperbolicity and more as evidence that selective curvature-sensitive routing can produce a genuine return on investment under the right structural conditions."


<br>
<br>


## Overview

The **Curvature Adaptation Hypothesis (CAH)** proposes that the brain does not operate within a single fixed geometric manifold. Instead, it may dynamically regulate its effective information geometry in response to hierarchical demand. In its foundational form, CAH identifies a plausible biophysical actuator—the Martinotti-cell subtype of Somatostatin (SST) interneurons—which regulates the apical-somatic conductance ratio (γ) and thereby modulates access to distinct routing regimes.

Within this framework, SST-mediated gating can shift cortical dynamics from a relatively flat baseline (κ≈0) toward more transport-efficient negative-curvature routing conditions (κ<0). The key claim is not simply that “more hyperbolic is better,” but that curvature-sensitive routing can reduce the effective relay burden of hierarchical inference, allowing the cortex to secure a real thermodynamic advantage under the right structural conditions.

<figure> <img src="biological_manifold.png"> <figcaption>Figure 2: Spiking Neural Network Validation. A PyNEST simulation of 280 integrate-and-fire neurons illustrating a plausible microcircuit mechanism for curvature-sensitive routing control. Left (0-500ms): Strong SST interneuron gating (red) suppresses pyramidal cell activity (blue), maintaining the network in a relatively conservative baseline regime. Right (500-1000ms): A step current activates VIP interneurons (green), which suppress the SST gates. This disinhibition allows the pyramidal population to spontaneously reorganize into highly synchronized 40Hz gamma-band pillars, providing a biologically plausible temporal signature of the local control dynamics that could open access to a more strongly integrated large-scale routing regime. </figcaption> </figure>

## Installation

The script **energy_ROI_tracker.py** depends on the physics engine in **run_CAH_scaling_analysis.py**. Please ensure **both files** are downloaded to the same directory before running.

Note: The PyNEST simulation requires NEST to be installed.

```bash
git clone https://github.com/MPender08/dendritic-curvature-adaptation.git
cd dendritic-curvature-adaptation
```

This project requires Python 3.8+ and the following scientific libraries:
```bash
pip install networkx numpy matplotlib pot tqdm joblib scipy
```

### Simulation Suite


**1. Finite-Size Scaling and Robustness** (run_CAH_scaling_analysis.py)
```bash
python run_CAH_scaling_analysis.py
```

  Reproduces Figure 1 from the manuscript. Tests whether the routing transition remains accessible across depths (N=3,5,7) and under degree-preserving topological scrambling.
  Optimization: Utilizes a Sparse Neighborhood Transport algorithm to reduce computational complexity for large graphs (N=7, ~8500 nodes), ignoring zero-mass entries in the distance matrix to accelerate the OT solver.

**2. Pathological Hubs (Manic State)** (run_CAH_with_Hubs.py)
```bash
python run_CAH_with_Hubs.py
```

  Reproduces Figure 2. Introduces high-centrality “VIP-like” hub nodes to show how hyper-connectivity biases the system away from its conservative baseline and toward less regulated, shortcut-dominated routing.

**3. Synaptic Pruning (Geometric Collapse)** (run_CAH_Pruning.py)
```bash
python run_CAH_Pruning.py
```

  Reproduces Figure 3. Simulates 30% stochastic spine loss to show how neurodegenerative pruning restricts geometric access and degrades hierarchical integration.

**4. Metabolic ROI Tracker** (energy_ROI_tracker.py)
```bash
python energy_ROI_tracker.py
```

  Reproduces Figure 4. Quantifies the metabolic ROI of SST-mediated gating by comparing local control cost against reduced distributed signaling burden in healthy versus pruned networks.
  

## Acknowledgments

Gemini AI was utilized as an interactive technical sounding board to rapidly prototype structural arguments and optimize PyTorch/NEST simulations. The author takes full and exclusive responsibility for the originality, validity, and final synthesis of the theoretical framework presented herein.
