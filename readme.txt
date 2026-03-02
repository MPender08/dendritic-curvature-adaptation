Dynamic Curvature Adaptation - Simulation Code
==============================================

Overview:
This folder contains the PyTorch and PyNEST source code used to generate the figures for the manuscript.

File Structure:
1. run_CAH_scaling_analysis.py //   Finite-size scaling and robustness tests.
2. run_CAH_with_Hubs.py        //   Simulation of hyper-integrative/manic states. 
3. run_CAH_Pruning.py          //   Simulation of neurodegenerative collapse.
4. energy_ROI_tracker.py       //   Metabolic expenditure modeling.
5. biological_manifold.py      //   PyNEST SNN validation of the VIP-SST-Pyramidal 40Hz phase transition.


Instructions:

1. Ensure both run_CAH_scaling_analysis.py AND energy_ROI_tracker.py are in the SAME folder.

2. Install dependencies: pip install -r requirements.txt
	NOTE: NEST is required to run the PyNEST simulation.

3. Run the main simulations: 
    python run_CAH_scaling_analysis.py
    python run_CAH_with_Hubs.py
    python run_CAH_Pruning.py
    python energy_ROI_tracker.py

4. Run the PyNEST simulation: 
    python biological_manifold.py


Output:
The scripts will generate figure files in the current directory.