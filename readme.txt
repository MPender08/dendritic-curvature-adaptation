Dynamic Curvature Adaptation - Simulation Code
==============================================

Overview:
This folder contains the Python source code used to generate the figures for the manuscript.

File Structure:
1. run_CAH_scaling_analysis.py //   Finite-size scaling and robustness tests.
2. run_CAH_with_Hubs.py        //   Simulation of hyper-integrative/manic states. 
3. run_CAH_Pruning.py          //   Simulation of neurodegenerative collapse.
4. energy_ROI_tracker.py       //   Metabolic expenditure modeling.


Instructions:
1. Ensure both run_CAH_scaling_analysis.py AND energy_ROI_tracker.py are in the SAME folder.
2. Install dependencies: pip install -r requirements.txt
3. Run the main simulation: python simulation.py

Output:
The script will generate figure files (e.g., 'CAH_Scaling.jpg') in the current directory.

