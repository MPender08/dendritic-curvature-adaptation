"""
Geometry-Aware Plasticity: The Biological Manifold
Demonstrating the VIP-SST-PC Phase Transition in Spiking Wetware
"""
import nest
import matplotlib.pyplot as plt
import numpy as np

# 1. INITIALIZE NEST
nest.set_verbosity("M_WARNING")
nest.ResetKernel()
nest.SetKernelStatus({"resolution": 0.1})

# 2. CREATE POPULATIONS (VIP-SST-PC Microcircuit)
pop_PC = nest.Create("iaf_psc_exp", 200)   
pop_SST = nest.Create("iaf_psc_exp", 40)   
pop_VIP = nest.Create("iaf_psc_exp", 40)   

# 3. STIMULUS GENERATORS
noise_PC = nest.Create("poisson_generator", params={"rate": 1500.0})
drive_SST = nest.Create("poisson_generator", params={"rate": 2500.0})
trigger_VIP = nest.Create("step_current_generator", 
                          params={"amplitude_times": [0.1, 500.0], 
                                  "amplitude_values": [0.0, 1500.0]}) 

# 4. RECORDERS
sr_PC = nest.Create("spike_recorder")
sr_SST = nest.Create("spike_recorder")
sr_VIP = nest.Create("spike_recorder")     

# 5. WIRING THE TOPOLOGY
# Baseline Drive
nest.Connect(noise_PC, pop_PC, syn_spec={"weight": 500.0})
nest.Connect(drive_SST, pop_SST, syn_spec={"weight": 600.0})
nest.Connect(trigger_VIP, pop_VIP)

# Oscilloscopes
nest.Connect(pop_PC, sr_PC)
nest.Connect(pop_SST, sr_SST)
nest.Connect(pop_VIP, sr_VIP)              

# The Geodesic Highway and the Shunts
nest.Connect(pop_PC, pop_PC, conn_spec={"rule": "fixed_indegree", "indegree": 20}, syn_spec={"weight": 100.0})
nest.Connect(pop_SST, pop_PC, conn_spec="all_to_all", syn_spec={"weight": -1000.0})
nest.Connect(pop_VIP, pop_SST, conn_spec="all_to_all", syn_spec={"weight": -1500.0})

# 6. RUN THE PHASE TRANSITION
print("Executing 40Hz Hyperbolic Plunge...")
nest.Simulate(1000.0)

# 7. PLOTTING THE MANIFOLD
ev_PC = sr_PC.get("events")
ev_SST = sr_SST.get("events")
ev_VIP = sr_VIP.get("events")          

plt.figure(figsize=(12, 6))
plt.plot(ev_PC["times"], ev_PC["senders"], '|', color='royalblue', markersize=3, label="Pyramidal Cells")
plt.plot(ev_SST["times"], ev_SST["senders"], '|', color='crimson', markersize=3, label="SST (The Gate)")
plt.plot(ev_VIP["times"], ev_VIP["senders"], '|', color='forestgreen', markersize=4, label="VIP (The Trigger)") 

plt.axvline(x=500.0, color='black', linestyle='--', linewidth=2)
plt.xlim(0, 1000)
plt.title("The Biological Manifold: VIP-SST-PC Phase Transition", fontsize=14, fontweight='bold')
plt.xlabel("Time (ms)", fontsize=12)
plt.ylabel("Neuron ID", fontsize=12)
plt.legend(loc="upper right")
plt.tight_layout()

# Save for publication and display
plt.savefig('biological_manifold_raster.png', dpi=300)
plt.show()