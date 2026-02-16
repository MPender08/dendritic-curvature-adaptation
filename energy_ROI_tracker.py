import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
# We import the physics engine from your main script
from run_CAH_scaling_analysis import generate_galton_watson, compute_curvature_stats_parallel

# ==========================================
# 1. NEW PATHOLOGY FUNCTION (Pruning)
# ==========================================
def get_pruned_topology(G, prune_fraction=0.3):
    """
    The 'Alzheimer's' Control (Geometric Collapse).
    Instead of rewiring (which keeps paths short), we DELETE edges.
    This forces signals to take longer, inefficient detours and destroys
    the high-density connectivity required for the phase transition.
    """
    G_pruned = G.copy()
    edges = list(G_pruned.edges())
    num_to_remove = int(len(edges) * prune_fraction)
    
    # Randomly remove edges
    if num_to_remove > 0:
        indices = np.random.choice(len(edges), num_to_remove, replace=False)
        edges_to_remove = [edges[i] for i in indices]
        G_pruned.remove_edges_from(edges_to_remove)
    
    # CRITICAL: If the graph becomes disconnected, the path length explodes.
    # To keep the simulation running, we add minimal random bridges, 
    # but the topology remains 'damaged' and inefficient.
    if not nx.is_connected(G_pruned):
        components = list(nx.connected_components(G_pruned))
        for i in range(len(components) - 1):
            # Connect a random node from comp[i] to comp[i+1]
            u = list(components[i])[0]
            v = list(components[i+1])[0]
            G_pruned.add_edge(u, v)
            
    return G_pruned

# ==========================================
# 2. THE SIMULATION
# ==========================================
def run_energy_roi_simulation_real():
    print("--- Starting Metabolic ROI Simulation (Data-Driven) ---")
    
    # Biophysical Parameters (Attwell & Laughlin, 2001)
    k_maint = 15.0   # Cost to maintain resting potential / SST gating
    k_sign  = 85.0   # Cost of Action Potentials (Signaling)
    base_path = 5.0  # Average path length baseline
    
    # Simulation Settings
    depth = 5        
    gammas = np.linspace(0.05, 0.95, 15) 
    
    # 2. Generate Topologies
    print(f"Generating hierarchical network (Depth={depth})...")
    G_tree = generate_galton_watson(depth=depth, mean_branching=3.0, seed=42)
    
    print("Generating Pruned control (Alzheimer's/Geometric Collapse)...")
    # Prune 30% of synapses (Significant Neurodegeneration)
    G_null = get_pruned_topology(G_tree, prune_fraction=0.30)
    
    results = {
        'gamma': gammas,
        'hierarchy': {'total': []},
        'pruned': {'total': []}
    }

    # 3. The Physics Engine
    for G, label in [(G_tree, 'hierarchy'), (G_null, 'pruned')]:
        print(f"\nSimulating Thermodynamics for: {label.upper()}")
        
        for gamma in gammas:
            # A. Calculate REAL Curvature
            # Calls the exact Wasserstein solver from your scaling script
            kappa_mean, _ = compute_curvature_stats_parallel(G, gamma)
            
            # B. The Metabolic Transfer Function
            c_maint = k_maint * gamma
            
            # Signaling Cost Logic:
            # Hyperbolic (Negative Kappa) = Geodesic Shortcuts = Lower Cost
            # Damaged/Euclidean (Positive/Flat Kappa) = Long Paths = Higher Cost
            effective_path = base_path * (1.0 + kappa_mean)
            
            # Clamp path length to realistic minimum
            effective_path = max(0.5, effective_path)
            
            c_sign = k_sign * effective_path
            total_energy = c_maint + c_sign
            
            results[label]['total'].append(total_energy)
            
            print(f"   Gamma: {gamma:.2f} | Kappa: {kappa_mean:.3f} | Total Energy: {total_energy:.1f}")

    # 4. Visualization
    plt.figure(figsize=(10, 7))
    
    # Plot Healthy Hierarchy (Red)
    plt.plot(gammas, results['hierarchy']['total'], 'r-', linewidth=3, label='Healthy Hierarchy (Witness State)')
    
    # Plot Pruned Pathology (Grey)
    plt.plot(gammas, results['pruned']['total'], 'k--', linewidth=2, alpha=0.6, label='Pruned/Alzheimer\'s (Geometric Collapse)')
    
    # Highlight the "Signaling Tax Haven" (Landauer Deficit)
    # The gap where Healthy is cheaper than Pruned
    y_healthy = np.array(results['hierarchy']['total'])
    y_pruned = np.array(results['pruned']['total'])
    
    plt.fill_between(gammas, y_healthy, y_pruned, where=(y_healthy < y_pruned), 
                     color='green', alpha=0.15, label='Landauer Deficit / Tax Haven')

    plt.title('Metabolic Phase Transition: The Landauer Deficit', fontsize=14)
    plt.xlabel(r'SST Gating Intensity ($\gamma$)', fontsize=12)
    plt.ylabel('Net Metabolic Cost (ATP Arbitrary Units)', fontsize=12)
    
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("Metabolic_Phase_Transition_Proof.png", dpi=300)
    print("\nSimulation Complete. Plot saved as 'Metabolic_Phase_Transition_Proof.png'")
    plt.show()

if __name__ == "__main__":
    run_energy_roi_simulation_real()