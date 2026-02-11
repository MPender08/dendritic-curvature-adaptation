import networkx as nx
import numpy as np
import ot
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing

# ==========================================
# 1. CORE PHYSICS ENGINE
# ==========================================

def get_transition_measure(G, node, gamma):
    """
    Constructs the measure mu_i for the Lazy Random Walk.
    """
    neighbors = list(G.neighbors(node))
    deg = len(neighbors)
    measure = {} # Use sparse dict for memory efficiency
    
    if deg == 0:
        measure[node] = 1.0
        return measure

    # Self-loop "laziness"
    measure[node] = 1 - gamma
    
    # Spread to neighbors
    if deg > 0:
        mass_per_neighbor = gamma / deg
        for neighbor in neighbors:
            measure[neighbor] = mass_per_neighbor
            
    return measure

def compute_edge_curvature(u, v, G, gamma, dist_matrix, node_index):
    """
    Helper function to compute curvature for a single edge (for parallelization).
    """
    # 1. Get distributions (sparse dictionaries)
    mu_u_dict = get_transition_measure(G, u, gamma)
    mu_v_dict = get_transition_measure(G, v, gamma)
    
    # 2. Convert to arrays for OT library
    # We only care about nodes that have mass (optimization)
    active_nodes = list(set(mu_u_dict.keys()) | set(mu_v_dict.keys()))
    
    # Map active nodes to indices in the local sub-matrix
    # (However, for exactness with pre-computed dist_matrix, we use global indices)
    
    # Create full density arrays (dense) - required for ot.emd2 with precomputed metric
    # Note: For N=7 (3000 nodes), dense arrays are fine (3000 floats is tiny).
    n_nodes = len(node_index)
    mu_u_vec = np.zeros(n_nodes)
    mu_v_vec = np.zeros(n_nodes)
    
    for node, mass in mu_u_dict.items():
        mu_u_vec[node_index[node]] = mass
    for node, mass in mu_v_dict.items():
        mu_v_vec[node_index[node]] = mass
        
    # 3. Compute Wasserstein Distance using the pre-computed global distance matrix
    # This is the heavy lifting
    w1 = ot.emd2(mu_u_vec, mu_v_vec, dist_matrix)
    
    return 1 - w1

def compute_network_curvature_parallel(G, gamma):
    """
    Computes average curvature using all CPU cores.
    """
    # 1. Pre-compute All-Pairs Shortest Paths (Distance Matrix)
    # This is fast for N=7 (< 5 seconds)
    nodes = list(G.nodes())
    n_nodes = len(nodes)
    node_index = {n: i for i, n in enumerate(nodes)}
    
    # Create Distance Matrix M (Floyd-Warshall is too slow, use BFS for unweighted)
    # NetworkX to_numpy_array with shortest path length is faster?
    # Actually, for unweighted graphs, dict(nx.all_pairs_shortest_path_length) is best.
    
    # Optimization: Calculate the matrix once.
    # For N=7 (~3200 nodes), a 3200x3200 matrix is ~80MB RAM. Totally fine.
    dist_gen = nx.all_pairs_shortest_path_length(G)
    M = np.zeros((n_nodes, n_nodes))
    
    # Fill matrix (this takes a moment but is done once per graph)
    for u, paths in dist_gen:
        i = node_index[u]
        for v, dist in paths.items():
            j = node_index[v]
            M[i, j] = dist
            
    edges = list(G.edges())
    
    # 2. Parallel Execution
    # usage of -1 uses all available cores
    num_cores = multiprocessing.cpu_count()
    
    results = Parallel(n_jobs=num_cores, backend="loky")(
        delayed(compute_edge_curvature)(u, v, G, gamma, M, node_index) 
        for u, v in edges
    )
    
    return np.mean(results)

# ==========================================
# 2. NULL MODEL
# ==========================================

def get_scrambled_topology(G):
    G_null = G.copy()
    n_swaps = 5 * G.number_of_edges()
    try:
        nx.connected_double_edge_swap(G_null, nswap=n_swaps)
    except:
        nx.double_edge_swap(G_null, nswap=n_swaps, max_tries=5000)
    return G_null

# ==========================================
# 3. EXPERIMENT RUNNER
# ==========================================

def run_scaling_experiment():
    # Depths to test. N=7 is the "Money Shot" for the paper.
    # Warning: N=7 might take 10-20 mins even with parallelization.
    depths = [3, 5, 7] 
    branching_factor = 3
    gamma_values = np.linspace(0, 0.95, 15)
    
    results = {}

    print(f"--- Starting CAH Local Simulation ---")
    print(f"Detected {multiprocessing.cpu_count()} CPU cores. Engaging parallel processing.")
    
    for depth in depths:
        print(f"\n[Phase] Simulating Depth N={depth}")
        
        # Generate Graphs
        G_tree = nx.balanced_tree(branching_factor, depth)
        print(f"   > Tree Generated: {G_tree.number_of_nodes()} nodes")
        
        G_null = get_scrambled_topology(G_tree)
        print(f"   > Null Model Generated (Degree Preserved)")
        
        curve_tree = []
        curve_null = []
        
        # Loop Gammas
        for gamma in tqdm(gamma_values, desc=f"   > Sweeping Gamma"):
            # We calculate both inside the loop
            k_tree = compute_network_curvature_parallel(G_tree, gamma)
            k_null = compute_network_curvature_parallel(G_null, gamma)
            
            curve_tree.append(k_tree)
            curve_null.append(k_null)
            
        results[depth] = {
            'gamma': gamma_values,
            'tree': curve_tree,
            'null': curve_null
        }

    return results

# ==========================================
# 4. PLOTTING
# ==========================================

def plot_results(results):
    plt.figure(figsize=(12, 7))
    
    colors = ['#ffb3b3', '#ff6666', '#cc0000'] # Light to Dark Red
    
    plt.axhline(0, color='black', linewidth=1)
    
    # Plot Logic
    for i, depth in enumerate(results.keys()):
        data = results[depth]
        gammas = data['gamma']
        
        # Hierarchy
        plt.plot(gammas, data['tree'], 
                 label=f'Hierarchy (N={depth})', 
                 color=colors[i % len(colors)], 
                 marker='o', markersize=4, linewidth=2)
        
        # Only plot the largest Null model to keep plot clean, or all dashed
        if i == len(results) - 1:
            plt.plot(gammas, data['null'], 
                     label=f'Scrambled Null (N={depth})', 
                     color='gray', linestyle='--', alpha=0.8)
        else:
            plt.plot(gammas, data['null'], 
                     color='gray', linestyle='--', alpha=0.3)

    # Highlight Region
    plt.axvspan(0.85, 0.95, color='red', alpha=0.1, label='Hyperbolic Regime')

    plt.xlabel(r"Apical-Somatic Conductance ($\gamma$)", fontsize=14)
    plt.ylabel(r"Mean Ollivier-Ricci Curvature ($\kappa$)", fontsize=14)
    plt.title("Finite-Size Scaling: Phase Transition Robustness", fontsize=16)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Save results
    plt.savefig("CAH_Scaling_Result.png", dpi=300)
    print("\nPlot saved to 'CAH_Scaling_Result.png'")
    plt.show()

if __name__ == "__main__":
    # Windows/Mac requires this block for multiprocessing
    data = run_scaling_experiment()
    plot_results(data)