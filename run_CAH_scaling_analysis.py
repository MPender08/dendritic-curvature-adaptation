import networkx as nx
import numpy as np
import ot
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing

# ==========================================
# 1. GRAPH GENERATION
# ==========================================

def generate_galton_watson(depth, mean_branching=3.0, seed=None):
    """
    Generates a Stochastic Tree (Galton-Watson process).
    approximating biological dendrites with Poisson branching.
    """
    if seed: np.random.seed(seed)
    
    G = nx.Graph()
    G.add_node(0, layer=0)
    current_layer_nodes = [0]
    next_node_id = 1
    
    for d in range(depth):
        next_layer_nodes = []
        for node in current_layer_nodes:
            # Branching factor (Poisson)
            if d < 2:
                n_children = max(1, np.random.poisson(mean_branching))
            else:
                n_children = np.random.poisson(mean_branching)
            
            for _ in range(n_children):
                G.add_edge(node, next_node_id)
                G.nodes[next_node_id]['layer'] = d + 1
                next_layer_nodes.append(next_node_id)
                next_node_id += 1
        
        current_layer_nodes = next_layer_nodes
        if not current_layer_nodes:
            break
            
    return G

def get_scrambled_topology(G):
    """
    The 'Scrambled Topology' Control.
    Preserves degree sequence exactly, but randomizes connections.
    """
    G_null = G.copy()
    n_swaps = 10 * G.number_of_edges()
    try:
        nx.connected_double_edge_swap(G_null, nswap=n_swaps)
    except:
        # Fallback for unconnected graphs
        nx.double_edge_swap(G_null, nswap=n_swaps, max_tries=5000)
    return G_null

# ==========================================
# 2. CORE PHYSICS (Ollivier-Ricci) - OPTIMIZED
# ==========================================

def get_transition_measure(G, node, gamma):
    neighbors = list(G.neighbors(node))
    deg = len(neighbors)
    measure = {} 
    
    if deg == 0:
        measure[node] = 1.0
        return measure

    measure[node] = 1.0 - gamma
    
    if deg > 0:
        mass_per_neighbor = gamma / deg
        for neighbor in neighbors:
            measure[neighbor] = mass_per_neighbor
            
    return measure

def compute_edge_curvature_optimized(u, v, G, gamma, dist_matrix, node_index):
    """
    OPTIMIZED WASSERSTEIN COMPUTATION (Local Sub-matrix Method)
    Instead of passing the full NxN distance matrix (mostly zeros in mass),
    we isolate only the 'Active Support'—the union of the two neighborhoods.
    This reduces the Linear Programming problem from ~3000 variables to ~10.
    """
    # 1. Get distributions (Sparse dictionaries)
    mu_u_dict = get_transition_measure(G, u, gamma)
    mu_v_dict = get_transition_measure(G, v, gamma)
    
    # 2. Identify the "Active Support" (Union of neighborhoods)
    # We only care about nodes that have non-zero mass in EITHER distribution.
    active_nodes = list(set(mu_u_dict.keys()) | set(mu_v_dict.keys()))
    n_active = len(active_nodes)
    
    # 3. Create Local Matrices
    # Map global indices to local indices (0 to n_active-1)
    
    mu_u_local = np.zeros(n_active)
    mu_v_local = np.zeros(n_active)
    
    # We use a tiny local cost matrix
    M_local = np.zeros((n_active, n_active))
    
    for i, node_i in enumerate(active_nodes):
        # Fill weights
        mu_u_local[i] = mu_u_dict.get(node_i, 0.0)
        mu_v_local[i] = mu_v_dict.get(node_i, 0.0)
        
        # Fill distances (lookup from global matrix)
        idx_i = node_index[node_i]
        for j, node_j in enumerate(active_nodes):
            idx_j = node_index[node_j]
            M_local[i, j] = dist_matrix[idx_i, idx_j]
            
    # 4. Compute Exact Wasserstein Distance on the tiny matrix
    # This is mathematically equivalent to the global solve but significantly faster.
    w1 = ot.emd2(mu_u_local, mu_v_local, M_local)
    
    return 1 - w1

def compute_curvature_stats_parallel(G, gamma):
    nodes = list(G.nodes())
    n_nodes = len(nodes)
    node_index = {n: i for i, n in enumerate(nodes)}
    
    # Optimized APSP
    dist_gen = nx.all_pairs_shortest_path_length(G)
    M = np.zeros((n_nodes, n_nodes))
    for u, paths in dist_gen:
        i = node_index[u]
        for v, dist in paths.items():
            j = node_index[v]
            M[i, j] = dist
            
    edges = list(G.edges())
    num_cores = multiprocessing.cpu_count()
    
    # UPDATED: Calling the optimized function
    results = Parallel(n_jobs=num_cores, backend="loky")(
        delayed(compute_edge_curvature_optimized)(u, v, G, gamma, M, node_index) 
        for u, v in edges
    )
    
    return np.mean(results), np.std(results)

# ==========================================
# 3. SCALING EXPERIMENT RUNNER
# ==========================================

def run_scaling_experiment():
    depths = [3, 5, 7] # Matches Abstract N=3,5,7
    gamma_values = np.linspace(0.05, 0.95, 20)
    n_seeds = 5 # Averaging over 5 random realizations per depth
    
    results = {
        'gamma': gamma_values,
        'depths': depths,
        'scaling_means': {}, # Dictionary to store mean curve for each depth
        'scrambled_mean': None # Control for N=7
    }

    print(f"--- Starting CAH Finite-Size Scaling Analysis ---")
    
    # 1. Loop through Hierarchical Depths
    for d in depths:
        print(f"\n--- Simulating Depth N={d} ---")
        temp_mean = np.zeros((n_seeds, len(gamma_values)))
        
        for s in range(n_seeds):
            G_stoch = generate_galton_watson(depth=d, mean_branching=3.0, seed=s)
            print(f"   > Seed {s+1}: {G_stoch.number_of_nodes()} nodes")
            
            # If this is the largest depth (N=7), save one instance for the Null Control later
            if d == 7 and s == 0:
                G_reference_for_null = G_stoch.copy()

            for i, gamma in enumerate(tqdm(gamma_values, desc=f"   > Sweeping Gamma (Seed {s+1})")):
                mu, _ = compute_curvature_stats_parallel(G_stoch, gamma)
                temp_mean[s, i] = mu
        
        # Average across seeds for this depth
        results['scaling_means'][d] = np.mean(temp_mean, axis=0)

    # 2. Run Scrambled Control (Only for Max Depth N=7)
    print(f"\n--- Simulating Scrambled Control (N=7) ---")
    # We scramble the N=7 graph we saved earlier to ensure comparable size/degree
    G_null = get_scrambled_topology(G_reference_for_null)
    
    temp_null_mean = []
    for gamma in tqdm(gamma_values, desc="   > Sweeping Gamma (Null)"):
        mu, _ = compute_curvature_stats_parallel(G_null, gamma)
        temp_null_mean.append(mu)
        
    results['scrambled_mean'] = np.array(temp_null_mean)
    
    return results

# ==========================================
# 4. PLOTTING (Figure 1 from Abstract)
# ==========================================

def plot_scaling_analysis(results):
    plt.figure(figsize=(12, 8))
    
    gammas = results['gamma']
    
    # Plot Hierarchies (Red Gradients)
    colors = ['#ffcccc', '#ff6666', '#cc0000'] # Light to Dark Red
    for i, d in enumerate(results['depths']):
        plt.plot(gammas, results['scaling_means'][d], 
                 marker='o', markersize=5, linewidth=2, color=colors[i], 
                 label=f'Hierarchy (N={d})')

    # Plot Scrambled Null (Grey Dashed)
    plt.plot(gammas, results['scrambled_mean'], 
             color='gray', linestyle='--', linewidth=2, marker='^', markersize=5,
             label='Scrambled Null (N=7)')
    
    # Formatting
    plt.axhline(0, color='black', linewidth=0.8)
    plt.xlabel(r"Coupling $\gamma$ (Apical Conductance)", fontsize=14)
    plt.ylabel(r"Mean Curvature $\bar{\kappa}$", fontsize=14)
    plt.title("Finite-Size Scaling: Phase Transition Robustness", fontsize=16)
    
    # Highlight the "Hyperbolic Regime"
    plt.axvspan(0.75, 0.95, color='red', alpha=0.1, label='Hyperbolic Regime')
    
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("CAH_Scaling_Analysis.png", dpi=300)
    print("\nSimulation Complete. Saved to CAH_Scaling_Analysis.png")
    plt.show()

if __name__ == "__main__":
    data = run_scaling_experiment()
    plot_scaling_analysis(data)