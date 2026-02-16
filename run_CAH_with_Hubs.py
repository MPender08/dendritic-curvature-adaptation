import networkx as nx
import numpy as np
import ot
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing

# ==========================================
# 1. GRAPH GENERATION (With Hubs)
# ==========================================

def generate_galton_watson(depth, mean_branching=3.0, seed=None):
    """
    Generates a Stochastic Tree (Galton-Watson process).
    """
    if seed: np.random.seed(seed)
    
    G = nx.Graph()
    G.add_node(0, layer=0)
    current_layer_nodes = [0]
    next_node_id = 1
    
    for d in range(depth):
        next_layer_nodes = []
        for node in current_layer_nodes:
            # Poisson branching creates heterogeneity
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

def add_hubs(G, n_hubs=3, connectivity=0.1):
    """
    Adds 'VIP-like' Hub Nodes.
    These connect to a random fraction of existing nodes, creating shortcuts.
    """
    nodes = list(G.nodes())
    n_existing = len(nodes)
    next_id = max(nodes) + 1
    
    # Add Hubs
    for i in range(n_hubs):
        hub_id = next_id + i
        G.add_node(hub_id, type='hub')
        
        # Connect to random subset of neurons (Global inhibition/disinhibition)
        targets = np.random.choice(nodes, size=int(n_existing * connectivity), replace=False)
        for t in targets:
            G.add_edge(hub_id, t)
            
    return G

def get_scrambled_topology(G):
    G_null = G.copy()
    n_swaps = 10 * G.number_of_edges()
    try:
        nx.connected_double_edge_swap(G_null, nswap=n_swaps)
    except:
        nx.double_edge_swap(G_null, nswap=n_swaps, max_tries=5000)
    return G_null

# ==========================================
# 2. CORE PHYSICS (Ollivier-Ricci)
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

def compute_edge_curvature(u, v, G, gamma, dist_matrix, node_index):
    # 1. Get distributions
    mu_u_dict = get_transition_measure(G, u, gamma)
    mu_v_dict = get_transition_measure(G, v, gamma)
    
    # 2. Map to dense arrays for OT solver
    n_nodes = len(node_index)
    mu_u_vec = np.zeros(n_nodes)
    mu_v_vec = np.zeros(n_nodes)
    
    for node, mass in mu_u_dict.items():
        mu_u_vec[node_index[node]] = mass
    for node, mass in mu_v_dict.items():
        mu_v_vec[node_index[node]] = mass
        
    # 3. Compute Wasserstein Distance
    w1 = ot.emd2(mu_u_vec, mu_v_vec, dist_matrix)
    
    return 1 - w1

def compute_curvature_stats_parallel(G, gamma):
    # Pre-compute All-Pairs Shortest Paths
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
    
    # Parallel Execution
    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores, backend="loky")(
        delayed(compute_edge_curvature)(u, v, G, gamma, M, node_index) 
        for u, v in edges
    )
    
    return np.mean(results), np.std(results)

# ==========================================
# 3. EXPERIMENT RUNNER
# ==========================================

def run_hub_experiment():
    depth = 5 
    gamma_values = np.linspace(0.05, 0.95, 15)
    n_seeds = 5 
    
    results = {
        'gamma': gamma_values,
        'tree_mean': [], 'tree_std': [],
        'tree_hub_mean': [], 'tree_hub_std': [] # New: Tree + Hubs
    }

    print(f"--- Starting CAH Hub Robustness Simulation ---")
    
    # Storage for averaging across seeds
    temp_tree_mean = np.zeros((n_seeds, len(gamma_values)))
    temp_hub_mean = np.zeros((n_seeds, len(gamma_values)))

    for s in range(n_seeds):
        print(f"\n[Seed {s+1}/{n_seeds}] Generating Graphs...")
        
        # 1. Base Tree
        G_tree = generate_galton_watson(depth, mean_branching=3.0, seed=s)
        
        # 2. Tree + Hubs (VIP Simulation)
        G_hubs = G_tree.copy()
        # Adding 5 hubs that connect to 15% of the network each
        G_hubs = add_hubs(G_hubs, n_hubs=5, connectivity=0.15) 
        
        for i, gamma in enumerate(tqdm(gamma_values, desc="   > Sweeping Gamma")):
            # Calc Base Tree
            mu, _ = compute_curvature_stats_parallel(G_tree, gamma)
            temp_tree_mean[s, i] = mu
            
            # Calc Tree + Hubs
            mu_h, _ = compute_curvature_stats_parallel(G_hubs, gamma)
            temp_hub_mean[s, i] = mu_h

    # Aggregate results (Mean across seeds)
    results['tree_mean'] = np.mean(temp_tree_mean, axis=0)
    results['tree_hub_mean'] = np.mean(temp_hub_mean, axis=0)
    
    return results

def plot_hub_impact(results):
    plt.figure(figsize=(10, 6))
    
    gammas = results['gamma']
    
    # --- Plot: The Curvature Shift (Mean) ---
    
    plt.plot(gammas, results['tree_mean'], 'r-o', linewidth=2, label='Standard Hierarchy (SST Only)')
    plt.plot(gammas, results['tree_hub_mean'], 'b-s', linewidth=2, label='Hierarchy + VIP Hubs')
    
    plt.axhline(0, color='gray', linewidth=0.5)
    plt.xlabel(r"Coupling $\gamma$ (Apical Conductance)", fontsize=12)
    plt.ylabel(r"Mean Curvature $\bar{\kappa}$", fontsize=12)
    plt.title("Impact of Hub Nodes (VIP) on Geometric Transition", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("CAH_Hub_Impact.png", dpi=300)
    print("\nSimulation Complete. Saved to CAH_Hub_Impact.png")
    plt.show()

if __name__ == "__main__":
    data = run_hub_experiment()
    plot_hub_impact(data)