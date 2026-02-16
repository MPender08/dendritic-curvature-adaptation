import networkx as nx
import numpy as np
import ot
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing

# ==========================================
# 1. GRAPH GENERATION & PRUNING
# ==========================================

def generate_galton_watson(depth, mean_branching=3.0, seed=None):
    if seed: np.random.seed(seed)
    G = nx.Graph()
    G.add_node(0, layer=0)
    current_layer_nodes = [0]
    next_node_id = 1
    
    for d in range(depth):
        next_layer_nodes = []
        for node in current_layer_nodes:
            if d < 2: n_children = max(1, np.random.poisson(mean_branching))
            else: n_children = np.random.poisson(mean_branching)
            
            for _ in range(n_children):
                G.add_edge(node, next_node_id)
                G.nodes[next_node_id]['layer'] = d + 1
                next_layer_nodes.append(next_node_id)
                next_node_id += 1
        current_layer_nodes = next_layer_nodes
        if not current_layer_nodes: break
    return G

def scramble_topology(G):
    """
    Randomizes the graph connections while preserving degree distribution.
    This creates the 'cycles' and 'redundancy' that generate 
    Positive Curvature (Spherical Bulge).
    """
    G_scrambled = G.copy()
    n_edges = G_scrambled.number_of_edges()
    # Attempt connected double edge swap, fall back if graph is too small/rigid
    try:
        nx.connected_double_edge_swap(G_scrambled, nswap=5*n_edges)
    except:
        try:
            nx.double_edge_swap(G_scrambled, nswap=5*n_edges, max_tries=5000)
        except:
            pass # Keep original if swapping fails
    return G_scrambled

def prune_network(G, damage_percent=0.30):
    """
    Simulates 'Spine Loss' or 'Synaptic Pruning'.
    Randomly removes a percentage of edges.
    Returns the Largest Connected Component (LCC).
    """
    G_damaged = G.copy()
    edges = list(G_damaged.edges())
    num_to_remove = int(len(edges) * damage_percent)
    
    if num_to_remove > 0:
        edges_to_cut = np.random.choice(len(edges), num_to_remove, replace=False)
        for i in edges_to_cut:
            u, v = edges[i]
            if G_damaged.has_edge(u, v):
                G_damaged.remove_edge(u, v)
            
    # Return only the largest surviving cluster (functional core)
    largest_cc = max(nx.connected_components(G_damaged), key=len)
    return G_damaged.subgraph(largest_cc).copy()

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
        for neighbor in neighbors: measure[neighbor] = mass_per_neighbor
    return measure

def compute_edge_curvature(u, v, G, gamma, dist_matrix, node_index):
    mu_u_dict = get_transition_measure(G, u, gamma)
    mu_v_dict = get_transition_measure(G, v, gamma)
    n_nodes = len(node_index)
    mu_u_vec = np.zeros(n_nodes)
    mu_v_vec = np.zeros(n_nodes)
    for node, mass in mu_u_dict.items(): mu_u_vec[node_index[node]] = mass
    for node, mass in mu_v_dict.items(): mu_v_vec[node_index[node]] = mass
    w1 = ot.emd2(mu_u_vec, mu_v_vec, dist_matrix)
    return 1 - w1

def compute_curvature_stats_parallel(G, gamma):
    nodes = list(G.nodes())
    n_nodes = len(nodes)
    node_index = {n: i for i, n in enumerate(nodes)}
    
    # Compute All-Pairs Shortest Paths
    # Note: For very large graphs, this matrix is the bottleneck.
    dist_gen = nx.all_pairs_shortest_path_length(G)
    M = np.zeros((n_nodes, n_nodes))
    for u, paths in dist_gen:
        i = node_index[u]
        for v, dist in paths.items():
            j = node_index[v]
            M[i, j] = dist
            
    edges = list(G.edges())
    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores, backend="loky")(
        delayed(compute_edge_curvature)(u, v, G, gamma, M, node_index) for u, v in edges
    )
    return np.mean(results), np.std(results)

# ==========================================
# 3. EXPERIMENT RUNNER
# ==========================================

def run_pruning_experiment():
    depth = 5 
    gamma_values = np.linspace(0.05, 0.95, 15)
    n_seeds = 5 
    damage_level = 0.30  # 30% Synaptic Loss
    
    results = {
        'gamma': gamma_values,
        'healthy_mean': [], 
        'damaged_mean': []
    }

    print(f"--- Starting CAH Pruning (Disease) Simulation ---")
    print(f"--- Simulating {int(damage_level*100)}% Synaptic Loss on Scrambled Topology ---")
    
    temp_healthy_mean = np.zeros((n_seeds, len(gamma_values)))
    temp_damaged_mean = np.zeros((n_seeds, len(gamma_values)))

    for s in range(n_seeds):
        print(f"\n[Seed {s+1}/{n_seeds}] Generating & Pruning...")
        
        # 1. Healthy Tree (Hierarchical)
        G_healthy = generate_galton_watson(depth, mean_branching=3.0, seed=s)
        
        # 2. Pathological: Scramble structure first, THEN prune
        # This creates the cycles needed for the "Spherical Bulge" (Positive Curvature)
        G_scrambled = scramble_topology(G_healthy)
        G_damaged = prune_network(G_scrambled, damage_percent=damage_level)
        
        print(f"   > Healthy (Tree) Size: {G_healthy.number_of_nodes()} nodes")
        print(f"   > Damaged (Scrambled+Pruned) Size: {G_damaged.number_of_nodes()} nodes")
        
        for i, gamma in enumerate(tqdm(gamma_values, desc="   > Sweeping Gamma")):
            # Calc Healthy
            mu, _ = compute_curvature_stats_parallel(G_healthy, gamma)
            temp_healthy_mean[s, i] = mu
            
            # Calc Damaged
            mu_d, _ = compute_curvature_stats_parallel(G_damaged, gamma)
            temp_damaged_mean[s, i] = mu_d

    results['healthy_mean'] = np.mean(temp_healthy_mean, axis=0)
    results['damaged_mean'] = np.mean(temp_damaged_mean, axis=0)
    
    return results

def plot_disease_impact(results):
    plt.figure(figsize=(10, 6))
    gammas = results['gamma']
    
    # Healthy Baseline
    plt.plot(gammas, results['healthy_mean'], 'r-o', linewidth=2, label='Healthy Hierarchy (Intact)')
    
    # Disease State
    plt.plot(gammas, results['damaged_mean'], 'g--^', linewidth=2, label='Pathology (Scrambled + 30% Pruning)')
    
    plt.axhline(0, color='gray', linewidth=0.5)
    plt.xlabel(r"Coupling $\gamma$ (Apical Conductance)", fontsize=12)
    plt.ylabel(r"Mean Curvature $\bar{\kappa}$", fontsize=12)
    plt.title("Geometric Collapse: The Effect of Synaptic Pruning", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("CAH_Disease_Collapse.png", dpi=300)
    print("\nSimulation Complete. Saved to CAH_Disease_Collapse.png")
    plt.show()

if __name__ == "__main__":
    data = run_pruning_experiment()
    plot_disease_impact(data)