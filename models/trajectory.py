import numpy as np
import pandas as pd
import networkx as nx
from viz.trajectory import plot_cells_on_trajectory

class TrajectoryGraph:
    def __init__(self, adata, cluster_key='leiden', random_state=0, laplace_h=1.0, root_cluster=None, topology_config=None):
        self.adata = adata
        self.random_state = random_state
        self.laplace_h = laplace_h
        self.cluster_key = cluster_key
        self.global_r2 = None # Will be initialized later
        self.G_traj = None
        self.levels = {}
        self.parent = {}
        self.split_nodes = []
        self.roots = []
        self.branch_probabilities = {}
        self.transition_probabilities = {}
        self.normalizing_constants = {}
        self.emission_params = {}
        self.node_emission = {}
        self.edge_list = []
        self.edge_segments = []
        self.node_for_cluster = {}
        self.cluster_for_node = {}

        
        self.root_cluster = root_cluster
        self.topology_config = topology_config
        self._initialize_graph()
        self._compute_edge_segments()

    def get_roots(self):
        return list(self.roots)

    def refresh_structure_after_pruning(self):
        """
        Rebuild internal structures after pruning or modifying the trajectory graph.
    
        This method:
        - Updates edge and node mappings
        - Drops orphaned nodes
        - Recomputes edge segments
        - Realigns emission parameters
        - Recomputes transition probabilities
        """
    
        G = self.G_traj
    
        # --- Drop orphan nodes ---
        orphan_nodes = [n for n in G.nodes() if G.in_degree(n) == 0 and G.out_degree(n) == 0]
        G.remove_nodes_from(orphan_nodes)
        # --- Recompute edge segments ---
        self.edge_list = list(G.edges())
        self._compute_edge_segments()
    
        # --- Realign or initialize emission parameters ---
        valid_edges = set(self.edge_list)
        self.emission_params = {
            edge: self.emission_params.get(edge, {
                "K": np.ones(self.adata.shape[1]),
            }) for edge in valid_edges
        }

    
        # --- Reset branch probabilities to 1.0 if missing ---
        self.branch_probabilities = {
            edge: self.branch_probabilities.get(edge, 1.0)
            for edge in self.edge_list
        }
    
        # --- Recompute transition probabilities ---
        self.transition_probabilities.clear()
        self.normalizing_constants.clear()
        self._initialize_transition_probabilities()
    
        return

    def log_branch_prior_for_edge(self, edge):
        u_name = edge[0]
        path = []
        node = u_name
        while True:
            cluster = self.cluster_for_node.get(node)
            parent = self.parent.get(cluster, None)
            if parent is None:
                break
            parent_name = self.node_for_cluster.get(parent)
            if parent_name is None:
                break
            path.append((parent_name, node))
            node = parent_name

        log_prob = 0.0
        for e in reversed(path):
            prob = self.branch_probabilities.get(e, 1e-6)
            log_prob += np.log(max(prob, 1e-10))

        return log_prob

    def get_branch_paths_for_edges(self, edges=None):
        if edges is None:
            edges = self.edge_list

        edge_to_path = {}
        edge_log_priors = {}

        for u_name, v_name in edges:
            current = u_name
            path = []
            while True:
                cluster = self.cluster_for_node.get(current)
                parent_cluster = self.parent.get(cluster, None)
                if parent_cluster is None:
                    break
                parent_name = self.node_for_cluster.get(parent_cluster)
                if parent_name is None:
                    break
                path.insert(0, (parent_name, current))
                current = parent_name

            log_prob = 0.0
            for edge in path + [(u_name, v_name)]:
                prob = self.branch_probabilities.get(edge, 1e-6)
                log_prob += np.log(max(prob, 1e-10))
            
            path.append((u_name, v_name))
            edge_to_path[(u_name, v_name)] = path
            edge_log_priors[(u_name, v_name)] = log_prob
        
        return edge_to_path, edge_log_priors

    def _initialize_graph(self):
        rng = np.random.default_rng(self.random_state)
        clusters = list(pd.Categorical(self.adata.obs[self.cluster_key]).categories)
        cluster_set = set(clusters)
    
        self.G_traj = nx.DiGraph()
        G_conn = nx.Graph()
    
        self.node_for_cluster.clear()
        self.cluster_for_node.clear()
        self.roots = []
        self.edge_list = []
        self.parent = {}
        self.levels = {}
    
        if self.topology_config is not None:
            # --- Manual topology config ---
            parents = {}
            children = {}
            for p, c in self.topology_config:
                if p not in cluster_set or c not in cluster_set:
                    raise ValueError(f"Invalid cluster in topology config: ({p}, {c})")
                G_conn.add_edge(p, c, weight=1.0)
                parents[c] = p
                children.setdefault(p, []).append(c)
                children.setdefault(c, [])
    
            root_candidates = [c for c in cluster_set if c not in parents]
            if not root_candidates:
                raise ValueError("No root clusters found in topology config.")
            self.roots = root_candidates
            self.parent = parents
    
            # Level computation
            self.levels = {}
            queue = [(r, 0) for r in self.roots]
            for r in self.roots:
                self.parent[r] = None
            while queue:
                cur, lvl = queue.pop(0)
                self.levels[cur] = lvl
                for child in children.get(cur, []):
                    queue.append((child, lvl + 1))
    
        else:
            # --- Automatic initialization from PAGA ---
            conn = self.adata.uns['paga']['connectivities']
            paga_clusters = list(pd.Categorical(self.adata.obs[self.cluster_key]).categories)
            cluster_to_idx = {k: i for i, k in enumerate(paga_clusters)}
            
            G_conn = nx.Graph()
            for c1 in clusters:
                for c2 in clusters:
                    if c1 not in cluster_to_idx or c2 not in cluster_to_idx:
                        continue
                    i, j = cluster_to_idx[c1], cluster_to_idx[c2]
                    w = conn[i, j]
                    if w > 0:
                        G_conn.add_edge(c1, c2, weight=w)

    
            components = list(nx.connected_components(G_conn))
            parent = {}
            levels = {}
            children = {}
    
            for comp in components:
                comp = list(map(str, comp))
                
                if self.root_cluster is not None and str(self.root_cluster) in comp:
                    root_cluster = str(self.root_cluster)
                else:
                    root_cluster = rng.choice(comp)
                
                self.roots.append(root_cluster)
                queue = [(root_cluster, 0)]
                parent[root_cluster] = None
                levels[root_cluster] = 0
                children[root_cluster] = []
    
                while queue:
                    cur, lvl = queue.pop(0)
                    neigh = [n for n in G_conn.neighbors(cur) if n not in parent]
                    if not neigh:
                        continue
                    weights = np.array([G_conn[cur][n]['weight'] for n in neigh])
                    probs = weights / weights.sum()
                    num_children = min(2, len(neigh))
                    chosen = rng.choice(neigh, size=num_children, replace=False, p=probs)
                    children[cur] = list(chosen)
                    for nb in chosen:
                        parent[nb] = cur
                        levels[nb] = lvl + 1
                        queue.append((nb, lvl + 1))
                        children[nb] = []
    
            self.parent = parent
            self.levels = levels
            self.split_nodes = [c for c in children if len(children[c]) > 1]
    
        # --- Construct trajectory graph ---
        def get_node_type(c):
            is_root = (self.parent[c] is None)
            num_children = sum(1 for k, v in self.parent.items() if v == c)
            node_kind = "split" if num_children > 1 else "leaf" if num_children == 0 else "int"
            return f"root_{node_kind}" if is_root else node_kind
    
        for c in clusters:
            label = f"{get_node_type(c)}__CL__{c}"
            self.node_for_cluster[c] = label
            self.cluster_for_node[label] = c
            self.G_traj.add_node(label, type=get_node_type(c))
    
        for c in clusters:
            p = self.parent.get(c, None)
            if p is not None:
                self.G_traj.add_edge(self.node_for_cluster[p], self.node_for_cluster[c], label=str(c))
    
        # Add synthetic root nodes
        comp_map = {cl: i for i, cl in enumerate(self.roots)}
        for c in self.roots:
            comp_idx = comp_map[c]
            root_label = f"RootNode_{comp_idx}"
            self.G_traj.add_node(root_label, type="root_node")
            self.G_traj.add_edge(root_label, self.node_for_cluster[c], label=str(c))
    
        self.edge_list = list(self.G_traj.edges())
        self.branch_probabilities = {e: 1.0 for e in self.edge_list}
    
        max_level = max(self.levels.values()) if self.levels else 1
        for node in self.G_traj.nodes():
            cluster = self.cluster_for_node.get(node, None)
            lvl = self.levels.get(cluster, 0) if cluster else 0
            self.G_traj.nodes[node]['t'] = lvl / max_level if max_level > 0 else 0.0


    def _initialize_transition_probabilities(self, cell_assignment=None):
        """
        Update transition probabilities based on:
        - Emission t-values
        - Observed cell flows
        - Branch probabilities from get_branch_paths_for_edges
        """
    
        self.transition_probabilities = {}
        self.normalizing_constants = {}
    
        edge_counts = {e: 0 for e in self.edge_list}
        if cell_assignment is not None:
            for edge in cell_assignment['edge']:
                edge_counts[edge] += 1
    
        for u in self.G_traj.nodes():
            children = list(self.G_traj.successors(u))
            if not children:
                continue
    
            t_u = self.G_traj.nodes[u].get('t', 0.0)
            # Default uniform edge weights
            base_weights = {v: edge_counts.get((u, v), 1e-3) for v in children}
    
            Z = (1 - t_u) + sum(base_weights.values())  # Eq. 6 from paper
                    
            if Z == 0.0: # Patch for division-by-zero
                Z = 1e-6  # Tiny constant to avoid crash, ensures valid probabilities
    
            for v in children:
                prob = base_weights[v] / Z
                self.transition_probabilities[(u, v)] = prob
                self.G_traj.edges[u, v]['A'] = prob
    
            self.normalizing_constants[u] = Z

    def _compute_edge_segments(self):
        self.edge_segments = []
        for u_name, v_name in self.edge_list:
            t_u = self.G_traj.nodes[u_name].get('t', 0.0)
            t_v = self.G_traj.nodes[v_name].get('t', 1.0)
            self.edge_segments.append((min(t_u, t_v), max(t_u, t_v)))

    def assign_cells_to_paths(self, random_state=0):
        rng = np.random.default_rng(random_state)
        cluster_to_edge = {
            data['label']: (src, dst)
            for src, dst, data in self.G_traj.edges(data=True) if 'label' in data
        }
        assignments = pd.DataFrame(index=self.adata.obs_names)
        cluster_labels = self.adata.obs[self.cluster_key].astype(str)
    
        # ✅ FIXED: assign edge column BEFORE checking for NaNs
        assignments['edge'] = cluster_labels.map(cluster_to_edge)
    
        # Sanity check
        missing = assignments['edge'].isna()
        if missing.any():
            unknown = cluster_labels[missing].unique()
            raise ValueError(f"Unmapped clusters in assignment: {unknown.tolist()}")
    
        assignments['latent_time'] = rng.random(size=self.adata.n_obs)
        return assignments


    def initialize_emission_parameters(self, cell_assignment):
        X = self.adata.X
        cell_to_index = {cell: idx for idx, cell in enumerate(self.adata.obs_names)}
    
        weighted_sum = {node: np.zeros(X.shape[1]) for node in self.G_traj.nodes}
        weight_total = {node: 0.0 for node in self.G_traj.nodes}
    
        for edge in self.edge_list:
            u_name, v_name = edge
            df = cell_assignment[cell_assignment['edge'] == edge]
    
            for cell in df.index:
                t_i = df.loc[cell, 'latent_time']
                expr_i = np.asarray(X[cell_to_index[cell]]).flatten()
                w_u = 1 - t_i
                w_v = t_i
                
                weighted_sum[u_name] += w_u * expr_i
                weight_total[u_name] += w_u
    
                weighted_sum[v_name] += w_v * expr_i
                weight_total[v_name] += w_v
    
        self.node_emission = {
            node: (weighted_sum[node] / weight_total[node]) if weight_total[node] > 0 else np.zeros(X.shape[1])
            for node in self.G_traj.nodes
        }
        
        # --- Initialize per-edge K to ones ---
        n_genes = X.shape[1]
        for edge in self.edge_list:
             self.emission_params[edge] = {'K': np.ones(n_genes)}
        
        # --- Calculate initial global r2 ---
        all_residuals_sq = []
        for edge in self.edge_list:
            u_name, v_name = edge
            df = cell_assignment[cell_assignment['edge'] == edge]
            if df.empty:
                continue
        
            # Use initialized node emissions and K=1 for initial r2
            g_a = self.node_emission[u_name]
            g_b = self.node_emission[v_name]
            # K_init = self.emission_params[edge]['K'] # K is just ones here
            K_init = np.ones(n_genes)
        
            for cell in df.index:
                t_i = df.loc[cell, 'latent_time']
                expr_i = np.asarray(X[cell_to_index[cell]]).flatten()
                # Use exponential model consistent with EMTrainer (even with K=1)
                w_u = np.exp(-K_init * t_i)
                w_v = 1.0 - w_u
                f_t = g_a * w_u + g_b * w_v # Equivalent form
                # f_t = g_b + (g_a - g_b) * np.exp(-K_init * t) # Alternative form
        
                all_residuals_sq.append((expr_i - f_t) ** 2)
        
        if not all_residuals_sq:
            print("[WARN] No cells assigned initially? Setting initial global r2 to ones.")
            self.global_r2 = np.ones(n_genes)
        else:
            all_residuals_sq = np.stack(all_residuals_sq, axis=0)
            self.global_r2 = np.mean(all_residuals_sq, axis=0)
            self.global_r2 = np.clip(self.global_r2, 1e-4, np.inf) # Apply floor
        
        # Return emission_params (which now only contains K per edge)
        return self.emission_params

    def plot_cells_on_trajectory(self, cell_assignment, **kwargs):
        plot_df = cell_assignment.copy()
        return plot_cells_on_trajectory(self.G_traj, plot_df, self.adata, branch_probs=self.branch_probabilities, **kwargs)

def initialize_trajectory(adata, random_state=None, cluster_key='leiden', root_cluster=None, topology_config=None, debug=False):
    traj_graph = TrajectoryGraph(
        adata, 
        cluster_key=cluster_key, 
        random_state=random_state or np.random.randint(0, 2**32 - 1),
        root_cluster=root_cluster,
        topology_config=topology_config
    )
    if hasattr(adata.X, "toarray"):
        adata.X = adata.X.toarray()
    elif hasattr(adata.X, "A"):
        adata.X = adata.X.A
    else:
        adata.X = np.asarray(adata.X)
    # --- Step 1: Get initial assignments ---
    cell_assignment = traj_graph.assign_cells_to_paths(random_state=42)
    # --- Step 2: Initialize emissions (g, K=1) and global r2 ---
    traj_graph.initialize_emission_parameters(cell_assignment)
    # --- Step 3: Initialize transition probabilities (A) based on initial assignments ---
    traj_graph._initialize_transition_probabilities(cell_assignment)

    if debug:
        print("Trajectory graph nodes:", list(traj_graph.G_traj.nodes()))
        print("Trajectory graph edges:", list(traj_graph.G_traj.edges()))
        print("Initial Global r2 (sample):", traj_graph.global_r2[:5])
        print("Initial Transition Probs (sample):", list(traj_graph.transition_probabilities.items())[:5])
        print("\nSample cell assignments (index-based):\n", cell_assignment.head())
    return traj_graph, cell_assignment
    