import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import networkx as nx
from scipy.stats import beta

from utils import cubic_bezier, cubic_bezier_derivative

def compute_bezier_control_points(P0, P3, curve_amount):
    P0 = np.asarray(P0)
    P3 = np.asarray(P3)
    return (
        P0 + (P3 - P0) / 3.0 + curve_amount * np.array([0, 1]) * (P3[1] - P0[1]),
        P0 + 2 * (P3 - P0) / 3.0 - curve_amount * np.array([0, 1]) * (P3[1] - P0[1])
    )


def get_node_color(node_attrs):
    ntype = node_attrs.get('type', '')
    mapping = {'root': 'red', 'split': 'orange', 'leaf': 'lightgreen'}
    for key in mapping:
        if key in ntype:
            return mapping[key]
    return 'lightblue'

def format_node_label(node_id, node_attrs):
    ntype = node_attrs.get('type', '')
    cluster = node_id.split('_')[-1]
    prefix = {'root': 'R', 'split': 'S', 'leaf': 'L'}.get(ntype.split('_')[-1], 'I')
    return f"{prefix}{cluster}"

def compute_graph_layout(G_traj):
    if hasattr(G_traj, "G_traj"):
        G_traj = G_traj.G_traj
    return nx.nx_agraph.graphviz_layout(G_traj, prog="dot", args="-Grankdir=LR")


def draw_graph_base(G_traj, pos, node_size=500, edge_width=5, edge_color='lightgrey', horizontal_thresh=0.01, curve_amount=0.8):
    if hasattr(G_traj, "G_traj"):
        G_traj = G_traj.G_traj
    edge_midpoints = []
    for node, (x, y) in pos.items():
        color = get_node_color(G_traj.nodes[node])
        label = format_node_label(node, G_traj.nodes[node])
        plt.scatter(x, y, color=color, s=node_size, zorder=3)
        plt.text(x, y, label, fontsize=10, ha="center", va="center", zorder=4)

    for src, dst in G_traj.edges():
        P0, P1 = pos[src], pos[dst]
        vertical_diff = abs(P1[1] - P0[1])
        is_horizontal = vertical_diff < horizontal_thresh * abs(P1[0] - P0[0])
        if is_horizontal:
            plt.plot([P0[0], P1[0]], [P0[1], P1[1]], color=edge_color, lw=edge_width, alpha=0.8, zorder=2)
            mid = ((P0[0] + P1[0]) / 2, (P0[1] + P1[1]) / 2)
        else:
            cp1, cp2 = compute_bezier_control_points(P0, P1, curve_amount)
            bez = cubic_bezier(P0, cp1, cp2, P1, np.array([0.5]))[0]
            plt.plot(*cubic_bezier(P0, cp1, cp2, P1, np.linspace(0, 1, 100)).T, color=edge_color, lw=edge_width, alpha=0.8, zorder=2)
            mid = bez
        edge_midpoints.append(((src, dst), mid))
    return edge_midpoints

def get_cell_color_map(adata, color_key, label_color_map=None):
    if color_key in adata.obs.columns:
        series = adata.obs[color_key]
    elif color_key in adata.var_names:
        gene_idx = adata.var_names.get_loc(color_key)
        X = adata.X[:, gene_idx]
        values = X.toarray().ravel() if hasattr(X, "toarray") else np.array(X).ravel()
        series = pd.Series(values, index=adata.obs_names)
    else:
        raise ValueError(f"color_key '{color_key}' not found in adata.obs or adata.var_names")

    if pd.api.types.is_numeric_dtype(series):
        norm = mcolors.Normalize(vmin=series.min(), vmax=series.max())
        cmap = cm.viridis
        return series, lambda val: cmap(norm(val)), True, cmap, norm
    else:
        cats = sorted(series.unique())
        if label_color_map:
            cat2color = {cat: label_color_map.get(cat, 'gray') for cat in cats}
        else:
            palette = cm.tab10.colors if len(cats) <= 10 else cm.tab20.colors
            cat2color = {cat: palette[i % len(palette)] for i, cat in enumerate(cats)}
        return series, lambda val: cat2color[val], False, cat2color, None


def plot_cells_on_trajectory(
    G_traj, assignments, adata, color_key='leiden',
    curve_amount=0.8, node_size=500, cell_size=30,
    horizontal_thresh=0.01, edge_width=5, edge_color='lightgrey',
    title="Cells on Stochastic PAGA Trajectory", branch_probs=None,
    plot_transitions=False, savepath=None, label_color_map=None
):

    """
    Visualize hard cell assignments on a trajectory graph.

    Each cell is assigned to a specific edge and a latent time (0-1) along that edge.
    Cells are plotted over a curved or straight PAGA-style graph, colored by a metadata key.

    Args:
        G_traj (nx.DiGraph or TrajectoryGraph): Directed trajectory graph.
        assignments (pd.DataFrame): Must contain 'edge' and 'latent_time' columns.
        adata (AnnData): Annotated data matrix (used for coloring).
        color_key (str): Key in adata.obs or adata.var_names for coloring cells.
        curve_amount (float): Bézier curve bend intensity (0 = straight).
        node_size (int): Size of graph nodes.
        cell_size (int): Size of individual cell markers.
        horizontal_thresh (float): Threshold for determining "straight" edges.
        edge_width (int): Width of the base graph edges.
        edge_color (str): Color for graph edges.
        title (str): Plot title.
        branch_probs (dict): Optional mapping of edge → branch probability.
        plot_transitions (bool): Whether to annotate transition matrix on edges.
        savepath (str or Path): Optional. If provided, saves the plot to this path.
    """
    if hasattr(G_traj, "G_traj"):
        G_traj = G_traj.G_traj

    pos = compute_graph_layout(G_traj)
    plt.figure(figsize=(12, 7))

    edge_midpoints = draw_graph_base(
        G_traj, pos,
        node_size=node_size,
        edge_width=edge_width,
        edge_color=edge_color,
        horizontal_thresh=horizontal_thresh,
        curve_amount=curve_amount
    )

    color_vals, get_color, is_continuous, color_info, norm = get_cell_color_map(adata, color_key, label_color_map)
    
    cell_x, cell_y, cell_colors = [], [], []
    for cell_id, row in assignments.iterrows():
        edge = row['edge']
        latent_time = row['latent_time']
        if edge and edge[0] in pos and edge[1] in pos:
            P0, P1 = pos[edge[0]], pos[edge[1]]
            cp1, cp2 = compute_bezier_control_points(P0, P1, curve_amount)
            pt = cubic_bezier(P0, cp1, cp2, P1, np.array([latent_time]))[0]
            cell_x.append(pt[0])
            cell_y.append(pt[1])
        else:
            cell_x.append(0)
            cell_y.append(0)
        val = color_vals.loc[cell_id]
        val = val.iloc[0] if isinstance(val, pd.Series) else val
        cell_colors.append(get_color(val))

    plt.scatter(cell_x, cell_y, color=cell_colors, s=cell_size, edgecolor='black', linewidth=0.5, zorder=5)

    if is_continuous:
        sm = cm.ScalarMappable(cmap=color_info, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, label=color_key)
    else:
        handles = [plt.Line2D([0], [0], marker='o', color='w', label=cat,
                              markerfacecolor=color_info[cat], markersize=8)
                   for cat in sorted(color_info)]
        plt.legend(handles=handles, title=color_key, bbox_to_anchor=(1.05, 1), loc='upper left')

    for (src, dst), mid in edge_midpoints:
        edge_label = G_traj.edges[src, dst].get('label', '')
        if branch_probs:
            bp = branch_probs.get((src, dst), None)
            if bp is not None:
                edge_label += f"\nB: {bp:.2f}"
        if plot_transitions and 'split' in G_traj.nodes[src].get('type', ''):
            A_val = G_traj.edges[src, dst].get('A', None)
            if A_val is not None:
                edge_label += f"\nA: {A_val:.2f}"
        if edge_label:
            circle = plt.Circle(mid, radius=node_size * 0.015, edgecolor=edge_color,
                                facecolor='white', zorder=7)
            plt.gca().add_patch(circle)
            plt.text(mid[0], mid[1], edge_label, fontsize=9, color="black", ha="center", va="center", zorder=8)

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    if savepath:
        import os
        os.makedirs(os.path.dirname(savepath), exist_ok=True) if os.path.dirname(savepath) else None
        ext = os.path.splitext(savepath)[1].lower()
        valid_exts = [".png", ".pdf", ".svg", ".jpg", ".jpeg", ".tiff"]
        if ext not in valid_exts:
            raise ValueError(f"Unsupported file format: '{ext}'. Must be one of {valid_exts}")
        plt.savefig(savepath, dpi=300, bbox_inches='tight')

    fig = plt.gcf()
    plt.show()
    return fig

    


def draw_pdf_height_legend(ax, pdf_scale):
    heights = [1.0, 0.5, 0.25]
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    base_x = xlim[0] + 0.05 * (xlim[1] - xlim[0])
    base_y = ylim[0] + 0.05 * (ylim[1] - ylim[0])
    reference_size = 0.1 * (ylim[1] - ylim[0])  # base visual size

    spacing_factor = 0.05 * (xlim[1] - xlim[0])  # space between circles

    current_x = base_x
    for h in heights:
        radius = pdf_scale * reference_size * h
        center = (current_x + radius, base_y + radius)

        # Draw circle
        circle = plt.Circle(center, radius, color='gray', alpha=1)
        ax.add_patch(circle)

        # Label below
        ax.text(center[0], base_y - 0.02 * (ylim[1] - ylim[0]), f"{h:.2f}",
                fontsize=9, ha='center', va='top')

        # Move x for next circle
        current_x += 2 * radius + spacing_factor

    ax.text(base_x, base_y + 0.15 * (ylim[1] - ylim[0]), "PDF Height Legend",
            fontsize=10, ha='left', va='bottom')


def plot_pdf_width_ribbons_on_trajectory(
    G_traj, beta_assignments, adata, color_key='leiden',
    node_size=500, edge_width=5, edge_color='lightgrey',
    resolution=100, pdf_scale=0.1, curve_amount=0.8,
    title="PDF Width Ribbons on Trajectory", savepath=None
):
    """
    Visualize probabilistic (beta-distributed) cell assignments as ribbon widths on a trajectory graph.

    Each cell contributes a beta PDF over latent time along an edge. These are visualized
    as thick ribbons whose width at each point is proportional to PDF height, aligned with the trajectory.

    Args:
        G_traj (nx.DiGraph or TrajectoryGraph): Directed graph with trajectory structure.
        beta_assignments (pd.DataFrame): Must contain 'edge', 'alpha', 'beta' columns per cell.
        adata (AnnData): Annotated data matrix for coloring cells.
        color_key (str): Key in adata.obs or adata.var_names to color ribbons by.
        node_size (int): Size of graph nodes.
        edge_width (int): Width of graph edges.
        edge_color (str): Color of graph edges.
        resolution (int): Number of points to evaluate PDFs per edge.
        pdf_scale (float): Height-to-width scaling for beta PDFs.
        curve_amount (float): Bézier curvature strength (0 = straight).
        title (str): Plot title.
        savepath (str or Path): Optional. If provided, saves the figure to this path.
    """
    if hasattr(G_traj, 'G_traj'):
        G_traj = G_traj.G_traj

    pos = compute_graph_layout(G_traj)
    plt.figure(figsize=(12, 7))

    draw_graph_base(
        G_traj, pos,
        node_size=node_size,
        edge_width=edge_width,
        edge_color=edge_color,
        horizontal_thresh=0.01,
        curve_amount=curve_amount
    )

    color_vals, get_color, is_continuous, color_info, norm = get_cell_color_map(adata, color_key)
    t_vals = np.linspace(0, 1, resolution)

    df = beta_assignments.copy()
    df = df[df['alpha'] > 0]
    df = df[df['beta'] > 0]
    df = df[df['edge'].notnull()]
    df = df.loc[df.index.intersection(color_vals.index)]

    for edge, group in df.groupby('edge'):
        src, dst = edge
        if src not in pos or dst not in pos:
            continue

        P0, P3 = np.array(pos[src]), np.array(pos[dst])
        direction = P3 - P0
        seg_length = np.linalg.norm(direction)
        if seg_length == 0:
            continue

        if curve_amount == 0.0:
            points = np.outer(t_vals, direction) + P0
            tangents = np.tile(direction / seg_length, (resolution, 1))
        else:
            cp1, cp2 = compute_bezier_control_points(P0, P3, curve_amount)
            points = cubic_bezier(P0, cp1, cp2, P3, t_vals)
            tangents = cubic_bezier_derivative(P0, cp1, cp2, P3, t_vals)
            tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)

        normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)

        alphas = group['alpha'].values[:, None]
        betas = group['beta'].values[:, None]
        pdfs = beta.pdf(t_vals[None, :], alphas, betas)
        pdfs = np.nan_to_num(pdfs)
        max_vals = pdfs.max(axis=1, keepdims=True)
        max_vals[max_vals == 0] = 1
        widths = pdfs * (pdf_scale * seg_length) / max_vals

        group_indices = group.index.to_numpy()
        for i, width in enumerate(widths):
            offset = (width[:, None] / 2) * normals
            top = points + offset
            bottom = points - offset
            ribbon = np.vstack([top, bottom[::-1]])

            val = color_vals.loc[group_indices[i]]
            val = val.iloc[0] if isinstance(val, pd.Series) else val
            color = get_color(val)

            plt.fill(ribbon[:, 0], ribbon[:, 1], color=color, alpha=1, lw=0)

    if is_continuous:
        sm = cm.ScalarMappable(cmap=color_info, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, label=color_key)
    else:
        handles = [plt.Line2D([0], [0], marker='s', color='w', label=cat,
                              markerfacecolor=color_info[cat], markersize=10)
                   for cat in sorted(color_info)]
        plt.legend(handles=handles, title=color_key, bbox_to_anchor=(1.05, 1), loc='upper left')

    ax = plt.gca()
    draw_pdf_height_legend(ax, pdf_scale=pdf_scale)

    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    if savepath:
        import os
        os.makedirs(os.path.dirname(savepath), exist_ok=True) if os.path.dirname(savepath) else None
        ext = os.path.splitext(savepath)[1].lower()
        valid_exts = [".png", ".pdf", ".svg", ".jpg", ".jpeg", ".tiff"]
        if ext not in valid_exts:
            raise ValueError(f"Unsupported file format: '{ext}'. Must be one of {valid_exts}")
        plt.savefig(savepath, dpi=300, bbox_inches='tight')

    plt.show()

