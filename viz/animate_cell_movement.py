import os
import matplotlib.pyplot as plt
import pandas as pd

def animate_cell_trajectory(
    assignment_history,
    traj_graph,
    output_dir="cell_movement_frames",
    cell_ids=None,
    interval=500,
    dpi=100,
    color_key="cell_type"
):
    """
    Save each EM iteration's cell assignment as a PNG frame.

    Parameters:
        assignment_history (List[pd.DataFrame]): Cell assignment per EM iteration.
        traj_graph (TrajectoryGraph): Graph with plotting logic.
        output_dir (str): Directory to store .png frames.
        cell_ids (List[str] or None): Filter to specific cells if needed.
        interval (int): Frame display delay (used only for naming).
        dpi (int): Image resolution.
        color_key (str): adata.obs key to color cells by.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"🖼️ Saving PNG frames to: {output_dir}")

    for frame_idx, assignment in enumerate(assignment_history):
        assignment_to_plot = assignment.copy()
        if cell_ids:
            assignment_to_plot = assignment_to_plot.loc[assignment_to_plot.index.isin(cell_ids)]

        # Explicitly set figure size using plt.figure before calling the plot function
        plt.figure(figsize=(10, 8))  # <-- Set size here manually
        fig = traj_graph.plot_cells_on_trajectory(
            cell_assignment=assignment_to_plot,
            color_key=color_key,
            node_size=500,
            cell_size=25,
            curve_amount=0.8,
            edge_width=3,
            edge_color='gray',
            plot_transitions=True,
            title=f"EM Iteration {frame_idx + 1}",
        )

        frame_path = os.path.join(output_dir, f"frame_{frame_idx:03d}.png")
        fig.savefig(frame_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)

    print("✅ All frames saved.")
