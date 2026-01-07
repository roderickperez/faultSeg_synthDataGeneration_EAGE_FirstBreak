import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def plot_fault_counts(fault_params_list):
    """Plots bar chart of total, normal, and inverse fault counts."""
    if not fault_params_list:
        print("No fault data to plot counts.")
        return None

    normal_count = sum(1 for f in fault_params_list if f['fault_type'] == 'Normal')
    inverse_count = sum(1 for f in fault_params_list if f['fault_type'] == 'Inverse')
    total_count = len(fault_params_list)

    labels = ['Total', 'Normal', 'Inverse']
    counts = [total_count, normal_count, inverse_count]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(labels, counts, color=['skyblue', 'lightgreen', 'salmon'])
    ax.set_ylabel('Number of Faults')
    ax.set_title('Fault Type Distribution Across All Cubes')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    for i, count in enumerate(counts):
        ax.text(i, count + max(counts) * 0.02, str(count), ha='center')
    plt.tight_layout()
    return fig

def plot_histogram(data, ax=None, title=None, xlabel=None, bins='auto'):
    """Plots a histogram on a given axes object."""
    if not data or (isinstance(data, list) and not any(x is not None for x in data)):
        print(f"No data to plot histogram for '{title}'.")
        return

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    filtered_data = [x for x in data if x is not None]
    if not filtered_data:
         print(f"No valid (non-None) data to plot histogram for '{title}'.")
         return

    ax.hist(filtered_data, bins=bins, edgecolor='black')
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_ylabel('Frequency')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    if 'fig' in locals() and ax.figure == fig:
        fig.tight_layout()

    return ax.figure

def plot_rose_diagram(strikes_deg, ax=None, title="Fault Strike Angle Distribution (Rose Diagram)"):
    """Plots a rose diagram for strike angles on a given axes object."""
    if not strikes_deg:
        print("No strike data to plot for rose diagram.")
        return

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
    else:
        fig = ax.figure

    strikes_rad = np.radians(strikes_deg)
    num_bins = 18
    bin_edges_deg = np.linspace(0, 360, num_bins + 1)
    bin_edges_rad = np.radians(bin_edges_deg)
    counts, _ = np.histogram(strikes_rad, bins=bin_edges_rad)
    bin_centers_rad = bin_edges_rad[:-1] + np.diff(bin_edges_rad)/2
    width = np.diff(bin_edges_rad)[0]
    ax.bar(bin_centers_rad, counts, width=width, bottom=0.0, align='center', alpha=0.7, color='skyblue', edgecolor='black')
    ax.set_title(title, va='bottom', y=1.1)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(135)
    ax.set_xticks(np.radians(np.arange(0, 360, 30)))
    ax.set_xticklabels([f'{i}°' for i in np.arange(0, 360, 30)])
    ax.grid(True)

    if 'fig' in locals() and ax.figure == fig:
        fig.tight_layout()

    return ax.figure

def append_param_to_cmd(cmd_list, arg_name, value):
    if value is not None:
        if isinstance(value, tuple):
            cmd_list += [f"--{arg_name}", f"{value[0]},{value[1]}"]
        else:
            cmd_list += [f"--{arg_name}", str(value)]

def normalize(v):
    return ((v - v.min()) / (v.max() - v.min()) * 255).astype(np.uint8)

def plot_fault_points(fig, slice_mask, axis_index, slice_type, color, label, nx, ny, nz):
    if slice_type == "inline":
        ks, js = np.where(slice_mask)
        fig.add_trace(go.Scatter3d(
            x=np.full_like(ks, axis_index),
            y=js,
            z=nz - 1 - ks,
            mode='markers',
            marker=dict(color=color, size=2),
            name=label
        ))
    elif slice_type == "crossline":
        ks, is_ = np.where(slice_mask)
        fig.add_trace(go.Scatter3d(
            x=is_,
            y=np.full_like(is_, axis_index),
            z=nz - 1 - ks,
            mode='markers',
            marker=dict(color=color, size=2),
            name=label
        ))
    elif slice_type == "timeslice":
        is_, js = np.where(slice_mask)
        fig.add_trace(go.Scatter3d(
            x=is_,
            y=js,
            z=np.full_like(is_, axis_index),
            mode='markers',
            marker=dict(color=color, size=2),
            name=label
        ))

def plot_3d_interactive(seismic_volume, fault_mask, title="Interactive 3D Fault Planes", downsample_factor=1):
    """
    Plots interactive 3D visualization of seismic data and fault masks with toggles.
    Supports multiclass fault visualization (0=background, 1=Normal, 2=Inverse).
    
    Args:
        seismic_volume: 3D numpy array of seismic data
        fault_mask: 3D numpy array of fault mask
        title: Plot title
        downsample_factor: Factor to downsample data for performance (default 1 for max detail)
    """
    
    # Downsample for performance if needed (default 1 to keep thin faults)
    s_vol = seismic_volume[::downsample_factor, ::downsample_factor, ::downsample_factor]
    f_mask = fault_mask[::downsample_factor, ::downsample_factor, ::downsample_factor]
    
    X, Y, Z = np.mgrid[0:s_vol.shape[0], 0:s_vol.shape[1], 0:s_vol.shape[2]]
    
    # Create figure
    fig = go.Figure()
    
    # 1. Seismic Volume (Volume rendering)
    fig.add_trace(go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=s_vol.flatten(),
        isomin=np.percentile(s_vol, 10),
        isomax=np.percentile(s_vol, 90),
        opacity=0.1, 
        surface_count=15, 
        colorscale='RdBu',
        name='Seismic',
        visible=True
    ))
    
    # Check if multiclass from data
    unique_vals = np.unique(f_mask)
    is_multiclass = np.any(unique_vals > 1)
    
    # 2. Fault Points (Scatter3d)
    if is_multiclass:
        # Class 1: Normal (Green)
        fx1, fy1, fz1 = np.where(f_mask == 1)
        if len(fx1) > 0:
            if len(fx1) > 5000:
                idx = np.random.choice(len(fx1), 5000, replace=False)
                fx1, fy1, fz1 = fx1[idx], fy1[idx], fz1[idx]
            fig.add_trace(go.Scatter3d(
                x=fx1, y=fy1, z=fz1, mode='markers',
                marker=dict(size=2, color='green', opacity=0.8),
                name='Normal Fault Points', visible=False
            ))
        
        # Class 2: Inverse (Purple)
        fx2, fy2, fz2 = np.where(f_mask == 2)
        if len(fx2) > 0:
            if len(fx2) > 5000:
                idx = np.random.choice(len(fx2), 5000, replace=False)
                fx2, fy2, fz2 = fx2[idx], fy2[idx], fz2[idx]
            fig.add_trace(go.Scatter3d(
                x=fx2, y=fy2, z=fz2, mode='markers',
                marker=dict(size=2, color='purple', opacity=0.8),
                name='Inverse Fault Points', visible=False
            ))
    else:
        # Binary (Red)
        fx, fy, fz = np.where(f_mask > 0)
        if len(fx) > 0:
            if len(fx) > 10000:
                idx = np.random.choice(len(fx), 10000, replace=False)
                fx, fy, fz = fx[idx], fy[idx], fz[idx]
            fig.add_trace(go.Scatter3d(
                x=fx, y=fy, z=fz, mode='markers',
                marker=dict(size=2, color='red', opacity=0.8),
                name='Fault Points', visible=False
            ))

    # 3. Fault Planes (Isosurface)
    if is_multiclass:
        # Class 1: Normal (Green)
        mask1 = (f_mask == 1).astype(int)
        if np.sum(mask1) > 0:
            fig.add_trace(go.Isosurface(
                x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
                value=mask1.flatten(),
                isomin=0.5, isomax=1,
                opacity=0.7, surface_count=2,
                colorscale=[[0, 'green'], [1, 'green']], 
                showscale=False,
                name='Normal Fault Planes', visible=True
            ))
            
        # Class 2: Inverse (Purple)
        mask2 = (f_mask == 2).astype(int)
        if np.sum(mask2) > 0:
            fig.add_trace(go.Isosurface(
                x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
                value=mask2.flatten(),
                isomin=0.5, isomax=1,
                opacity=0.7, surface_count=2,
                colorscale=[[0, 'purple'], [1, 'purple']], 
                showscale=False,
                name='Inverse Fault Planes', visible=True
            ))
    else:
        # Binary (Red)
        if np.sum(f_mask) > 0:
            fig.add_trace(go.Isosurface(
                x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
                value=f_mask.flatten(),
                isomin=0.5, isomax=f_mask.max(),
                opacity=0.7, surface_count=2,
                colorscale=[[0, 'red'], [1, 'red']], # Force red for binary
                showscale=False,
                name='Fault Planes', visible=True
            ))

    # Helper to create visibility list based on actual traces present
    trace_names = [t.name for t in fig.data]
    
    def make_vis(seismic=True, points=False, planes=False):
        vis = []
        for name in trace_names:
            if 'Seismic' in name:
                vis.append(seismic)
            elif 'Points' in name:
                vis.append(points)
            elif 'Planes' in name:
                vis.append(planes)
            else:
                vis.append(True)
        return vis

    fig.update_layout(
        title=title,
        scene=dict(xaxis_title='Inline', yaxis_title='Crossline', zaxis_title='Time/Depth'),
        updatemenus=[
            dict(
                type="buttons", direction="left",
                buttons=list([
                    dict(label="Default", method="update", args=[{"visible": make_vis(seismic=True, points=False, planes=True)}]),
                    dict(label="Seismic Only", method="update", args=[{"visible": make_vis(seismic=True, points=False, planes=False)}]),
                    dict(label="Seismic + Points", method="update", args=[{"visible": make_vis(seismic=True, points=True, planes=False)}]),
                    dict(label="Points Only", method="update", args=[{"visible": make_vis(seismic=False, points=True, planes=False)}]),
                    dict(label="Planes Only", method="update", args=[{"visible": make_vis(seismic=False, points=False, planes=True)}]),
                    dict(label="All", method="update", args=[{"visible": make_vis(seismic=True, points=True, planes=True)}]),
                ]),
                pad={"r": 10, "t": 10}, showactive=True, x=0.05, xanchor="left", y=1.1, yanchor="top"
            ),
        ]
    )
    
    return fig
