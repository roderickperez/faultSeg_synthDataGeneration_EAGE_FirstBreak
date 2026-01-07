import marimo

__generated_with = "0.18.4"
app = marimo.App(
    width="columns",
    layout_file="layouts/seismicFaultSegmentation2D_V2.grid.json",
)


@app.cell(column=0, hide_code=True)
def _(mo):
    mo.md(r"""
    # Import Libraries
    """)
    return


@app.cell
def _():
    import marimo as mo
    import os
    import json
    import numpy as np
    import pandas as pd
    import scipy.ndimage as ndimage
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, Normalize
    import matplotlib.patches as mpatches
    return ListedColormap, json, mo, mpatches, ndimage, np, os, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 1. Configuration & Parameters

    Adjust the parameters below and click **Generate Dataset** to run the simulation.
    """)
    return


@app.cell
def _(mo):
    # --- UI DEFINITIONS ---

    # 1. Dataset & Dim
    n_total_slider = mo.ui.slider(start=1, stop=50, value=10, label="Total Pairs")
    is_3d_toggle = mo.ui.switch(value=True, label="Generate 3D Volume")
    # n goes 32, 64, 96, 128, ... 256
    n_slider = mo.ui.slider(start=32, stop=256, step=32, value=128, label="Grid Size (n)")

    # 2. Layers
    # Double slider simulation using two sliders
    n_layers_min = mo.ui.slider(start=1, stop=10, value=3, label="Min Layers")
    n_layers_max = mo.ui.slider(start=1, stop=20, value=6, label="Max Layers")

    rand_thick_radio = mo.ui.radio(options={"Yes": True, "No": False}, value="Yes", label="Random Thickness", inline=True)
    thick_min = mo.ui.slider(start=5, stop=50, value=10, label="Min Thickness (px)")
    thick_max = mo.ui.slider(start=5, stop=100, value=30, label="Max Thickness (px)")

    # 3. Domes
    n_domes_min = mo.ui.slider(start=0, stop=5, value=1, label="Min Domes")
    n_domes_max = mo.ui.slider(start=0, stop=10, value=3, label="Max Domes")
    random_dome_loc = mo.ui.checkbox(value=False, label="Random Dome Location")

    # 4. Faults
    binary_seg_switch = mo.ui.switch(value=False, label="Binary Segmentation (0/1)")
    n_faults_min = mo.ui.slider(start=0, stop=5, value=1, label="Min Faults")
    n_faults_max = mo.ui.slider(start=0, stop=10, value=1, label="Max Faults")

    # Strike multiselect
    strike_options = {
        "N-S (0)": (0, 0),
        "E-W (90)": (90, 90),
        "Diagonal (45)": (45, 45),
        "Random (0-360)": (0, 360)
    }
    strike_multi = mo.ui.multiselect(
        options=strike_options, 
        value=["N-S (0)", "E-W (90)"], 
        label="Strike Ranges"
    )

    throw_min = mo.ui.number(start=-100, stop=100, value=-20, label="Max Throw (Neg=Reverse)")
    throw_max = mo.ui.number(start=-100, stop=100, value=20, label="Max Throw (Pos=Normal)")

    # 5. Signal
    noise_min = mo.ui.slider(start=0.0, stop=0.5, step=0.01, value=0.01, label="Min Noise")
    noise_max = mo.ui.slider(start=0.0, stop=0.5, step=0.01, value=0.1, label="Max Noise")
    freq_min = mo.ui.slider(start=5, stop=60, value=25, label="Min Freq (Hz)")
    freq_max = mo.ui.slider(start=5, stop=60, value=30, label="Max Freq (Hz)")

    # Grouping into a Form so it doesn't regenerate on every slide
    config_form = mo.ui.form(
        element=mo.ui.array([
            n_total_slider, is_3d_toggle, n_slider,
            n_layers_min, n_layers_max, rand_thick_radio, thick_min, thick_max,
            n_domes_min, n_domes_max, random_dome_loc,
            binary_seg_switch, n_faults_min, n_faults_max, strike_multi, throw_min, throw_max,
            noise_min, noise_max, freq_min, freq_max
        ]),
        label="Configuration & Generation",
        submit_button_label="Generate Dataset"
    )
    return (config_form,)


@app.cell
def _(config_form, mo):
    # Display the form
    mo.vstack([
        mo.md("### ⚙️ Simulation Control"),
        config_form
    ])
    return


@app.cell
def _(config_form):
    # --- EXTRACT VARIABLES FROM FORM ---
    # Accessing via index of the array passed to the form
    _vals = config_form.value

    # 1. Dataset
    N_TOTAL_PAIRS = _vals[0]
    IS_3D = _vals[1]
    n = _vals[2]
    DIMS = (n, n, n) if IS_3D else (n, n)

    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.2
    OUTPUT_DIR = "dataset_flat"
    OUTPUT_FORMAT = 'npy'

    # 2. Layers
    NUMBER_LAYERS_RANGE = (_vals[3], _vals[4])
    RANDOM_LAYER_THICKNESS = _vals[5]
    LAYER_THICKNESS_RANGE = (_vals[6], _vals[7])

    # 3. Domes
    N_DOMES_RANGE = (_vals[8], _vals[9])
    RANDOM_DOME_LOCATION = _vals[10]
    DOME_CENTER = (int(n/2), int(n/2), int(n/2)) # Auto-center based on n
    SIGMA_K_RANGE = (15, 25)
    MAX_FOLD_RANGE = (15.0, 25.0)

    # 4. Faults
    BINARY_SEGMENTATION = _vals[11]
    N_FAULTS_RANGE = (_vals[12], _vals[13])
    STRIKE_RANGE = _vals[14] # Multiselect returns list of values
    THROW_RANGE = [_vals[15], _vals[16]]

    DIP_RANGE = (45, 60)
    FAULT_RADIUS_RANGE = (int(n*1.5), int(n*1.5)) # Scale fault to grid
    FAULT_THICKNESS = 1

    # 5. Signal
    NOISE_LEVEL_RANGE = (_vals[17], _vals[18])
    FREQ_RANGE = (_vals[19], _vals[20])

    # 6. Stats
    SAVE_STATISTICS = True
    SHOW_STATISTICS_TABLE = True
    SHOW_STATISTICS_PLOT = True
    return (
        BINARY_SEGMENTATION,
        DIMS,
        DIP_RANGE,
        DOME_CENTER,
        FAULT_RADIUS_RANGE,
        FAULT_THICKNESS,
        FREQ_RANGE,
        IS_3D,
        LAYER_THICKNESS_RANGE,
        MAX_FOLD_RANGE,
        NOISE_LEVEL_RANGE,
        NUMBER_LAYERS_RANGE,
        N_DOMES_RANGE,
        N_FAULTS_RANGE,
        N_TOTAL_PAIRS,
        OUTPUT_DIR,
        RANDOM_DOME_LOCATION,
        RANDOM_LAYER_THICKNESS,
        SAVE_STATISTICS,
        SHOW_STATISTICS_PLOT,
        SHOW_STATISTICS_TABLE,
        SIGMA_K_RANGE,
        STRIKE_RANGE,
        THROW_RANGE,
        TRAIN_RATIO,
    )


@app.cell
def _(mo):
    # --- VISUALIZATION WIDGETS (Instant Update) ---

    viz_index_slider = mo.ui.number(start=0, stop=100, value=0, label="Sample Index")
    viz_slice_slider = mo.ui.slider(start=0, stop=256, value=64, label="Slice Index (3D)")

    viz_axis_check = mo.ui.checkbox(value=True, label="Show Axis")
    viz_cross_check = mo.ui.checkbox(value=True, label="Show Crosshairs")
    viz_legend_check = mo.ui.checkbox(value=True, label="Show Legend")

    color_options = ['red', 'blue', 'yellow', 'green', 'cyan', 'magenta', 'white', 'black']

    inline_col = mo.ui.dropdown(options=color_options, value='red', label="Inline Color")
    xline_col = mo.ui.dropdown(options=color_options, value='blue', label="Xline Color")
    time_col = mo.ui.dropdown(options=color_options, value='yellow', label="Time Color")

    viz_settings = mo.md(f"""
    ### 👁️ Visualization Settings
    {mo.hstack([viz_index_slider, viz_slice_slider], justify='start')}
    {mo.hstack([viz_axis_check, viz_cross_check, viz_legend_check], justify='start')}
    {mo.hstack([inline_col, xline_col, time_col], justify='start')}
    """).callout(kind="info")

    return (
        inline_col,
        time_col,
        viz_axis_check,
        viz_cross_check,
        viz_index_slider,
        viz_legend_check,
        viz_settings,
        viz_slice_slider,
        xline_col,
    )


@app.cell
def _(mo, viz_settings):
    mo.output.replace(viz_settings)
    return


@app.cell
def _(
    inline_col,
    time_col,
    viz_axis_check,
    viz_cross_check,
    viz_index_slider,
    viz_legend_check,
    viz_slice_slider,
    xline_col,
):
    # Extract Visualization Values directly (no form needed for instant update)
    VISUALIZE_SAMPLE_INDEX = viz_index_slider.value
    VISUALIZE_SLICE_TYPE = viz_slice_slider.value # integer value

    SHOW_AXIS_NUMBERS = viz_axis_check.value
    SHOW_CROSSHAIRS = viz_cross_check.value
    SHOW_CROSSHAIRS_LEGEND = viz_legend_check.value

    INLINE_COLOR = inline_col.value
    XLINE_COLOR = xline_col.value
    TIMESLICE_COLOR = time_col.value

    CROSSHAIR_THICKNESS = 1.5
    return (
        CROSSHAIR_THICKNESS,
        INLINE_COLOR,
        SHOW_AXIS_NUMBERS,
        SHOW_CROSSHAIRS,
        SHOW_CROSSHAIRS_LEGEND,
        TIMESLICE_COLOR,
        VISUALIZE_SAMPLE_INDEX,
        VISUALIZE_SLICE_TYPE,
        XLINE_COLOR,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Directories Setup
    """)
    return


@app.cell
def _(OUTPUT_DIR, os):
    def setup_directories(base_dir):
        """Creates the directory structure."""
        for split in ['train', 'validation']:
            for type_ in ['seismic', 'mask']:
                path = os.path.join(base_dir, split, type_)
                os.makedirs(path, exist_ok=True)

        # Stats directories
        os.makedirs(os.path.join(base_dir, "statistics"), exist_ok=True)

    setup_directories(OUTPUT_DIR)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 2. Reflectivity, RGT, and Folding
    """)
    return


@app.cell
def _(ndimage, np):
    def apply_displacement(data, displacement, axis=0):
        """Warps data (2D or 3D) using cubic interpolation."""
        coords = np.indices(data.shape).astype(float)
        coords[axis] -= displacement
        return ndimage.map_coordinates(data, coords, order=1, mode='nearest')

    def generate_reflectivity(dims, number_layers, random_thickness=False, thickness_range=(10, 30)):
        """Generates layered reflectivity and RGT."""
        nz = dims[0]

        # 1. Generate RGT
        if len(dims) == 3:
            rgt = np.linspace(0, 1, nz)[:, None, None] * np.ones(dims)
        else:
            rgt = np.linspace(0, 1, nz)[:, None] * np.ones(dims)

        # 2. Generate Reflectivity
        r = np.zeros(dims)

        layer_indices = []
        if random_thickness:
            current_z = 10
            while current_z < (nz - 10):
                layer_indices.append(int(current_z))
                thickness = np.random.randint(thickness_range[0], thickness_range[1] + 1)
                current_z += thickness
        else:
            n_layers = max(1, number_layers)
            layer_indices = np.linspace(10, nz-10, n_layers, dtype=int)

        if len(dims) == 3:
            for idx in layer_indices:
                r[idx, :, :] = np.random.choice([-1, 1])
        else:
            for idx in layer_indices:
                r[idx, :] = np.random.choice([-1, 1])

        return r, rgt

    def add_folding_and_shearing(img, rgt, dims, is_3d, n_domes, sigma_k_range, max_fold_range, random_loc=True, manual_center=None):
        """
        Adds discrete dome structures. Uses RANGES for sigma and fold.
        """
        if n_domes == 0:
            return img, rgt

        total_disp = np.zeros(dims)
        coords = np.indices(dims)

        for _ in range(n_domes):
            # Sample parameters from ranges
            sigma_k = np.random.uniform(sigma_k_range[0], sigma_k_range[1])
            max_fold = np.random.uniform(max_fold_range[0], max_fold_range[1])

            if max_fold == 0: continue

            # Determine Center
            if random_loc:
                c = [np.random.randint(0, d) for d in dims]
            else:
                if manual_center is not None and len(manual_center) == len(dims):
                    c = manual_center
                else:
                    c = [d // 2 for d in dims]

            # Calculate Distance
            dist_sq = np.zeros(dims)
            for i, coord_grid in enumerate(coords):
                dist_sq += (coord_grid - c[i])**2

            # Gaussian Dome
            dome_shape = max_fold * np.exp(-dist_sq / (2 * sigma_k**2))
            total_disp += dome_shape

        img = apply_displacement(img, total_disp, axis=0)
        rgt = apply_displacement(rgt, total_disp, axis=0)

        return img, rgt
    return add_folding_and_shearing, apply_displacement, generate_reflectivity


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 3. Finite Faulting
    """)
    return


@app.cell
def _(
    BINARY_SEGMENTATION,
    DIP_RANGE,
    FAULT_RADIUS_RANGE,
    FAULT_THICKNESS,
    STRIKE_RANGE,
    THROW_RANGE,
    apply_displacement,
    ndimage,
    np,
):
    def add_finite_fault(img, lbl, rgt, dims, is_3d):
        """Adds a single finite fault."""

        nz = dims[0]
        c = [np.random.randint(20, d-20) for d in dims]
        dip_deg = np.random.uniform(*DIP_RANGE)
        dip = np.radians(dip_deg)

        # Strike Selection Logic
        strike_deg = 0
        if is_3d:
            # Check if STRIKE_RANGE is a list of tuples (sets) or a single tuple
            if isinstance(STRIKE_RANGE, list) and len(STRIKE_RANGE) > 0 and isinstance(STRIKE_RANGE[0], (tuple, list)):
                # Pick one set randomly
                chosen_set = STRIKE_RANGE[np.random.randint(len(STRIKE_RANGE))]
                strike_deg = np.random.uniform(chosen_set[0], chosen_set[1])
            else:
                # Single range
                strike_deg = np.random.uniform(STRIKE_RANGE[0], STRIKE_RANGE[1])

            strike = np.radians(strike_deg)

        # Mechanics
        max_throw = np.random.uniform(THROW_RANGE[0], THROW_RANGE[1])
        # Determine mechanics: 1 (Normal) if throw > 0, else 2 (Reverse)
        mech_type = 1 if max_throw > 0 else 2

        if is_3d:
            cz, cy, cx = c
            n_x = -np.sin(strike) * np.sin(dip)
            n_y = np.cos(strike) * np.sin(dip)
            n_z = -np.cos(dip)
            z, y, x = np.indices(dims)
            dist_plane = n_x*(x-cx) + n_y*(y-cy) + n_z*(z-cz)
            dist_center = np.sqrt((x-cx)**2 + (y-cy)**2 + (z-cz)**2)
        else:
            cz, cx = c
            n_x = np.sin(dip)
            n_z = -np.cos(dip)
            z, x = np.indices(dims)
            dist_plane = n_x*(x-cx) + n_z*(z-cz)
            dist_center = np.sqrt((x-cx)**2 + (z-cz)**2)

        # Decay
        radius = np.random.uniform(*FAULT_RADIUS_RANGE)
        if radius == 0: radius = 1e-5

        decay = np.exp(-dist_center**2 / (2 * (radius/2)**2))
        throw_map = np.abs(max_throw) * decay

        shift_field = np.zeros(dims)
        hw_mask = dist_plane > 0

        # Apply throw direction
        if mech_type == 1: # Normal
            shift_field[hw_mask] = throw_map[hw_mask]
        else: # Reverse
            shift_field[hw_mask] = -throw_map[hw_mask]

        # Apply Shift
        img = apply_displacement(img, shift_field, axis=0)
        rgt = apply_displacement(rgt, shift_field, axis=0)

        # Update Label
        edge = np.zeros(dims)
        for ax in range(len(dims)):
            edge += np.abs(ndimage.sobel(hw_mask.astype(float), axis=ax))

        # Basic valid zone
        valid_zone = (edge > 0.5) & (decay > 0.1)

        # Apply Dilation (Thickness)
        if FAULT_THICKNESS > 0:
            struct = ndimage.generate_binary_structure(len(dims), 2)
            valid_zone = ndimage.binary_dilation(valid_zone, structure=struct, iterations=FAULT_THICKNESS)

        # Assign Label based on mode
        if BINARY_SEGMENTATION:
            lbl[valid_zone] = 1 # Just "Fault"
        else:
            lbl[valid_zone] = mech_type # 1 (Normal) or 2 (Reverse)

        # Collect Stats for this fault
        fault_stats = {
            "dip": dip_deg,
            "strike": strike_deg if is_3d else 0,
            "max_throw": max_throw,
            "radius": radius,
            "type": "Normal" if mech_type == 1 else "Inverse",
            "center": [int(x) for x in c]
        }

        return img, lbl, rgt, fault_stats
    return (add_finite_fault,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Seismic Generation
    """)
    return


@app.cell
def _(FREQ_RANGE, NOISE_LEVEL_RANGE, ndimage, np):
    def ricker_wavelet(f0, dt=0.004, length=0.1):
        if f0 == 0: return np.ones(int(length/dt))
        t = np.arange(-length/2, length/2, dt)
        y = (1.0 - 2.0*(np.pi**2)*(f0**2)*(t**2)) * np.exp(-(np.pi**2)*(f0**2)*(t**2))
        return y

    def finalize_seismic(reflectivity, is_3d):
        """Convolves, blurs, adds colored noise, and NORMALIZES to [-1, 1]."""
        # 1. Convolution
        freq = np.random.uniform(*FREQ_RANGE)
        wavelet = ricker_wavelet(freq)
        seismic = np.apply_along_axis(lambda m: np.convolve(m, wavelet, mode='same'), 0, reflectivity)

        # 2. PSF Blurring
        sigma_psf = [1.0] * len(reflectivity.shape)
        seismic = ndimage.gaussian_filter(seismic, sigma=sigma_psf)

        # 3. Colored Noise
        noise_level = np.random.uniform(*NOISE_LEVEL_RANGE)
        if noise_level > 0:
            raw_noise = np.random.randn(*seismic.shape)
            smooth_noise = ndimage.gaussian_filter(raw_noise, sigma=1.0)
            scaled_noise = noise_level * np.max(np.abs(seismic)) * smooth_noise
            seismic = seismic + scaled_noise

        # 4. Normalize to [-1, 1]
        max_val = np.max(np.abs(seismic))
        if max_val > 0:
            seismic = seismic / max_val

        return seismic, freq, noise_level
    return (finalize_seismic,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Main Generation Loop & Statistics Collection
    """)
    return


@app.cell
def _(
    DIMS,
    DOME_CENTER,
    IS_3D,
    LAYER_THICKNESS_RANGE,
    MAX_FOLD_RANGE,
    NUMBER_LAYERS_RANGE,
    N_DOMES_RANGE,
    N_FAULTS_RANGE,
    N_TOTAL_PAIRS,
    OUTPUT_DIR,
    RANDOM_DOME_LOCATION,
    RANDOM_LAYER_THICKNESS,
    SAVE_STATISTICS,
    SIGMA_K_RANGE,
    TRAIN_RATIO,
    add_finite_fault,
    add_folding_and_shearing,
    finalize_seismic,
    generate_reflectivity,
    json,
    np,
    os,
):
    # --- MAIN GENERATION LOOP ---
    print(f"Generating {N_TOTAL_PAIRS} pairs. 3D Mode: {IS_3D}")

    train_data = []
    val_data = []

    # Statistics Containers
    stats_train = {'cube_params': [], 'fault_params': []}
    stats_val = {'cube_params': [], 'fault_params': []}

    for i in range(N_TOTAL_PAIRS):
        split = 'train' if i < (N_TOTAL_PAIRS * TRAIN_RATIO) else 'validation'

        # Pick random number of layers for this sample
        n_layers_sample = np.random.randint(NUMBER_LAYERS_RANGE[0], NUMBER_LAYERS_RANGE[1] + 1)

        # 1. Base Stratigraphy
        vol, rgt = generate_reflectivity(
            DIMS,
            n_layers_sample,
            random_thickness=RANDOM_LAYER_THICKNESS,
            thickness_range=LAYER_THICKNESS_RANGE
        )

        # 2. Add Folding (Ranges)
        # Randomly select number of domes for this sample
        n_domes_sample = np.random.randint(N_DOMES_RANGE[0], N_DOMES_RANGE[1] + 1)

        vol, rgt = add_folding_and_shearing(
            vol, rgt, DIMS, IS_3D,
            n_domes=n_domes_sample,
            sigma_k_range=SIGMA_K_RANGE,
            max_fold_range=MAX_FOLD_RANGE,
            random_loc=RANDOM_DOME_LOCATION,
            manual_center=DOME_CENTER
        )

        # 3. Add Faults
        lbl = np.zeros(DIMS)
        n_faults = np.random.randint(N_FAULTS_RANGE[0], N_FAULTS_RANGE[1] + 1) if N_FAULTS_RANGE[1] > 0 else 0

        sample_fault_stats = []
        for _ in range(n_faults):
            vol, lbl, rgt, f_stat = add_finite_fault(vol, lbl, rgt, DIMS, IS_3D)
            sample_fault_stats.append(f_stat)

        reflectivity_vol = vol.copy()
        seis, freq_used, noise_used = finalize_seismic(vol, IS_3D)

        # --- DATA STORAGE ---
        sample_dict = {
            'seismic': seis,
            'label': lbl,
            'rgt': rgt,
            'reflectivity': reflectivity_vol,
            'id': i
        }

        # Save to disk logic would go here...

        if split == 'train':
            train_data.append(sample_dict)
            stats_list = stats_train
        else:
            val_data.append(sample_dict)
            stats_list = stats_val

        # --- COLLECT STATS ---
        cube_stats = {
            "id": i,
            "n_layers": int(n_layers_sample),
            "n_domes": int(n_domes_sample),
            "n_faults": int(n_faults),
            "ricker_freq": float(freq_used),
            "noise_sigma": float(noise_used)
        }
        stats_list['cube_params'].append(cube_stats)
        stats_list['fault_params'].extend(sample_fault_stats)

        if i % 5 == 0: print(f"Generated pair {i}/{N_TOTAL_PAIRS} ({split})")

    # --- SAVE STATISTICS ---
    if SAVE_STATISTICS:
        # Full stats
        full_stats = {
            'cube_params': stats_train['cube_params'] + stats_val['cube_params'],
            'fault_params': stats_train['fault_params'] + stats_val['fault_params']
        }

        stats_dir = os.path.join(OUTPUT_DIR, "statistics")
        with open(os.path.join(stats_dir, "statistics_train.json"), 'w') as f:
            json.dump(stats_train, f, indent=4)
        with open(os.path.join(stats_dir, "statistics_validation.json"), 'w') as f:
            json.dump(stats_val, f, indent=4)
        with open(os.path.join(stats_dir, "statistics_full.json"), 'w') as f:
            json.dump(full_stats, f, indent=4)
        print(f"Statistics saved to {stats_dir}")
    return full_stats, stats_train, stats_val, train_data, val_data


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Visualization and Quality Control
    """)
    return


@app.cell
def _(
    BINARY_SEGMENTATION,
    CROSSHAIR_THICKNESS,
    INLINE_COLOR,
    ListedColormap,
    TIMESLICE_COLOR,
    XLINE_COLOR,
    mpatches,
    np,
    plt,
):
    def visualize_dataset(dataset, split_name, is_3d, sample_idx=None, slice_type='mid', show_axis=False, show_crosshairs=False, show_legend=True):
        if not dataset:
            print(f"No data in {split_name} set to visualize.")
            return None

        if sample_idx is None:
            idx = np.random.randint(0, len(dataset))
        else:
            idx = max(0, min(sample_idx, len(dataset) - 1))

        sample = dataset[idx]
        seismic = sample['seismic']
        label = sample['label']
        rgt = sample['rgt']
        reflectivity = sample['reflectivity']
        sample_id = sample['id']

        if is_3d:
            z, y, x = seismic.shape
            if slice_type == 'mid':
                idx_z, idx_y, idx_x = z // 2, y // 2, x // 2
            else:
                val = int(slice_type)
                idx_z = min(val, z-1)
                idx_y = min(val, y-1)
                idx_x = min(val, x-1)
        else:
            idx_z, idx_y, idx_x = 0, 0, 0

        # Colors
        if BINARY_SEGMENTATION:
            cmap_mask = ListedColormap(['white', 'black'])
            norm_mask = plt.Normalize(vmin=0, vmax=1)
            mask_ticks = [0, 1]
        else:
            cmap_mask = ListedColormap(['white', '#32CD32', 'magenta'])
            norm_mask = plt.Normalize(vmin=0, vmax=2)
            mask_ticks = [0, 1, 2]

        cbar_pad = 0.1

        def plot_row(ax_row, seis_slice, lbl_slice, rgt_slice, refl_slice, title_prefix,
                     v_line_idx=None, h_line_idx=None, v_line_color=None, h_line_color=None,
                     v_line_label=None, h_line_label=None):

            # 1. RGT
            im1 = ax_row[0].imshow(rgt_slice, cmap='jet', aspect='equal')
            ax_row[0].set_title(f"{title_prefix} - RGT")
            plt.colorbar(im1, ax=ax_row[0], orientation='horizontal', fraction=0.046, pad=cbar_pad)

            # 2. Fault Label
            im2 = ax_row[1].imshow(lbl_slice, cmap=cmap_mask, norm=norm_mask, aspect='equal', interpolation='nearest')
            ax_row[1].set_title(f"{title_prefix} - Fault Mask")
            plt.colorbar(im2, ax=ax_row[1], orientation='horizontal', fraction=0.046, pad=cbar_pad, ticks=mask_ticks)

            # 3. Reflectivity
            im3 = ax_row[2].imshow(refl_slice, cmap='Greys', aspect='equal')
            ax_row[2].set_title(f"{title_prefix} - Reflectivity")
            plt.colorbar(im3, ax=ax_row[2], orientation='horizontal', fraction=0.046, pad=cbar_pad)

            # 4. Seismic
            im4 = ax_row[3].imshow(seis_slice, cmap='seismic', aspect='equal', vmin=-1, vmax=1)
            ax_row[3].set_title(f"{title_prefix} - Seismic")
            plt.colorbar(im4, ax=ax_row[3], orientation='horizontal', fraction=0.046, pad=cbar_pad)

            # Overlays
            fault_overlay = np.zeros((*lbl_slice.shape, 4))
            if BINARY_SEGMENTATION:
                mask = lbl_slice == 1
                fault_overlay[mask] = [0.0, 0.0, 0.0, 1.0]
            else:
                mask1 = lbl_slice == 1
                mask2 = lbl_slice == 2
                fault_overlay[mask1] = [0.0, 1.0, 0.0, 1.0] # Lime
                fault_overlay[mask2] = [1.0, 0.0, 1.0, 1.0] # Magenta

            # 5. Seis + Fault
            im5 = ax_row[4].imshow(seis_slice, cmap='seismic', aspect='equal', vmin=-1, vmax=1)
            ax_row[4].imshow(fault_overlay, aspect='equal')
            ax_row[4].set_title(f"{title_prefix} - Seis+Fault")
            plt.colorbar(im5, ax=ax_row[4], orientation='horizontal', fraction=0.046, pad=cbar_pad)

            # 6. Composite
            ax_row[5].imshow(seis_slice, cmap='Greys', aspect='equal', vmin=-1, vmax=1)
            im6 = ax_row[5].imshow(rgt_slice, cmap='jet', aspect='equal', alpha=0.3)
            ax_row[5].imshow(fault_overlay, aspect='equal')
            ax_row[5].set_title(f"{title_prefix} - Composite")
            plt.colorbar(im6, ax=ax_row[5], orientation='horizontal', fraction=0.046, pad=cbar_pad)

            # Crosshairs
            if is_3d and show_crosshairs:
                legend_handles = []
                if show_legend:
                    if v_line_idx is not None and v_line_label:
                        legend_handles.append(mpatches.Patch(color=v_line_color, label=v_line_label))
                    if h_line_idx is not None and h_line_label:
                        legend_handles.append(mpatches.Patch(color=h_line_color, label=h_line_label))

                for ax in ax_row:
                    if v_line_idx is not None and v_line_color:
                        ax.axvline(x=v_line_idx, color=v_line_color, linestyle='--', linewidth=CROSSHAIR_THICKNESS)
                    if h_line_idx is not None and h_line_color:
                        ax.axhline(y=h_line_idx, color=h_line_color, linestyle='--', linewidth=CROSSHAIR_THICKNESS)
                    if legend_handles:
                        ax.legend(handles=legend_handles, loc='upper right', fontsize='xx-small', framealpha=0.6)

            for ax in ax_row:
                if not show_axis: ax.axis('off')

        if is_3d:
            fig, axs = plt.subplots(3, 6, figsize=(24, 12))
            plot_row(axs[0], seismic[:, idx_y, :], label[:, idx_y, :], rgt[:, idx_y, :], reflectivity[:, idx_y, :],
                     f"Inline (Y={idx_y})", v_line_idx=idx_x, v_line_color=XLINE_COLOR, v_line_label="Xline", h_line_idx=idx_z, h_line_color=TIMESLICE_COLOR, h_line_label="Time")
            plot_row(axs[1], seismic[:, :, idx_x], label[:, :, idx_x], rgt[:, :, idx_x], reflectivity[:, :, idx_x],
                     f"Xline (X={idx_x})", v_line_idx=idx_y, v_line_color=INLINE_COLOR, v_line_label="Inline", h_line_idx=idx_z, h_line_color=TIMESLICE_COLOR, h_line_label="Time")
            plot_row(axs[2], seismic[idx_z, :, :], label[idx_z, :, :], rgt[idx_z, :, :], reflectivity[idx_z, :, :],
                     f"Time (Z={idx_z})", v_line_idx=idx_x, v_line_color=XLINE_COLOR, v_line_label="Xline", h_line_idx=idx_y, h_line_color=INLINE_COLOR, h_line_label="Inline")
        else:
            fig, axs = plt.subplots(1, 6, figsize=(24, 5))
            if len(axs.shape) == 1: axs = [axs]
            plot_row(axs[0], seismic, label, rgt, reflectivity, "2D")

        fig.suptitle(f"{split_name} Sample {sample_id} (Binary={BINARY_SEGMENTATION})", fontsize=16)
        plt.tight_layout()
        plt.show()  # Explicitly show the plot in the output area
        return fig
    return (visualize_dataset,)


@app.cell
def _(
    SHOW_AXIS_NUMBERS,
    SHOW_CROSSHAIRS,
    SHOW_CROSSHAIRS_LEGEND,
    VISUALIZE_SAMPLE_INDEX,
    VISUALIZE_SLICE_TYPE,
):
    # --- EXECUTE VISUALIZATION ---
    VIZ_SAMPLE_INDEX = VISUALIZE_SAMPLE_INDEX
    VIZ_SLICE_TYPE = VISUALIZE_SLICE_TYPE
    VIZ_SHOW_AXIS = SHOW_AXIS_NUMBERS
    VIZ_SHOW_CROSSHAIRS = SHOW_CROSSHAIRS
    VIZ_SHOW_LEGEND = SHOW_CROSSHAIRS_LEGEND
    return (
        VIZ_SAMPLE_INDEX,
        VIZ_SHOW_AXIS,
        VIZ_SHOW_CROSSHAIRS,
        VIZ_SHOW_LEGEND,
        VIZ_SLICE_TYPE,
    )


@app.cell
def _():
    return


@app.cell(column=1)
def _(
    IS_3D,
    VIZ_SAMPLE_INDEX,
    VIZ_SHOW_AXIS,
    VIZ_SHOW_CROSSHAIRS,
    VIZ_SHOW_LEGEND,
    VIZ_SLICE_TYPE,
    mo,
    train_data,
    visualize_dataset,
):
    fig_train = visualize_dataset(
        train_data, "Train", IS_3D,
        VIZ_SAMPLE_INDEX, VIZ_SLICE_TYPE, VIZ_SHOW_AXIS,
        VIZ_SHOW_CROSSHAIRS, VIZ_SHOW_LEGEND
    )
    if fig_train:
        mo.output.replace(mo.vstack([mo.md("--- Visualizing Training Sample ---"), fig_train]))
    return


@app.cell
def _(
    IS_3D,
    VIZ_SHOW_AXIS,
    VIZ_SHOW_CROSSHAIRS,
    VIZ_SHOW_LEGEND,
    VIZ_SLICE_TYPE,
    mo,
    val_data,
    visualize_dataset,
):
    fig_val = visualize_dataset(
        val_data, "Validation", IS_3D,
        None, VIZ_SLICE_TYPE, VIZ_SHOW_AXIS,
        VIZ_SHOW_CROSSHAIRS, VIZ_SHOW_LEGEND
    )
    if fig_val:
        mo.output.replace(mo.vstack([mo.md("--- Visualizing Validation Sample ---"), fig_val]))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 7. Statistics Display
    """)
    return


@app.cell
def _(np, plt):
    def plot_statistics(stats, dataset_name):
        """Generates statistical plots for a given dataset dictionary."""
        if not stats or not stats['fault_params']:
            print(f"No fault statistics available for {dataset_name}.")
            return None

        fault_params = stats['fault_params']
        cube_params = stats['cube_params']

        # Extract data
        strikes = [f['strike'] for f in fault_params]
        dips = [f['dip'] for f in fault_params]
        throws = [f['max_throw'] for f in fault_params]
        types = [f['type'] for f in fault_params]
        noise_lvls = [c['noise_sigma'] for c in cube_params]

        # Figure
        fig = plt.figure(figsize=(18, 12))
        fig.suptitle(f"{dataset_name} Dataset Statistics", fontsize=20)
        gs = fig.add_gridspec(2, 3)

        # 1. Fault Type Counts
        ax1 = fig.add_subplot(gs[0, 0])
        normal_count = types.count('Normal')
        inverse_count = types.count('Inverse')
        ax1.bar(['Normal', 'Inverse'], [normal_count, inverse_count], color=['lightgreen', 'salmon'])
        ax1.set_title("Fault Type Distribution")
        ax1.set_ylabel("Count")

        # 2. Strike Rose Diagram
        ax2 = fig.add_subplot(gs[0, 1], projection='polar')
        if len(strikes) > 0:
            strikes_rad = np.radians(strikes)
            ax2.hist(strikes_rad, bins=18, edgecolor='black', alpha=0.7)
            ax2.set_title("Fault Strike Angles")
            ax2.set_theta_zero_location("N")
            ax2.set_theta_direction(-1)

        # 3. Dip Histogram
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.hist(dips, bins=10, edgecolor='black', alpha=0.7)
        ax3.set_title("Fault Dip Angles")
        ax3.set_xlabel("Dip (deg)")

        # 4. Throw Histogram
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.hist(throws, bins=10, edgecolor='black', alpha=0.7, color='orange')
        ax4.set_title("Fault Displacement (Throw)")
        ax4.set_xlabel("Pixels")

        # 5. Noise Level Histogram
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.hist(noise_lvls, bins=10, edgecolor='black', alpha=0.7, color='purple')
        ax5.set_title("Seismic Noise Levels")
        ax5.set_xlabel("Noise Sigma")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        return fig
    return (plot_statistics,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Show Statistics
    """)
    return


@app.cell
def _(SHOW_STATISTICS_TABLE, full_stats, mo, pd, stats_train, stats_val):
    stats_table_output = None
    if SHOW_STATISTICS_TABLE:
        elements = [mo.md("--- Statistics Summary Tables ---")]
        for name, stats in [("Full", full_stats), ("Train", stats_train), ("Validation", stats_val)]:
            if stats and stats['fault_params']:
                df = pd.DataFrame(stats['fault_params'])
                summary = df.describe()
                elements.append(mo.md(f"### {name} Dataset - Fault Parameters Summary:"))
                elements.append(mo.ui.table(summary))

        # Explicitly output the vstack to the cell result area
        mo.output.replace(mo.vstack(elements))
        stats_table_output = True
    return


@app.cell
def _(SHOW_STATISTICS_PLOT, full_stats, mo, plot_statistics):
    fig_stats_full = None
    if SHOW_STATISTICS_PLOT:
        fig_stats_full = plot_statistics(full_stats, "Full")
        if fig_stats_full:
            # Explicitly output the combination of text and figure
            mo.output.replace(mo.vstack([mo.md("--- Statistics Plots (Full) ---"), fig_stats_full]))
    return


@app.cell
def _(SHOW_STATISTICS_PLOT, mo, plot_statistics, stats_train):
    fig_stats_train = None
    if SHOW_STATISTICS_PLOT:
        fig_stats_train = plot_statistics(stats_train, "Training")
        if fig_stats_train:
            mo.output.replace(mo.vstack([mo.md("--- Statistics Plots (Training) ---"), fig_stats_train]))
    return


@app.cell
def _(SHOW_STATISTICS_PLOT, mo, plot_statistics, stats_val):
    fig_stats_val = None
    if SHOW_STATISTICS_PLOT:
        fig_stats_val = plot_statistics(stats_val, "Validation")
        if fig_stats_val:
            mo.output.replace(mo.vstack([mo.md("--- Statistics Plots (Validation) ---"), fig_stats_val]))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
