import marimo

__generated_with = "0.18.4"
app = marimo.App(
    width="columns",
    layout_file="layouts/seismicFaultSegmentation2D_V1.grid.json",
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
    # 1. Parameter Documentation

    ### 1. Dataset Organization
    * **`N_TOTAL_PAIRS`**: The total number of synthetic seismic samples to generate.
    * **`TRAIN_RATIO`**: The fraction of the total dataset assigned to the training set.
    * **`VAL_RATIO`**: The fraction of the total dataset assigned to the validation set.
    * **`OUTPUT_DIR`**: The main directory for generated files.
    * **`OUTPUT_FORMAT`**: File format (`'npy'` or `'dat'`).

    ### 2. Dimensionality
    * **`IS_3D`**: `True` for 3D volumes `(Z, Y, X)`, `False` for 2D cross-sections.
    * **`n`**: Base grid size.
    * **`DIMS`**: Shape of the data volume.

    ### 3. Geological Layering
    * **`NUMBER_LAYERS_RANGE`**: Range for the number of horizons.
    * **`RANDOM_LAYER_THICKNESS`**: Varying vs equal spacing.
    * **`LAYER_THICKNESS_RANGE`**: Min/Max thickness in pixels.

    ### 4. Structural Deformation
    * **`N_DOMES_RANGE`**: Range of dome structures.
    * **`SIGMA_K_RANGE`**: Width/smoothness of domes.
    * **`MAX_FOLD_RANGE`**: Amplitude of folds (+Anticline, -Syncline).
    * **`RANDOM_DOME_LOCATION`**: Random placement vs fixed center.

    ### 5. Faulting Mechanics
    

[Image of geologic fault types]

    * **`BINARY_SEGMENTATION`**: `True` (0,1) or `False` (0=Bg, 1=Normal, 2=Reverse).
    * **`N_FAULTS_RANGE`**: Number of faults per sample.
    * **`DIP_RANGE`**: Steepness (degrees).
    * **`STRIKE_RANGE`**: Orientation (degrees).
    * **`FAULT_RADIUS_RANGE`**: Extent/Size of the fault.
    * **`THROW_RANGE`**: Displacement (Positive=Normal, Negative=Reverse).
    * **`FAULT_THICKNESS`**: Label dilation thickness.
    * **`FAULT_CENTER_MARGIN_RANGE`**: A tuple `(min, max)` defining the buffer zone (in pixels) from the edge where fault centers cannot be placed. A range allows for varying degrees of "edge avoidance".

    ### 6. Signal & Noise
    
    * **`NOISE_LEVEL_RANGE`**: Noise-to-signal ratio.
    * **`NOISE_SPATIAL_SMOOTHING_RANGE`**: A tuple `(min, max)` for the `sigma` of the Gaussian filter applied to random noise.
    * **`FREQ_RANGE`**: Ricker wavelet frequency (Hz).
    * **`DT`**: Sampling interval in seconds (e.g., 0.004s = 4ms).
    * **`WAVELET_LENGTH_RANGE`**: A tuple `(min, max)` defining the duration of the Ricker wavelet in seconds.

    ### 7. Visualization
    * **`SHOW_AXIS_NUMBERS`**, **`SHOW_CROSSHAIRS`**, etc.: Plotting preferences.

    ### 8. Statistics
    * **`SAVE_STATISTICS`**: Save JSON metadata.
    * **`SHOW_STATISTICS_TABLE`**: Display summary tables.
    * **`SHOW_STATISTICS_PLOT`**: Display histograms and class balance charts.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 1. Configuration Parameters
    """)
    return


@app.cell
def _():
    # --- 1. DATASET ORGANIZATION ---
    # Total samples to generate. 
    # Suggested: >100 for testing, >1000 for small training, >5000 for robust DL.
    N_TOTAL_PAIRS = 10 
    
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.2
    OUTPUT_DIR = "dataset_flat"
    OUTPUT_FORMAT = 'npy'

    # --- 2. DIMENSIONALITY ---
    IS_3D = True
    n = 128
    DIMS = (n, n, n) if IS_3D else (n, n)

    # --- 3. GEOLOGICAL LAYERING ---
    # Number of distinct reflective horizons. 
    # Suggested: (3, 6) for simple models, (10, 20) for complex stratigraphy.
    NUMBER_LAYERS_RANGE = (3, 6)
    
    RANDOM_LAYER_THICKNESS = True
    
    # Thickness of layers in pixels. 
    # Suggested: (10, 30) for standard resolution, smaller for thin beds.
    LAYER_THICKNESS_RANGE = (10, 30)

    # --- 4. STRUCTURAL DEFORMATION ---
    # Number of fold/dome structures per sample.
    # Suggested: (1, 3) prevents over-distortion.
    N_DOMES_RANGE = (1, 3)
    
    # Smoothness of the dome (Gaussian sigma). 
    # Suggested: (15, 25). Higher = wider/gentler folds, Lower = tighter peaks.
    SIGMA_K_RANGE = (15, 25)
    
    # Amplitude of the fold in pixels. 
    # Suggested: (15.0, 25.0). +/- determines Up (Anticline) or Down (Syncline).
    MAX_FOLD_RANGE = (15.0, 25.0)
    
    RANDOM_DOME_LOCATION = True
    DOME_CENTER = (100, 64, 64)

    # --- 5. FAULTING MECHANICS ---
    # False = Multiclass (0=Bg, 1=Normal, 2=Reverse)
    # True = Binary (0=Bg, 1=Fault)
    BINARY_SEGMENTATION = True 

    # Number of faults per volume. 
    # Suggested: (1, 3) for clean separation, higher for complex fracture networks.
    N_FAULTS_RANGE = (1, 2)
    
    # Fault dip angle in degrees (90 is vertical).
    # Suggested: (45, 75) covers most realistic normal/reverse faults.
    DIP_RANGE = (45, 60)

    # 4 Orthogonal Sets: 0(N), 90(E), 180(S), 270(W). 
    # Added +/- 5 deg variance for realism (Better than strict 0,0).
    STRIKE_RANGE = [(-5, 5), (85, 95), (175, 185), (265, 275)]

    # Spatial radius (decay) of the fault plane.
    # Suggested: (50, 200). 200+ usually cuts the whole volume.
    FAULT_RADIUS_RANGE = (50, 200)
    
    # Displacement in pixels. 
    # Suggested: [-20, 20]. Positive = Normal, Negative = Reverse.
    THROW_RANGE = [-20, 20] 
    
    # Pixels to dilate label. 
    # Suggested: 0 for hairline (1px), 1 or 2 for thicker masks (better for training).
    FAULT_THICKNESS = 0 
    
    # Margin in pixels from the edge where fault centers cannot be placed.
    # Suggested: (10, 30). Prevents faults from being generated entirely off-screen.
    FAULT_CENTER_MARGIN_RANGE = (20, 20)

    # --- 6. SIGNAL & NOISE ---
    # Noise-to-Signal ratio. 
    # Suggested: (0.0, 0.2) for clean data, (0.5, 1.0) for heavy noise.
    NOISE_LEVEL_RANGE = (0.01, 1.0)
    
    # Ricker wavelet peak frequency (Hz).
    # Suggested: (20, 40) for standard seismic, higher for high-res.
    FREQ_RANGE = (25, 50)
    
    # Seismic Sampling Parameters
    DT = 0.004 # Sampling interval in seconds (4ms is standard)
    
    # Length of the wavelet in seconds.
    # Suggested: (0.1, 0.2). Controls the time duration of the source signature.
    WAVELET_LENGTH_RANGE = (0.1, 0.1)
    
    # Noise Spatial Smoothing (Sigma).
    # Suggested: (0.5, 1.5). 
    # 0.0 = White Noise (uncorrelated). 1.0+ = Spatially correlated "blobby" noise.
    NOISE_SPATIAL_SMOOTHING_RANGE = (0.9, 0.9)

    # --- 7. VISUALIZATION SETTINGS ---
    SHOW_AXIS_NUMBERS = True
    VISUALIZE_SAMPLE_INDEX = 0
    VISUALIZE_SLICE_TYPE = 'mid'

    SHOW_CROSSHAIRS = True
    SHOW_CROSSHAIRS_LEGEND = True

    CROSSHAIR_THICKNESS = 1.5
    INLINE_COLOR = 'red'
    XLINE_COLOR = 'blue'
    TIMESLICE_COLOR = 'yellow'
    
    SAVE_FIGURES = True
    FIGURES_DIR = "figures"

    # --- 8. STATISTICS SETTINGS ---
    SAVE_STATISTICS = True
    SHOW_STATISTICS_TABLE = True
    SHOW_STATISTICS_PLOT = True
    return (
        BINARY_SEGMENTATION,
        CROSSHAIR_THICKNESS,
        DIMS,
        DIP_RANGE,
        DOME_CENTER,
        DT,
        FAULT_CENTER_MARGIN_RANGE,
        FAULT_RADIUS_RANGE,
        FAULT_THICKNESS,
        FREQ_RANGE,
        INLINE_COLOR,
        IS_3D,
        LAYER_THICKNESS_RANGE,
        MAX_FOLD_RANGE,
        NOISE_LEVEL_RANGE,
        NOISE_SPATIAL_SMOOTHING_RANGE,
        NUMBER_LAYERS_RANGE,
        N_DOMES_RANGE,
        N_FAULTS_RANGE,
        N_TOTAL_PAIRS,
        OUTPUT_DIR,
        RANDOM_DOME_LOCATION,
        RANDOM_LAYER_THICKNESS,
        SAVE_STATISTICS,
        SHOW_AXIS_NUMBERS,
        SHOW_CROSSHAIRS,
        SHOW_CROSSHAIRS_LEGEND,
        SHOW_STATISTICS_PLOT,
        SHOW_STATISTICS_TABLE,
        SIGMA_K_RANGE,
        STRIKE_RANGE,
        THROW_RANGE,
        TIMESLICE_COLOR,
        TRAIN_RATIO,
        VISUALIZE_SAMPLE_INDEX,
        VISUALIZE_SLICE_TYPE,
        WAVELET_LENGTH_RANGE,
        XLINE_COLOR,
        SAVE_FIGURES,
        FIGURES_DIR,
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

        # Stats and Figures directories
        os.makedirs(os.path.join(base_dir, "statistics"), exist_ok=True)
        os.makedirs("figures", exist_ok=True)

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
        """Adds discrete dome structures."""
        if n_domes == 0:
            return img, rgt

        total_disp = np.zeros(dims)
        coords = np.indices(dims)

        for _ in range(n_domes):
            sigma_k = np.random.uniform(sigma_k_range[0], sigma_k_range[1])
            max_fold = np.random.uniform(max_fold_range[0], max_fold_range[1])

            if max_fold == 0: continue

            if random_loc:
                c = [np.random.randint(0, d) for d in dims]
            else:
                if manual_center is not None and len(manual_center) == len(dims):
                    c = manual_center
                else:
                    c = [d // 2 for d in dims]

            dist_sq = np.zeros(dims)
            for i, coord_grid in enumerate(coords):
                dist_sq += (coord_grid - c[i])**2

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
    def add_finite_fault(img, lbl, rgt, dims, is_3d, margin):
        """Adds a single finite fault."""

        nz = dims[0]
        # Use the passed specific margin
        c = [np.random.randint(margin, d-margin) for d in dims]
        
        dip_deg = np.random.uniform(*DIP_RANGE)
        dip = np.radians(dip_deg)

        # Strike Selection Logic
        strike_deg = 0
        if is_3d:
            if isinstance(STRIKE_RANGE, list) and len(STRIKE_RANGE) > 0 and isinstance(STRIKE_RANGE[0], (tuple, list)):
                chosen_set = STRIKE_RANGE[np.random.randint(len(STRIKE_RANGE))]
                strike_deg = np.random.uniform(chosen_set[0], chosen_set[1])
            else:
                strike_deg = np.random.uniform(STRIKE_RANGE[0], STRIKE_RANGE[1])
            strike = np.radians(strike_deg)

        # Mechanics
        max_throw = np.random.uniform(THROW_RANGE[0], THROW_RANGE[1])
        mech_type = 1 if max_throw > 0 else 2  # 1=Normal, 2=Inverse

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

        valid_zone = (edge > 0.5) & (decay > 0.1)

        if FAULT_THICKNESS > 0:
            struct = ndimage.generate_binary_structure(len(dims), 2)
            valid_zone = ndimage.binary_dilation(valid_zone, structure=struct, iterations=FAULT_THICKNESS)

        if BINARY_SEGMENTATION:
            lbl[valid_zone] = 1
        else:
            lbl[valid_zone] = mech_type

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
    def ricker_wavelet(f0, dt, length):
        if f0 == 0: return np.ones(int(length/dt))
        t = np.arange(-length/2, length/2, dt)
        y = (1.0 - 2.0*(np.pi**2)*(f0**2)*(t**2)) * np.exp(-(np.pi**2)*(f0**2)*(t**2))
        return y

    def finalize_seismic(reflectivity, is_3d, dt, wavelet_length, noise_smooth_sigma):
        freq = np.random.uniform(*FREQ_RANGE)
        wavelet = ricker_wavelet(freq, dt, wavelet_length)
        seismic = np.apply_along_axis(lambda m: np.convolve(m, wavelet, mode='same'), 0, reflectivity)

        sigma_psf = [1.0] * len(reflectivity.shape)
        seismic = ndimage.gaussian_filter(seismic, sigma=sigma_psf)

        noise_level = np.random.uniform(*NOISE_LEVEL_RANGE)
        if noise_level > 0:
            raw_noise = np.random.randn(*seismic.shape)
            # Use the passed parameter sigma
            smooth_noise = ndimage.gaussian_filter(raw_noise, sigma=noise_smooth_sigma)
            scaled_noise = noise_level * np.max(np.abs(seismic)) * smooth_noise
            seismic = seismic + scaled_noise

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
    DT,
    FAULT_CENTER_MARGIN_RANGE,
    IS_3D,
    LAYER_THICKNESS_RANGE,
    MAX_FOLD_RANGE,
    NOISE_SPATIAL_SMOOTHING_RANGE,
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
    WAVELET_LENGTH_RANGE,
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

    stats_train = {'cube_params': [], 'fault_params': []}
    stats_val = {'cube_params': [], 'fault_params': []}

    for i in range(N_TOTAL_PAIRS):
        split = 'train' if i < (N_TOTAL_PAIRS * TRAIN_RATIO) else 'validation'

        n_layers_sample = np.random.randint(NUMBER_LAYERS_RANGE[0], NUMBER_LAYERS_RANGE[1] + 1)

        # 1. Base Stratigraphy
        vol, rgt = generate_reflectivity(
            DIMS,
            n_layers_sample,
            random_thickness=RANDOM_LAYER_THICKNESS,
            thickness_range=LAYER_THICKNESS_RANGE
        )

        # 2. Add Folding
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
            # Sample random margin from range per fault
            margin_sample = np.random.randint(FAULT_CENTER_MARGIN_RANGE[0], FAULT_CENTER_MARGIN_RANGE[1] + 1)
            vol, lbl, rgt, f_stat = add_finite_fault(vol, lbl, rgt, DIMS, IS_3D, margin=margin_sample)
            sample_fault_stats.append(f_stat)

        # 4. Final Seismic
        reflectivity_vol = vol.copy()
        
        # Sample parameters from ranges
        wavelet_len_sample = np.random.uniform(WAVELET_LENGTH_RANGE[0], WAVELET_LENGTH_RANGE[1])
        noise_smooth_sample = np.random.uniform(NOISE_SPATIAL_SMOOTHING_RANGE[0], NOISE_SPATIAL_SMOOTHING_RANGE[1])
        
        seis, freq_used, noise_used = finalize_seismic(
            vol, 
            IS_3D, 
            dt=DT, 
            wavelet_length=wavelet_len_sample, 
            noise_smooth_sigma=noise_smooth_sample
        )

        # --- CALCULATE PIXEL STATS FOR PLOTTING ---
        # Get counts for background(0), normal/fault(1), reverse(2)
        unique_vals, unique_counts = np.unique(lbl, return_counts=True)
        count_map = dict(zip(unique_vals, unique_counts))

        # Standardize keys to 0, 1, 2
        pixel_stats = {
            "0": int(count_map.get(0, 0)),
            "1": int(count_map.get(1, 0)),
            "2": int(count_map.get(2, 0))
        }

        # --- DATA STORAGE ---
        sample_dict = {
            'seismic': seis,
            'label': lbl,
            'rgt': rgt,
            'reflectivity': reflectivity_vol,
            'id': i,
            'fault_stats': sample_fault_stats, # Saved here for visualization retrieval
            'pixel_stats': pixel_stats
        }

        if split == 'train':
            train_data.append(sample_dict)
            stats_list = stats_train
        else:
            val_data.append(sample_dict)
            stats_list = stats_val

        # --- COLLECT METADATA ---
        cube_stats = {
            "id": i,
            "n_layers": int(n_layers_sample),
            "n_domes": int(n_domes_sample),
            "n_faults": int(n_faults),
            "ricker_freq": float(freq_used),
            "noise_sigma": float(noise_used),
            "pixel_counts": pixel_stats # Saved here for aggregate stats plotting
        }
        stats_list['cube_params'].append(cube_stats)
        stats_list['fault_params'].extend(sample_fault_stats)

        if i % 5 == 0: print(f"Generated pair {i}/{N_TOTAL_PAIRS} ({split})")

    # --- SAVE STATISTICS ---
    full_stats = {
        'cube_params': stats_train['cube_params'] + stats_val['cube_params'],
        'fault_params': stats_train['fault_params'] + stats_val['fault_params']
    }

    if SAVE_STATISTICS:
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

        # Extract Fault Info for Subtitle
        fault_infos = sample.get('fault_stats', [])
        if not fault_infos:
            subtitle = "No Faults"
        else:
            n_f = len(fault_infos)
            # Gather throws
            throws = [round(f['max_throw'], 1) for f in fault_infos]
            dips = [int(f['dip']) for f in fault_infos]
            subtitle = f"Num Faults: {n_f} | Throws: {throws} px | Dips: {dips}°"

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
            cmap_mask = ListedColormap(['white', '#32CD32', 'magenta']) # White, Lime, Magenta
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

        # Main Title moved UP (y=0.98), Subtitle below it (y=0.95)
        fig.suptitle(f"{split_name} Sample {sample_id} (Binary={BINARY_SEGMENTATION})", fontsize=16, y=0.98)
        fig.text(0.5, 0.95, subtitle, ha='center', fontsize=12, color='darkblue')

        plt.tight_layout(rect=[0, 0, 1, 0.94]) # Adjust layout to make room for titles
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
    FIGURES_DIR,
    IS_3D,
    SAVE_FIGURES,
    VIZ_SAMPLE_INDEX,
    VIZ_SHOW_AXIS,
    VIZ_SHOW_CROSSHAIRS,
    VIZ_SHOW_LEGEND,
    VIZ_SLICE_TYPE,
    mo,
    os,
    train_data,
    visualize_dataset,
):
    fig_train = visualize_dataset(
        train_data, "Train", IS_3D,
        VIZ_SAMPLE_INDEX, VIZ_SLICE_TYPE, VIZ_SHOW_AXIS,
        VIZ_SHOW_CROSSHAIRS, VIZ_SHOW_LEGEND
    )
    if fig_train:
        if SAVE_FIGURES:
            fig_idx = VIZ_SAMPLE_INDEX if VIZ_SAMPLE_INDEX is not None else 0
            fname = os.path.join(FIGURES_DIR, f"sample_train_{fig_idx}.png")
            fig_train.savefig(fname, dpi=300, bbox_inches='tight')
        mo.output.replace(mo.vstack([mo.md("--- Visualizing Training Sample ---"), fig_train]))
    return


@app.cell
def _(
    FIGURES_DIR,
    IS_3D,
    SAVE_FIGURES,
    VIZ_SHOW_AXIS,
    VIZ_SHOW_CROSSHAIRS,
    VIZ_SHOW_LEGEND,
    VIZ_SLICE_TYPE,
    mo,
    os,
    val_data,
    visualize_dataset,
):
    fig_val = visualize_dataset(
        val_data, "Validation", IS_3D,
        None, VIZ_SLICE_TYPE, VIZ_SHOW_AXIS,
        VIZ_SHOW_CROSSHAIRS, VIZ_SHOW_LEGEND
    )
    if fig_val:
        if SAVE_FIGURES:
            fname = os.path.join(FIGURES_DIR, "sample_validation_random.png")
            fig_val.savefig(fname, dpi=300, bbox_inches='tight')
        mo.output.replace(mo.vstack([mo.md("--- Visualizing Validation Sample ---"), fig_val]))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 7. Statistics Display
    """)
    return


@app.cell
def _(BINARY_SEGMENTATION, np, plt):
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

        # 6. Class Balance (Pixels)
        ax6 = fig.add_subplot(gs[1, 2])

        # Aggregate pixel counts across all samples in this set
        total_0 = sum([c['pixel_counts'].get('0', 0) for c in cube_params])
        total_1 = sum([c['pixel_counts'].get('1', 0) for c in cube_params])
        total_2 = sum([c['pixel_counts'].get('2', 0) for c in cube_params])

        total_pixels = total_0 + total_1 + total_2
        if total_pixels == 0: total_pixels = 1

        p0 = (total_0 / total_pixels) * 100
        p1 = (total_1 / total_pixels) * 100
        p2 = (total_2 / total_pixels) * 100

        if BINARY_SEGMENTATION:
            # Simple Bar: Background vs Fault
            bars = ax6.bar(['Non-Fault', 'Fault'], [p0, p1], color=['lightgray', 'black'], edgecolor='black')
            ax6.set_title("Class Balance (Binary)")
            ax6.set_ylabel("Percentage (%)")

            # Add text labels on top
            for bar in bars:
                yval = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.2f}%", ha='center', va='bottom', fontsize=10)

            # Adjust y limits for text
            ax6.set_ylim(0, max(p0, p1) * 1.15)

        else:
            # Stacked Bar: Grouping Non-Fault vs Faults (Normal+Inverse)
            # Bar 1: Non-Fault (Gray)
            # Bar 2: Fault (Stacked Lime + Magenta)

            # Bar 1
            bar_nf = ax6.bar(['Non-Fault'], [p0], color='lightgray', edgecolor='black', label='Background')

            # Bar 2 (Stacked)
            bar_f1 = ax6.bar(['Faults'], [p1], color='#32CD32', edgecolor='black', label='Normal')
            bar_f2 = ax6.bar(['Faults'], [p2], bottom=[p1], color='magenta', edgecolor='black', label='Inverse')

            ax6.set_title("Class Balance (Multiclass)")
            ax6.set_ylabel("Percentage (%)")
            ax6.legend(loc='upper right')

            # Text for Non-Fault
            ax6.text(bar_nf[0].get_x() + bar_nf[0].get_width()/2, p0 + 0.5, f"{p0:.2f}%", ha='center', va='bottom', fontsize=10)

            # Text for Faults (Total)
            total_fault_p = p1 + p2
            ax6.text(bar_f1[0].get_x() + bar_f1[0].get_width()/2, total_fault_p + 0.5, f"{total_fault_p:.2f}%", ha='center', va='bottom', fontsize=10)

            # Optional: Text inside segments if they are large enough? 
            # Usually better to just show total fault % on top as requested ("grouping... the faults")

            ax6.set_ylim(0, max(p0, total_fault_p) * 1.15)

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
def _(
    FIGURES_DIR,
    SAVE_FIGURES,
    SHOW_STATISTICS_PLOT,
    full_stats,
    mo,
    os,
    plot_statistics,
):
    fig_stats_full = None
    if SHOW_STATISTICS_PLOT:
        fig_stats_full = plot_statistics(full_stats, "Full")
        if fig_stats_full:
            if SAVE_FIGURES:
                fname = os.path.join(FIGURES_DIR, "statistics_full.png")
                fig_stats_full.savefig(fname, dpi=300, bbox_inches='tight')
            # Explicitly output the combination of text and figure
            mo.output.replace(mo.vstack([mo.md("--- Statistics Plots (Full) ---"), fig_stats_full]))
    return


@app.cell
def _(
    FIGURES_DIR,
    SAVE_FIGURES,
    SHOW_STATISTICS_PLOT,
    mo,
    os,
    plot_statistics,
    stats_train,
):
    fig_stats_train = None
    if SHOW_STATISTICS_PLOT:
        fig_stats_train = plot_statistics(stats_train, "Training")
        if fig_stats_train:
            if SAVE_FIGURES:
                fname = os.path.join(FIGURES_DIR, "statistics_train.png")
                fig_stats_train.savefig(fname, dpi=300, bbox_inches='tight')
            mo.output.replace(mo.vstack([mo.md("--- Statistics Plots (Training) ---"), fig_stats_train]))
    return


@app.cell
def _(
    FIGURES_DIR,
    SAVE_FIGURES,
    SHOW_STATISTICS_PLOT,
    mo,
    os,
    plot_statistics,
    stats_val,
):
    fig_stats_val = None
    if SHOW_STATISTICS_PLOT:
        fig_stats_val = plot_statistics(stats_val, "Validation")
        if fig_stats_val:
            if SAVE_FIGURES:
                fname = os.path.join(FIGURES_DIR, "statistics_validation.png")
                fig_stats_val.savefig(fname, dpi=300, bbox_inches='tight')
            mo.output.replace(mo.vstack([mo.md("--- Statistics Plots (Validation) ---"), fig_stats_val]))
    return


if __name__ == "__main__":
    app.run()