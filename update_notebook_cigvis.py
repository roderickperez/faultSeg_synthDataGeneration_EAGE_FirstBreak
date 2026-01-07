import json
import os

notebook_path = '/Users/roderickperez/DS_PROJECTS/faultSeg_datasetGeneration/generateDataset_cigvis.ipynb'

with open(notebook_path, 'r') as f:
    nb = json.load(f)

# 1. Add Imports
# Find the first code cell or the import cell
import_cell = None
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if 'import numpy' in source or 'import matplotlib' in source:
            import_cell = cell
            break

if import_cell:
    # Add cigvis import if not present
    if 'import cigvis' not in "".join(import_cell['source']):
        import_cell['source'].insert(0, "import cigvis\n")
        import_cell['source'].insert(1, "from cigvis import colormap\n")

# 2. Update Visualization Code
# Find the cell with the visualization loop
target_cell = None
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source_str = "".join(cell['source'])
        if '# Loop over splits' in source_str:
            target_cell = cell
            break

if target_cell:
    # New source code for the cell using cigvis
    new_source = [
        "# Loop over splits\n",
        "for split in [\"train\", \"validation\"]:\n",
        "    seismic_dir = os.path.join(base_out, split, \"seis\")\n",
        "    mask_dir = os.path.join(base_out, split, \"fault\")\n",
        "\n",
        "    if not (os.path.exists(seismic_dir) and os.path.exists(mask_dir)):\n",
        "        print(f\"[{split}] Skipping: directories not found.\")\n",
        "        continue\n",
        "\n",
        "    files = sorted(set(os.listdir(seismic_dir)).intersection(os.listdir(mask_dir)))\n",
        "    if not files:\n",
        "        print(f\"[{split}] No files to visualize.\")\n",
        "        continue\n",
        "\n",
        "    selected_file = random.choice([f for f in files if f.endswith((\".npy\", \".dat\"))])\n",
        "    print(f\"[{split}] Selected file: {selected_file}\")\n",
        "\n",
        "    # Load data\n",
        "    seismic_path = os.path.join(seismic_dir, selected_file)\n",
        "    mask_path = os.path.join(mask_dir, selected_file)\n",
        "\n",
        "    if selected_file.endswith(\".npy\"):\n",
        "        seismic = np.load(seismic_path)\n",
        "        mask = np.load(mask_path)\n",
        "    else:\n",
        "        seismic = np.fromfile(seismic_path, dtype=np.float32).reshape((cube_size, cube_size, cube_size))\n",
        "        mask = np.fromfile(mask_path, dtype=np.uint8).reshape((cube_size, cube_size, cube_size))\n",
        "\n",
        "    nx, ny, nz = seismic.shape\n",
        "    \n",
        "    # Cigvis Visualization\n",
        "    nodes = []\n",
        "    \n",
        "    # 1. Seismic Slices\n",
        "    # Display slices at center\n",
        "    nodes += cigvis.create_slices(seismic, pos=[nx//2, ny//2, nz//2], cmap='gray')\n",
        "    \n",
        "    # 2. Fault Bodies\n",
        "    if mask_mode == 1: # Multiclass\n",
        "        # Normal Faults (1) -> Green\n",
        "        mask_normal = np.where(mask == 1, 1.0, 0.0).astype(np.float32)\n",
        "        if np.sum(mask_normal) > 0:\n",
        "            nodes += cigvis.create_bodys(mask_normal, 0.5, 0.0, color='green')\n",
        "            \n",
        "        # Inverse Faults (2) -> Purple\n",
        "        mask_inverse = np.where(mask == 2, 1.0, 0.0).astype(np.float32)\n",
        "        if np.sum(mask_inverse) > 0:\n",
        "            nodes += cigvis.create_bodys(mask_inverse, 0.5, 0.0, color='purple')\n",
        "    else: # Binary\n",
        "        # Faults (1) -> Red\n",
        "        mask_bin = np.where(mask > 0, 1.0, 0.0).astype(np.float32)\n",
        "        if np.sum(mask_bin) > 0:\n",
        "            nodes += cigvis.create_bodys(mask_bin, 0.5, 0.0, color='red')\n",
        "            \n",
        "    print(f\"Visualizing {selected_file} with cigvis...\")\n",
        "    cigvis.plot3D(nodes, savequality=5)\n"
    ]
    target_cell['source'] = new_source

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully with cigvis visualization.")
