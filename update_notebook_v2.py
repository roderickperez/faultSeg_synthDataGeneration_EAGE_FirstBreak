import json
import os

notebook_path = '/Users/roderickperez/DS_PROJECTS/faultSeg_datasetGeneration/generateDataset.ipynb'

with open(notebook_path, 'r') as f:
    nb = json.load(f)

# Update Visualization Code: Replace the entire cell content
target_cell = None
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source_str = "".join(cell['source'])
        if '# Loop over splits' in source_str and 'go.Figure()' in source_str:
            target_cell = cell
            break

if target_cell:
    # New source code for the cell
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
        "    # 3D Interactive Visualization (Planes)\n",
        "    # This replaces the old slice-based visualization\n",
        "    fig = plot_3d_interactive(seismic, mask, title=f\"Interactive 3D Fault Planes \u2014 {split.upper()} / {selected_file}\")\n",
        "    fig.show()\n"
    ]
    target_cell['source'] = new_source

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully: Replaced old visualization.")
