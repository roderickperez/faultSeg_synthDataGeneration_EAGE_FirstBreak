import json
import os

notebook_path = '/Users/roderickperez/DS_PROJECTS/faultSeg_datasetGeneration/generateDataset.ipynb'

with open(notebook_path, 'r') as f:
    nb = json.load(f)

# 1. Update Import
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        for i, line in enumerate(source):
            if 'from utilities import' in line:
                if 'plot_3d_interactive' not in line:
                    new_line = line.strip() + ', plot_3d_interactive\n'
                    # Remove the newline from the original string before appending if it was there, 
                    # but strip() removes it. 
                    # Let's be more careful.
                    if line.endswith('\n'):
                        new_line = line.replace('\n', ', plot_3d_interactive\n')
                    else:
                        new_line = line + ', plot_3d_interactive'
                    source[i] = new_line
                break
        if 'from utilities import' in "".join(source):
            break

# 2. Update Visualization Code
# Find the cell with the visualization loop
target_cell = None
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source_str = "".join(cell['source'])
        if '# Loop over splits' in source_str and 'go.Figure()' in source_str:
            target_cell = cell
            break

if target_cell:
    source = target_cell['source']
    new_source = []
    skip = False
    for line in source:
        if '# Slices' in line:
            skip = True
            # Append the new visualization call
            new_source.append('    # 3D Interactive Visualization\n')
            new_source.append('    plot_3d_interactive(seismic, mask, title=f"3D Fault Visualization — {split.upper()} / {selected_file}")\n')
            new_source.append('    # fig.show() is handled inside plot_3d_interactive if we returned fig, but here we need to show it.\n')
            new_source.append('    # The function returns fig, so we should show it.\n')
            new_source.append('    fig = plot_3d_interactive(seismic, mask, title=f"3D Fault Visualization — {split.upper()} / {selected_file}")\n')
            new_source.append('    fig.show()\n')
        
        if 'fig.show()' in line and skip:
            skip = False
            continue
            
        if not skip:
            new_source.append(line)
    
    target_cell['source'] = new_source

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully.")
