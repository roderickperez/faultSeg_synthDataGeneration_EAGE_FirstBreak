# Seismic Fault Segmentation: Synthetic Data Generation

This project provides a robust framework for generating synthetic 2D and 3D seismic data tailored for training Deep Learning models for **Fault Segmentation**. The data includes complex geological features such as layered stratigraphy, structural folding (domes/anticlines), and realistic finite faults with variable dip, strike, and displacement.

## Key Features

- **Multi-Dimensional Support**: Generate both 2D cross-sections and 3D volumes.
- **Geological Complexity**:
    - **Layering**: Variable horizon thickness and reflectivity.
    - **Folding**: Synthetic domes and anticlines using Gaussian-weighted displacements.
    - **Finite Faults**: Realistic fault planes with spatial decay, supporting both **Normal** and **Reverse** mechanics.
- **Seismic Simulation**: Signal generation using Ricker wavelets with configurable peak frequencies and spatially correlated noise.
- **Dataset Management**:
    - Automatic creation of training and validation splits.
    - Export to `.npy` or `.dat` formats.
    - Detailed metadata/statistics collection (JSON format).
- **Interactive Visualization**: Powered by [Marimo](https://marimo.io/), allowing for real-time parameter tuning and quality control (QC).

## Installation

This project uses `uv` for fast dependency management.

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd faultSeg_synthDataGeneration_EAGE_FirstBreak
   ```

2. **Install dependencies**:
   ```bash
   uv sync
   ```

## Usage

The main generation logic is contained within a Marimo notebook for an interactive experience.

### Run with Marimo
To open the interactive editor and run the generation loop:
```bash
uv run marimo edit seismicFaultSegmentation2D_V1.py
```

## Data Generation Pipeline

1. **Reflectivity Generation**: Creates a base layered model ($RGT$) with random reflectivity coefficients.
2. **Structural Deformation**: Applies "doming" or folding effects to the RGT and reflectivity volumes.
3. **Faulting**: Injects finite faults. Each fault is defined by a center, dip, strike (for 3D), and a "throw" (displacement). A spatial decay function ensures faults are localized.
4. **Seismic Synthesis**: Convolves the reflectivity with a Ricker wavelet and adds Gaussian noise with specified spatial smoothing (sigma).
5. **Mask Generation**: Creates precise ground-truth labels for fault locations, supporting both Binary (Fault/No-Fault) and Multi-class (Normal/Reverse/Background) segmentation.

## Project Structure

- `seismicFaultSegmentation2D_V1.py`: The core Marimo notebook for generation and visualization.
- `layouts/`: JSON files defining the Marimo UI arrangement.
- `dataset_flat/`: Default output directory for generated seismic pairs and statistics.
- `pyproject.toml`: Project metadata and dependencies.

## Statistics & QC

The notebook provides built-in tools to visualize:
- **Fault Parameter Distribution**: Histograms of dip, strike, and throw.
- **Class Balance**: Analysis of pixel/voxel distribution between background and fault classes.
- **Sample QC**: Inline, Xline, and Time-slice visualizations with fault mask overlays.

---
*Created for EAGE/First Break research and development.*
