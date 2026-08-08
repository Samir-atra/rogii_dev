# Automated Geosteering for Wellbore Geology Prediction


This repository contains a complete, high-performance pipeline for the **ROGII Wellbore Geology Prediction** challenge. The project leverages JAX, Keras, and Polars to build, train, and evaluate deep learning models for automated geosteering. The core task is to predict the True Vertical Thickness (TVT) by aligning real-time horizontal well log data against a vertical reference type well.

## Table of Contents

- [Geological Context](#geological-context)
- [Key Features](#key-features)
- [Dataset Structure](#dataset-structure)
- [Methodology](#methodology)
  - [Sequence-to-Sequence Alignment](#sequence-to-sequence-alignment)
  - [Modeling Architectures](#modeling-architectures)
  - [High-Precision Computing with JAX](#high-precision-computing-with-jax)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
  - [Formation Class Distributions](#formation-class-distributions)
  - [Gamma Ray vs. Depth](#gamma-ray-vs-depth)
  - [3D Wellbore Trajectories](#3d-wellbore-trajectories)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [How to Run](#how-to-run)
  - [1. Run Data Analysis Scripts](#1-run-data-analysis-scripts)
  - [2. Prepare Augmented Test Data](#2-prepare-augmented-test-data)
  - [3. Run Unit Tests](#3-run-unit-tests)
- [References](#references)
- [License](#license)

## Geological Context

The project focuses on a continuous Late Cretaceous stratigraphic sequence in South Texas. Accurate geosteering requires understanding the distinct geophysical properties of each formation. The formations are ordered from youngest (top) to oldest (bottom):

| Code  | Formation Name        | Gamma Ray (GR) Behavior                                       | Role in Geosteering                                       |
| :---- | :-------------------- | :------------------------------------------------------------ | :-------------------------------------------------------- |
| `ANCC`  | **Anacacho Limestone**  | Low GR with sharp, high-intensity spikes (volcanic ash).      | Regional structural ceiling, far above the target.        |
| `ASTNU` | **Upper Austin Chalk**  | Rhythmic, "wavy" medium-low GR signature.                     | High-accuracy correlation pattern for navigation.         |
| `ASTNL` | **Lower Austin Chalk**  | Higher baseline GR than Upper Chalk due to more clay.         | Critical marker above the primary target zone.            |
| `EGFDU` | **Upper Eagle Ford**    | Highly variable, "comb-like" GR from alternating layers.      | Diagnostic signature indicating proximity to the target.  |
| `EGFDL` | **Lower Eagle Ford**    | **Very high, distinct GR** (often >120 API) due to organics.  | **Primary hydrocarbon reservoir and geosteering target.** |
| `BUDA`  | **Buda Limestone**      | **Extremely low, clean GR** (<30 API) with a blocky signature. | Critical structural "floor"; drilling into it is an error. |

The main objective is to keep the drill bit within the **Lower Eagle Ford (`EGFDL`)**, the organic-rich "sweet spot".

## Key Features

*   **High-Performance Data I/O**: Utilizes **Polars** for blazingly fast, parallelized ingestion and feature engineering on large CSV files.
*   **Advanced Sequence Modeling**: Implements and tests multiple architectures, including **Conv1D** networks and **Deep Echo State Networks (DeepESN)**, for robust log correlation.
*   **JAX-Powered Backend**: Leverages the **JAX backend for Keras** to enable XLA compilation, custom loss functions, and high-precision `float64` arithmetic, which is critical for handling large coordinate values without precision loss.
*   **Comprehensive EDA**: Includes scripts to generate publication-quality visualizations of geological class distributions, spatial trajectories, and Gamma Ray signatures.
*   **Robust Testing**: A suite of unit tests (`pytest`) validates data processing, model construction, and the numerical stability of the prediction pipeline.
*   **Modular & Readable Code**: The codebase is organized into clear modules for data analysis, preparation, and modeling, with a focus on readability and maintainability.

## Dataset Structure

For each well, the dataset provides three key files:

1.  **`{well_id}__horizontal_well.csv`**: The primary input data containing real-time measurements from Logging While Drilling (LWD) tools.
    *   `MD`: Measured Depth along the wellbore.
    *   `X`, `Y`, `Z`: Spatial coordinates (Easting, Northing, True Vertical Depth).
    *   `GR`: Gamma Ray log, the most critical feature for geological correlation.
    *   `TVT_input`: The partial TVT log provided for the test set.
    *   `ANCC`, `ASTNU`, etc.: The Z-depth of each formation top (structural markers).

2.  **`{well_id}__typewell.csv`**: A vertical reference well that provides the stratigraphic "DNA" for the area.
    *   `TVT`: The vertical depth scale.
    *   `GR`: The reference Gamma Ray signature for each formation.
    *   `Geology`: The ground-truth formation name at each depth.

3.  **`{well_id}.png`**: A cross-section plot visualizing the well trajectory against the geological model, used for manual verification.

## Methodology

### Sequence-to-Sequence Alignment

Geosteering is framed as a **sequence-to-sequence spatial alignment task**. The goal is to map the sequential measurements from the horizontal well (indexed by Measured Depth) onto the vertical stratigraphic sequence defined by the Type Well (indexed by True Vertical Thickness).


### Modeling Architectures

The project explores sequence models capable of learning the complex relationships between wellbore trajectory, GR signatures, and geological formations.

*   **Convolutional Neural Networks (Conv1D)**: Use causal convolutions to act as a fast and effective feature extractor on sliding windows of log data.
*   **Echo State Networks (DeepESN)**: A type of Reservoir Computing that uses a fixed, recurrent reservoir of neurons to project the input sequence into a high-dimensional space. A simple linear readout (trained with Ridge Regression) can then effectively solve the prediction task. This approach is extremely fast to train.

### High-Precision Computing with JAX

Standard `float32` precision is insufficient for geosteering calculations, as the spatial coordinates (`X`, `Y`, `Z`) are large-magnitude numbers. Subtracting two large numbers can lead to catastrophic cancellation and loss of precision.

This project solves this by **enabling `float64` precision in JAX**. All normalization, denormalization, and geometric calculations are performed in double-precision to ensure the model's predictions are numerically stable and accurate.

## Exploratory Data Analysis (EDA)

The `analytics/` directory contains detailed visualizations generated by the scripts in `src/`.

### Formation Class Distributions

The analysis reveals a significant class imbalance. The left chart shows the total thickness of each formation in the reference wells, while the right chart shows how much of the drilled horizontal path was spent in each formation. As expected, the **Lower Eagle Ford (`EGFDL`)** is the most drilled horizontal target.


### Gamma Ray vs. Depth

This scatter plot shows the characteristic Gamma Ray signatures at different vertical depths. The dense, high-GR cluster corresponds to the target **EGFDL** formation, while the low-GR cluster represents the **Buda** and **Anacacho** limestones.


### 3D Wellbore Trajectories

This plot visualizes the spatial paths of all wellbores in the training set, colored by depth. It provides an intuitive understanding of the drilling geometry and reservoir structure across the field.


## Project Structure

```
rogii-geosteering/
├── analytics/
│   ├── class_distributions.png
│   ├── gr_scatter.png
│   ├── missing_values_report.csv
│   └── spatial_3d_scatter.png
├── dataset/
│   ├── train/
│   └── test/
├── docs/
│   ├── drilling_and_geosteering_guide.md
│   └── inference_nb_edits.md
├── src/
│   ├── analyze_missing_values.py   # Scans dataset for NaN values
│   ├── create_augmented_test_set.py # Prepares test data for training
│   ├── deep_esn.py                 # Deep Echo State Network implementation
│   ├── plot_class_counts.py        # Generates class distribution chart
│   ├── spatial_gr_plots.py         # Generates GR and 3D spatial plots
│   └── verify_gr_inclusion.py      # Checks GR value overlap between wells
├── tests/
│   ├── test_conv1d_training.py     # Unit tests for Conv1D model pipeline
│   ├── test_deep_esn.py            # Unit tests for DeepESN model
│   └── test_predict_well.py        # Unit tests for high-precision prediction
├── README.md
└── requirements.txt
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd rogii-geosteering
    ```

2.  **Create a Python virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    The project relies on JAX, which requires specific installation commands depending on your hardware (CPU/GPU).

    *   **For CPU:**
        ```bash
        pip install -r requirements.txt -f https://storage.googleapis.com/jax-releases/jax_releases.html
        ```

    *   **For NVIDIA GPU (CUDA):**
        Find your CUDA version and install the correct JAX release from the [official guide](https://github.com/google/jax#installation). For example, with CUDA 12:
        ```bash
        pip install -r requirements.txt -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
        ```

## How to Run

### 1. Run Data Analysis Scripts

Execute the scripts in the `src/` directory to reproduce the EDA results. The output will be saved to the `analytics/` folder.

```bash
# Generate class distribution bar chart
python src/plot_class_counts.py

# Generate spatial and GR scatter plots
python src/spatial_gr_plots.py

# Analyze missing values across the entire dataset
python src/analyze_missing_values.py
```

### 2. Prepare Augmented Test Data

To use the labeled portions of the test set for validation or augmented training, run the following script. It copies the `TVT_input` column to a new `TVT` column, making it compatible with the training pipeline.

```bash
python src/create_augmented_test_set.py
```

This will create a new directory: `dataset/test_with_tvt/`.

### 3. Run Unit Tests

To verify the correctness of the data processing and modeling components, run the `pytest` suite from the root directory.

```bash
pytest
```

## References

This project builds upon principles from academic research and industry best practices. Key papers are located in the `literature/` directory (not included here, but referenced in `docs/drilling_and_geosteering_guide.md`).

1.  **Alyaev, S., & Elsheikh, A. H. (2021).** *Direct multi-modal inversion of geophysical logs using deep learning.*
2.  **Muhammad, R. B., et al. (2024).** *High-Precision Geosteering via Reinforcement Learning and Particle Filters.*
3.  **Rammay, M. H., et al. (2022).** *Strategic Geosteering Workflow with Uncertainty Quantification and Deep Learning.*
4.  **Sakoe, H., & Chiba, S. (1978).** *Dynamic programming algorithm optimization for spoken word recognition.* (Foundation for DTW).

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

*This README was generated based on the project's source code and documentation.*


