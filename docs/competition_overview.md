# ROGII Wellbore Geology Prediction - Competition Overview

## 1. Project Goal
The objective is to automate geological interpretation by predicting **True Vertical Thickness (TVT)** along a horizontal wellbore. This is a "geosteering" task where horizontal log data (Gamma Ray, coordinates) must be mapped back to a vertical reference (Type Well) to determine the exact stratigraphic position of the drill bit.

## 2. Data Description

### Horizontal Wells (`horizontal_well.csv`)
These files represent the path of the horizontal well and the logs collected during drilling.
- **MD**: Measured Depth (length of the wellbore).
- **X, Y, Z**: Spatial coordinates.
- **GR**: Gamma Ray log (measures radioactivity of formations). Often contains `NaN` values.
- **TVT / TVT_input**: True Vertical Thickness. This is the target variable.
  - In **Training**: `TVT` is provided.
  - In **Test**: `TVT_input` is provided but contains `NaN` values in evaluation zones.
- **Markers (Training Only)**: `ANCC`, `ASTNU`, `ASTNL`, `EGFDU`, `EGFDL`, `BUDA`. These are geological boundaries that provide context for the target formation.

### Type Wells (`typewell.csv`)
These are vertical reference wells used as a geological "gold standard".
- **TVT**: True Vertical Thickness (vertical depth scale).
- **GR**: Gamma Ray log at that vertical depth.
- **Geology**: Formation labels (often empty).

## 3. Key Considerations & Challenges

### A. Data Alignment (Geosteering)
The core challenge is matching the Gamma Ray signature from the horizontal well to the signature in the vertical Type Well. 
- The horizontal well might move "up" or "down" relative to the stratigraphic layers.
- The same layer will appear at different `MD` in the horizontal well depending on the dip angle.

### B. Missing Data Handling
- The `GR` logs contain frequent gaps (`NaN`).
- Robust interpolation (linear, spline, or model-based) is required.

#### Dataset Missing Value Statistics
The following table summarizes the distribution of missing values across the entire dataset (Train and Test combined), as calculated by `src/analyze_missing_values.py`:

| Feature   |   Null_Count |   Percentage (%) |
|:----------|-------------:|-----------------:|
| TVT_input |      3798140 |            56.7  |
| GR        |      1511756 |            22.57 |
| Geology   |       523474 |             7.81 |
| ANCC      |        45634 |             0.68 |
| EGFDL     |         6067 |             0.09 |
| id        |            0 |             0    |
| tvt       |            0 |             0    |
| TVT       |            0 |             0    |
| MD        |            0 |             0    |
| X         |            0 |             0    |
| Y         |            0 |             0    |
| Z         |            0 |             0    |
| ASTNU     |            0 |             0    |
| ASTNL     |            0 |             0    |
| EGFDU     |            0 |             0    |
| BUDA      |            0 |             0    |

### C. Feature Engineering
- **Spatial Gradients**: Changes in `Z` relative to `X, Y` indicate the well's inclination.
- **Log Windows**: Using rolling statistics or sequence-based models to capture geological "shapes" rather than individual points.
- **Spatial Context**: Coordinates `X, Y, Z` help in regional geological modeling.

### D. Modeling Strategy
- **Sequence Models**: LSTM and GRU are suitable for capturing the sequential nature of depth-indexed data.
- **Transformers**: Small Transformer blocks can capture long-range dependencies and non-linear mappings between horizontal and vertical signatures.
- **Backend**: JAX and Keras will be used for high-performance training on the available hardware (NVIDIA RTX 3050).

## 4. Evaluation Metric
The competition likely uses **Root Mean Squared Error (RMSE)** between predicted and actual `tvt` values.
- Precision in evaluation zones is critical for geosteering accuracy.
