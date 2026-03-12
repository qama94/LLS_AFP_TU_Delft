# Automated Dimensional Characterisation of Overlap Bead Peaks from Laser Line Scanner Surface Profiles

**Author:** Gamar Ismayilova
**Affiliation:** TU Delft, Department of Aerospace Structures and Materials
**Contact:** gamar.ismayilova@gmail.com

---

## 1. Overview

This repository provides a reproducible algorithm for the automated dimensional characterisation of overlap bead peaks from cross-section surface profiles acquired by a Laser Line Scanner (LLS). The algorithm is designed for Automated Fibre Placement (AFP) tape specimens and performs batch processing of multiple specimens under identical conditions, producing quantitative geometric descriptors suitable for statistical analysis.

### Example output

| Type 1 (2-layer) | Type 2 (2-layer + repass) | Type 3 (3-layer) |
|:-:|:-:|:-:|
| ![Type1](examples/Sample%201_1A_peak_analysis.png) | ![Type2](examples/Sample%202_1A_peak_analysis.png) | ![Type3](examples/Sample%203_1A_peak_analysis.png) |

### Repository contents

| File | Description |
|------|-------------|
| `overlap_peak_analysis.py` | Main algorithm — single file, self-contained |
| `statistical_analysis.py` | Statistical analysis (ANOVA, descriptive stats, box plots) |
| `README.md` | Methodology, mathematical foundations, and references |
| `LICENSE` | MIT licence |
| `CITATION.cff` | Machine-readable citation metadata |
| `requirements.txt` | Python dependencies |
| `tests/test_algorithm.py` | Validation tests (7 tests) |
| `examples/` | Example output figures for each specimen type |

---

## 2. Mathematical Foundations

### 2.1 Signal Conditioning

Raw LLS profiles contain two types of measurement noise: isolated spikes (instrument artefacts) and broadband noise (electronic/optical fluctuations). These are treated sequentially.

#### 2.1.1 Spike Removal via Median Absolute Deviation (MAD)

For each sample *z_i* in the height signal, a symmetric neighbourhood *N_i* of *w* points centred on index *i* is defined. Two robust estimators are computed within this window:

```
m_i = median({z_j : j ∈ N_i})

MAD_i = median({|z_j − m_i| : j ∈ N_i})
```

The MAD is converted to a scale estimator consistent with the standard deviation of a Gaussian distribution via the consistency constant *c* = 1.4826, which satisfies *c* · MAD = *σ* under normality (Hampel, 1974):

```
σ̂_i = c · MAD_i
```

The normalised deviation of each sample is:

```
z̃_i = |z_i − m_i| / σ̂_i
```

A sample is classified as a spike if *z̃_i* > *k*, where *k* is the rejection threshold (default *k* = 3, corresponding to a 0.27% false-positive rate under Gaussianity). Detected spikes are replaced by the local median *m_i*.

The MAD estimator has a 50% breakdown point: up to half the data within the window can be corrupted before the estimator fails (Leys et al., 2013). This makes the method robust against the very outliers it seeks to remove, unlike classical methods based on mean ± standard deviation.

**Default parameters:** *w* = 15 samples, *k* = 3.0

#### 2.1.2 Median Filtering

After spike removal, a running median filter of kernel width *K* = 5 suppresses broadband noise while preserving sharp step edges at bead boundaries. The median filter is a rank-order non-linear operator:

```
z_filtered(i) = median({z(j) : j ∈ [i − K/2, i + K/2]})
```

Unlike linear smoothing (e.g., Gaussian), the median filter does not introduce ringing artefacts near discontinuities (Tukey, 1977).

#### 2.1.3 Origin Normalisation

Both the lateral position *X* and height *Z* are translated so that their minima equal zero:

```
X_norm = X − min(X)
Z_norm = Z − min(Z)
```

This removes scanner coordinate offsets and enables direct comparison across specimens.

### 2.2 Substrate Region Identification

The substrate (deposited bead material) occupies the upper portion of the height distribution. Its boundary is established by applying a percentile-based threshold to a Gaussian-smoothed signal:

```
Z_smooth = G_σ * Z       (Gaussian convolution, σ = 3.0)
threshold = P_q(Z_smooth) (q-th percentile, default q = 67)
```

All indices where *Z_smooth* > *threshold* are classified as substrate. Only substrate-region data is used for peak detection.

### 2.3 Peak Detection with Two-Stage Validation

#### Stage 1: Prominence-Based Detection

Peaks are detected using `scipy.signal.find_peaks` with the following constraints:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Minimum prominence | 0.04 mm | Rejects noise fluctuations |
| Minimum width | 5 samples | Rejects narrow spikes |
| Minimum distance | 10 samples | Prevents double-counting |

#### Stage 2: Post-Detection Validation

Two filters reject artefacts that pass the prominence criterion:

**Filter A — Minimum base width:**
Peaks with base width < 0.5 mm are rejected as surface roughness features rather than genuine overlap beads.

**Filter B — Spatial coherence:**
Peaks located in the outer 10% of the substrate lateral extent are rejected as edge artefacts. For a substrate spanning [*X_min*, *X_max*], a peak at position *X_p* is rejected if:

```
X_p < X_min + 0.1 · (X_max − X_min)    or
X_p > X_max − 0.1 · (X_max − X_min)
```

#### Stage 3: Property Recomputation

After rejection, the substrate region is flattened at the rejected peak locations, and `find_peaks` is re-run to obtain accurate properties (prominences, base widths) for the remaining valid peaks.

### 2.4 Geometric Characterisation

For each validated peak, the algorithm computes the following dimensional parameters:

#### 2.4.1 Base Boundary Detection

The base boundaries are determined by scanning outward from the peak apex until the signal drops below a transition level:

```
Z_transition = Z_saddle + 0.25 · prominence
```

where *Z_saddle* is the height of the higher of the two bounding saddle points. Scanning outward from the apex:

- **Left base** (*X_L*): first position where *Z*(*X*) < *Z_transition* going left from apex.
- **Right base** (*X_R*): first position where *Z*(*X*) < *Z_transition* going right from apex.

```
Base width:  W_base = |X_R − X_L|
```

#### 2.4.2 Width at Half Maximum (FWHM)

```
Z_half = Z_saddle + 0.5 · prominence
FWHM = |X_half_right − X_half_left|
```

where *X_half_left* and *X_half_right* are the interpolated positions where the signal crosses *Z_half*.

#### 2.4.3 Width at 25% Prominence (W25)

```
Z_25 = Z_saddle + 0.25 · prominence
W25 = |X_25_right − X_25_left|
```

#### 2.4.4 Cross-Sectional Area

The area above a linear baseline connecting the two base points is computed via trapezoidal integration:

```
A = ∫_{X_L}^{X_R} [Z(X) − Z_baseline(X)] dX
```

where *Z_baseline*(*X*) is the linear interpolation between (*X_L*, *Z_L*) and (*X_R*, *Z_R*). This quantity is proportional to the excess material volume per unit bead length.

#### 2.4.5 Flank Slope Angles

Linear regression of each flank (base to apex) yields slope angles:

```
θ_L = arctan(dZ/dX) for the left flank  [X_L, X_apex]
θ_R = arctan(dZ/dX) for the right flank [X_apex, X_R]
θ_avg = (|θ_L| + |θ_R|) / 2
```

#### 2.4.6 Asymmetry Ratio

```
Asymmetry = W_left / W_right
```

where *W_left* = |*X_apex* − *X_half_left*| and *W_right* = |*X_half_right* − *X_apex*| are the left and right half-widths at half-prominence. A value of 1.0 indicates perfect symmetry.

#### 2.4.7 Apex Curvature

The curvature at the peak apex is estimated from the second derivative of the smoothed signal:

```
κ = d²Z/dX² |_{X = X_apex}
```

computed via central finite differences. Negative curvature indicates a convex (peaked) apex; large magnitude indicates a sharp peak.

---

## 3. Installation

```bash
pip install -r requirements.txt
```

Dependencies: numpy (≥1.21), scipy (≥1.7), matplotlib (≥3.5). Works on any platform: Windows, macOS, Linux, Google Colab, Jupyter, VS Code.

---

## 4. Usage

### Single specimen

```python
from overlap_peak_analysis import analyze_sample

# Just pass the file path — sample name is auto-derived from filename
# Sample_2.1A_00001.slk → plot title "Sample 2.1A"
result = analyze_sample("/path/to/Sample_2.1A_00001.slk")
```

### With explicit output folder

```python
result = analyze_sample(
    "/path/to/Sample_2.1A_00001.slk",
    output_dir="/path/to/output"
)
```

### Batch processing

```python
from overlap_peak_analysis import analyze_multiple_samples
import glob

files = sorted(glob.glob("/path/to/data/*.slk"))
results = analyze_multiple_samples(files, names=None, output_dir="/path/to/output")
```

### Statistical analysis

```bash
python statistical_analysis.py all_measurements.csv -o stats_output
```

---

## 5. Input Format

SYLK (.slk) files containing two-column profile data: X position (column 1) and Z height (column 2). This is the standard export format from commercial laser line scanner software.

---

## 6. Output

For each specimen:
- **PNG figure** — two-panel plot: overview (full profile + overlap zone) and zoomed detail (annotated peak dimensions).
- **Text report** — structured measurement table with all geometric parameters.
- **Summary CSV** — batch results in `all_samples_summary.csv`.

---

## 7. Default Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `median_kernel` | 5 | Median filter window (samples) |
| `spike_window` | 15 | MAD neighbourhood size (samples) |
| `spike_threshold` | 3.0 | Spike rejection threshold (MAD units) |
| `substrate_percentile` | 67 | Percentile for substrate delineation |
| `prominence_threshold` | 0.04 | Minimum peak prominence (mm) |
| `min_peak_width` | 5 | Minimum peak width (samples) |
| `min_peak_distance` | 10 | Minimum inter-peak distance (samples) |
| `min_base_width_mm` | 0.5 | Minimum base width for validation (mm) |
| `spatial_outlier_factor` | 2.0 | Edge rejection zone (×IQR from median position) |

---

## 8. Validation

```bash
# With pytest:
python -m pytest tests/ -v

# Without pytest:
python tests/test_algorithm.py
```

The 7 validation tests verify:
1. Spike removal detects and replaces artificial spikes.
2. Clean data is not corrupted by false spike detections.
3. Origin normalisation produces zero-referenced coordinates.
4. Substrate region is correctly identified.
5. Peak detection finds peaks at valid positions.
6. Correct number of peaks is detected.
7. Geometric measurements are within physically reasonable ranges.

---

## 9. References

1. Hampel, F. R. (1974). The influence curve and its role in robust estimation. *Journal of the American Statistical Association*, 69(346), 383–393. https://doi.org/10.1080/01621459.1974.10482962

2. Tukey, J. W. (1977). *Exploratory Data Analysis*. Addison-Wesley. ISBN: 978-0-201-07616-5

3. Leys, C., Ley, C., Klein, O., Bernard, P., & Licata, L. (2013). Detecting outliers: Do not use standard deviation around the mean, use absolute deviation around the median. *Journal of Experimental Social Psychology*, 49(4), 764–766. https://doi.org/10.1016/j.jesp.2013.03.013

---

## 10. Licence

MIT Licence. See [LICENSE](LICENSE) for the full text.

---

## 11. Citation

If you use this software in published research, please cite it using the metadata in [CITATION.cff](CITATION.cff). On GitHub, click the "Cite this repository" button for a formatted citation.
