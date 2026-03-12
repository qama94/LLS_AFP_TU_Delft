# Automated Dimensional Characterisation of Overlap Bead Peaks from Laser Line Scanner Surface Profiles

**Author:** Gamar Ismayilova
**Affiliation:** TU Delft, Department of Aerospace Structures and Materials
**Contact:** gamar.ismayilova@gmail.com

---

## 1. Overview

This repository provides a reproducible algorithm for the automated dimensional characterisation of overlap bead peaks from cross-section surface profiles acquired by a Laser Line Scanner (LLS). The algorithm is designed for Automated Fibre Placement (AFP) tape specimens and performs batch processing of multiple specimens under identical conditions, producing quantitative geometric descriptors suitable for statistical analysis.


### Repository contents

| File | Description |
|------|-------------|
| `overlap_peak_analysis.py` | Main algorithm — single file, self-contained |
| `README.md` | Methodology, usage instructions, and references |
| `LICENSE` | MIT licence — permits free reuse with attribution |
| `CITATION.cff` | Machine-readable citation metadata (used by GitHub) |
| `requirements.txt` | Python dependencies (numpy, scipy, matplotlib) |
| `tests/test_algorithm.py` | Validation tests (7 tests, run with pytest or standalone) |
| `tests/__init__.py` | Empty file required by Python to recognise the tests folder |
| `examples/` | Example output figures |

---

## 2. Algorithm Pipeline

The algorithm processes each specimen through seven sequential stages:

1. **Data ingestion** — parsing of SYLK (.slk) formatted profile exports into position–height (X, Z) arrays.
2. **Spike removal** — robust outlier detection based on the Median Absolute Deviation (MAD) within a sliding window (Hampel, 1974; Leys et al., 2013).
3. **Median filtering** — non-linear noise suppression that preserves sharp geometric transitions (Tukey, 1977).
4. **Origin normalisation** — translation of both axes to a zero reference.
5. **Substrate identification** — delineation of the bead region via percentile-based thresholding of a Gaussian-smoothed signal.
6. **Peak detection** — prominence-based detection with two-stage validation (minimum base width and spatial coherence filters).
7. **Geometric characterisation** — computation of base width, full width at half maximum (FWHM), width at 25% prominence, cross-sectional area, flank slope angles, asymmetry ratio, and apex curvature for each detected peak.

### 2.1 Signal Conditioning

#### Spike Removal

For each sample z_i, a neighbourhood of w points is defined. The local median m_i and MAD are computed:

    MAD_i = median(|z_j - m_i|)

The MAD is scaled by the consistency constant c = 1.4826 to obtain a robust standard deviation estimate. A sample is flagged as a spike if:

    |z_i - m_i| / (c * MAD_i) > k

where k = 3 (default). Detected spikes are replaced by m_i. The MAD estimator has a 50% breakdown point, making it robust against the outliers it seeks to remove (Leys et al., 2013).

#### Median Filtering

A running median filter of width 5 samples suppresses broadband noise while preserving edge sharpness (Tukey, 1977).

### 2.2 Peak Detection

Peaks are detected using `scipy.signal.find_peaks` with a minimum prominence threshold of 0.04 mm, minimum width of 5 samples, and minimum inter-peak distance of 10 samples. A two-stage post-detection validation rejects:

- Peaks with base width below 0.5 mm (roughness artefacts).
- Peaks in the outer 10% of the substrate extent (edge artefacts).

### 2.3 Geometric Characterisation

For each validated peak, the algorithm computes:

| Parameter | Definition |
|-----------|------------|
| Base width | Distance between left and right base points at the saddle level |
| FWHM | Width at 50% of prominence above the saddle |
| W25 | Width at 25% of prominence |
| Cross-sectional area | Integrated area above the saddle level (trapezoidal rule) |
| Left/right flank slope | Inclination angle of each flank (degrees) |
| Asymmetry ratio | Ratio of left to right half-widths at half-prominence |
| Apex curvature | Second derivative at the peak apex |

The base boundaries are determined by scanning outward from the apex until the signal drops below 25% of the peak prominence above the saddle.

---

## 3. Installation

```bash
pip install -r requirements.txt
```

This installs the three required packages: numpy, scipy, and matplotlib. No other dependencies are needed.

---

## 4. Usage

### Single specimen

The sample name is automatically extracted from the filename:

```python
from overlap_peak_analysis import analyze_sample

# Just pass the file path — name and output folder are automatic
result = analyze_sample("/content/Sample_2.1A_00001.slk")
```

The filename `Sample_2.1A_00001.slk` produces the plot title `Sample 2.1A`.

### With explicit output folder

```python
result = analyze_sample(
    "/content/Sample_2.1A_00001.slk",
    output_dir="/content/results"
)
```

### Batch processing

```python
from overlap_peak_analysis import analyze_multiple_samples
import glob

files = sorted(glob.glob("/content/data/*.slk"))
names = None  # auto-derived from filenames

results = analyze_multiple_samples(files, names, output_dir="/content/results")
```

---

## 5. Input Format

The algorithm reads SYLK (.slk) files containing two-column profile data (X position in column 1, Z height in column 2). These are the standard export format from commercial laser line scanner software.

---

## 6. Output

For each specimen, the algorithm produces:

- **PNG figure** — two-panel plot: overview (full profile with overlap zone) and zoomed detail (peak dimensions annotated with base width, FWHM, prominence, area, and slope angles).
- **Text report** — structured measurement table with all geometric parameters.

For batch runs, a summary CSV (`all_samples_summary.csv`) is additionally generated.

---

## 7. Default Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `median_kernel` | 5 | Median filter window (samples) |
| `spike_window` | 15 | MAD neighbourhood size (samples) |
| `spike_threshold` | 3.0 | Spike rejection threshold (MAD units) |
| `prominence_threshold` | 0.04 | Minimum peak prominence (mm) |
| `min_peak_width` | 5 | Minimum peak width (samples) |
| `min_peak_distance` | 10 | Minimum inter-peak distance (samples) |
| `min_base_width_mm` | 0.5 | Minimum base width for validation (mm) |

---

## 8. Validation

Run the test suite to verify algorithm correctness:

```bash
# With pytest installed:
python -m pytest tests/ -v

# Without pytest:
python tests/test_algorithm.py
```

The 7 tests verify:
- SYLK file parsing produces valid arrays.
- Spike removal detects and replaces artificial spikes.
- Clean data is not corrupted by false spike detection.
- Output coordinates are normalised to zero.
- Substrate region is correctly identified.
- Peak detection finds the expected number of peaks.
- Geometric measurements are within physically reasonable ranges.

---

## 9. References

1. Hampel, F. R. (1974). The influence curve and its role in robust estimation. *Journal of the American Statistical Association*, 69(346), 383–393. https://doi.org/10.1080/01621459.1974.10482962

2. Tukey, J. W. (1977). *Exploratory Data Analysis*. Addison-Wesley. ISBN: 978-0-201-07616-5

3. Leys, C., Ley, C., Klein, O., Bernard, P., & Licata, L. (2013). Detecting outliers: Do not use standard deviation around the mean, use absolute deviation around the median. *Journal of Experimental Social Psychology*, 49(4), 764–766. https://doi.org/10.1016/j.jesp.2013.03.013

---

## 10. Licence

MIT Licence. See [LICENSE](LICENSE) for the full text. This permits free use, modification, and distribution with attribution.

---

## 11. Citation

If you use this software in your research, please cite it. GitHub provides a formatted citation via the "Cite this repository" button, which reads from [CITATION.cff](CITATION.cff).
