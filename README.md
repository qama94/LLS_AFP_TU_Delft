# AFP Surface Profilometry Analysis

> Automated Python pipeline for processing laser line scanner data from a KUKA KR210 industrial robot. 
> Two algorithms — overlap bead peak detection and tape layer geometry characterisation — validated 
> across 32 AFP composite specimens. Published at ACM7 2026.

**Author:** Gamar Ismayilova  
**Affiliation:** TU Delft, Department of Aerospace Structures and Materials  
**Contact:** gamar.ismayilova@gmail.com

---

## Algorithms

This repository contains two complementary algorithms for automated dimensional characterisation 
of Automated Fibre Placement (AFP) composite specimens. Each algorithm has its own detailed README 
with methodology, parameters, and example outputs.

### [Algorithm 1: Overlap Bead Peak Characterisation →](Overlap_Analysis/README.md)

Detects overlap bead peaks from cross-section profiles and computes geometric descriptors:
base width, FWHM, cross-sectional area, flank slope angles, asymmetry ratio, and apex curvature.

### [Algorithm 2: Tape Layer Detection →](Layer_analysis/README.md)

Detects tape layer count, reference height levels, and geometric vertices (feet and shoulders) 
from cross-section profiles. Handles 2-layer, 2-layer with repass, and 3-layer configurations 
without manual parameter tuning.

---

## Repository Structure

```

├── Overlap_Analysis/          # Algorithm 1: Overlap bead peak characterisation
│   ├── overlap_peak_analysis.py
│   ├── statistical_analysis.py
│   ├── test_algorithm.py
│   └── examples/
│
├── Layer_analysis/            # Algorithm 2: Tape layer detection
│   ├── layer_detection.py
│   ├── statistical_analysis.py
│   ├── test_layer_detection.py
│   └── examples/
│
├── LLS_raw_data/              # Raw .slk profile data (32 specimens)
├── CITATION.cff
├── LICENSE
├── README.md
└── requirements.txt

```

---

## Quick Start

```bash
pip install -r requirements.txt
```

```python
# Overlap peak analysis
from Overlap_Analysis.overlap_peak_analysis import analyze_sample
result = analyze_sample("LLS_raw_data/Sample_3_1A_00010.slk", output_dir="results")

# Layer detection
from Layer_analysis.layer_detection import run
run("LLS_raw_data/Sample_3_1A_00010.slk", output_dir="results")
```

---

## Specimen Design

| Type | Layup                         | Specimens                          |
| ---- | ----------------------------- | ---------------------------------- |
| 1    | 2-layer with overlap          | 12 (positions 1–4, replicates A–C) |
| 2    | 2-layer with overlap + repass | 11 (positions 1–4, replicates A–C) |
| 3    | 3-layer                       | 9 (positions 1–3, replicates A–C)  |

---

## Results

- Validated across 32 specimens across 3 specimen types
- Repassing consistently increases overlap size by approximately 1mm per additional pass (1mm → 2mm → 3mm)
- Full factorial experimental design across specimen types
- Cross-validated against confocal microscopy measurements

---

## Publication

Ismayilova G., Florindo A., Peeters D.  
**"Effects of repassing on the geometric characteristics of CF/LM-PAEK tapes deposited via Humm3-assisted in-situ consolidated AFP"**  
7th International Symposium on Automated Composite Manufacturing (ACM7), SAM XL, TU Delft, April 2026.  
[Abstract & manuscript](https://conf.acm7.nl/proceedings/proceedings/display_manuscript/38.htm)

---

## Licence

MIT. See [LICENSE](LICENSE).
