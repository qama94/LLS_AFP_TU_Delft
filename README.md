# AFP Surface Profilometry Analysis

**Author:** Gamar Ismayilova
**Affiliation:** TU Delft, Department of Aerospace Structures and Materials

Automated analysis of Laser Line Scanner (LLS) surface profiles from Automated Fibre Placement (AFP) tape specimens. Two complementary algorithms for dimensional characterisation of overlap beads and tape layer geometry.

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
├── .gitignore
├── CITATION.cff
├── LICENSE
├── README.md
└── requirements.txt
```

## Quick Start

```bash
pip install -r requirements.txt
```

```python
# Overlap peak analysis
from overlap_peak_analysis.overlap_peak_analysis import analyze_sample
result = analyze_sample("data/Sample_3_1A_00010.slk", output_dir="results")

# Layer detection
from layer_detection.layer_detection import run
run("data/Sample_3_1A_00010.slk", "Sample 3_1A", "results/Sample_3_1A_layers.png")
```

## Specimen Design

| Type | Layup | Specimens |
|------|-------|-----------|
| 1 | 2-layer with overlap | 12 (positions 1–4, replicates A–C) |
| 2 | 2-layer with overlap + repass | 11 (positions 1–4, replicates A–C) |
| 3 | 3-layer | 9 (positions 1–3, replicates A–C) |

## Licence

MIT. See [LICENSE](LICENSE).
