#!/usr/bin/env python3
"""
Statistical Analysis of Layer Detection Results
Author: Gamar Ismayilova, TU Delft
See README.md for methodology.
"""

import os, sys, csv, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from layer_detection import run, format_sample_name

METRICS = [
    ('n_tape_layers', 'Number of Tape Layers'),
    ('layer1_width', 'Layer 1 Width [mm]'),
    ('layer1_height', 'Layer 1 Avg Height [µm]'),
    ('layer2_width', 'Layer 2 Width [mm]'),
    ('layer2_height', 'Layer 2 Avg Height [µm]'),
]

TYPE_LABELS = {'1': 'Type 1 (2-layer)', '2': 'Type 2 (2L+repass)', '3': 'Type 3 (3-layer)'}
BOX_COLORS = {'1': '#4a90d9', '2': '#e8a838', '3': '#d94a4a'}


def run_batch(data_dir, output_dir):
    """Run layer detection on all .slk files and collect measurements."""
    os.makedirs(output_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(data_dir, '*.slk')))
    print(f'Found {len(files)} .slk files in {data_dir}')

    rows = []
    for fp in files:
        name = format_sample_name(fp)
        base = os.path.basename(fp).replace('_00010.slk', '').replace('_00001.slk', '')
        parts = base.split('_')
        spec_type = parts[1] if len(parts) > 1 else '0'
        pos_rep = parts[2] if len(parts) > 2 else '0'
        pos = pos_rep[0] if pos_rep else '0'
        rep = pos_rep[1] if len(pos_rep) > 1 else ''

        try:
            result = run(fp, output_dir=os.path.join(output_dir, 'figures'))
            row = {
                'sample': name,
                'specimen_type': spec_type,
                'position': pos,
                'replicate': rep,
                'n_tape_layers': result['n_tape_layers'],
            }
            # Extract per-layer widths and heights
            for i in range(1, 4):
                lname = f'Layer {i}'
                row[f'layer{i}_width'] = result['layer_widths'].get(lname, None)
                h = result['layer_heights'].get(lname, None)
                row[f'layer{i}_height'] = h * 1000 if h is not None else None  # mm → µm
            rows.append(row)
        except Exception as e:
            print(f'  ERROR {name}: {e}')

    # Save CSV
    csv_path = os.path.join(output_dir, 'layer_measurements.csv')
    if rows:
        keys = rows[0].keys()
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f'\nSaved {len(rows)} measurements to {csv_path}')
    return rows


def compute_stats(rows, output_dir):
    """Descriptive statistics + ANOVA + box plots."""
    os.makedirs(output_dir, exist_ok=True)

    # Group by specimen type
    groups = {}
    for r in rows:
        g = str(r['specimen_type'])
        groups.setdefault(g, []).append(r)
    gnames = sorted(groups.keys())

    # Descriptive stats
    print(f'\n{"="*80}')
    print(f'  DESCRIPTIVE STATISTICS BY SPECIMEN TYPE')
    print(f'{"="*80}')

    for mk, ml in METRICS:
        print(f'\n  {ml}:')
        print(f'  {"Group":<22} {"N":>3} {"Mean":>10} {"Std":>10} {"CV%":>7}')
        print(f'  {"-"*55}')
        for g in gnames:
            vals = [float(r[mk]) for r in groups[g] if r.get(mk) is not None]
            if vals:
                m = np.mean(vals)
                s = np.std(vals, ddof=1) if len(vals) > 1 else 0
                cv = (s / m * 100) if m != 0 else 0
                lab = TYPE_LABELS.get(g, f'Type {g}')
                print(f'  {lab:<22} {len(vals):>3} {m:>10.2f} {s:>10.3f} {cv:>7.1f}')

    # ANOVA
    print(f'\n{"="*80}')
    print(f'  ONE-WAY ANOVA BY SPECIMEN TYPE')
    print(f'{"="*80}')
    print(f'\n  {"Metric":<30} {"F":>10} {"p-value":>12} {"Result":>15}')
    print(f'  {"-"*70}')

    for mk, ml in METRICS:
        gvals = []
        for g in gnames:
            vals = [float(r[mk]) for r in groups[g] if r.get(mk) is not None]
            gvals.append(vals)
        valid = [v for v in gvals if len(v) >= 2]
        if len(valid) >= 2:
            f_stat, p_val = stats.f_oneway(*valid)
            sig = 'Significant *' if p_val < 0.05 else 'Not significant'
            print(f'  {ml:<30} {f_stat:>10.3f} {p_val:>12.6f} {sig:>15}')
            if p_val < 0.05 and len(gnames) > 2:
                nc = len(list(combinations(range(len(gnames)), 2)))
                for i, j in combinations(range(len(gnames)), 2):
                    if len(gvals[i]) >= 2 and len(gvals[j]) >= 2:
                        _, tp = stats.ttest_ind(gvals[i], gvals[j])
                        cp = min(tp * nc, 1.0)
                        print(f'    Post-hoc: Type {gnames[i]} vs Type {gnames[j]}: '
                              f'p={cp:.4f}{" *" if cp < 0.05 else ""}')

    # Box plots
    for mk, ml in METRICS:
        data = []
        labels = []
        for g in gnames:
            vals = [float(r[mk]) for r in groups[g] if r.get(mk) is not None]
            if vals:
                data.append(vals)
                labels.append(TYPE_LABELS.get(g, f'Type {g}'))

        if not data:
            continue

        fig, ax = plt.subplots(figsize=(8, 6))
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.5,
                        showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor='white',
                                       markeredgecolor='black', markersize=7),
                        medianprops=dict(color='black', linewidth=2))
        for patch, g in zip(bp['boxes'], gnames):
            patch.set_facecolor(BOX_COLORS.get(g, '#ccc'))
            patch.set_alpha(0.6)
            patch.set_linewidth(1.5)
        for i, vals in enumerate(data):
            xj = np.random.default_rng(42).normal(i + 1, 0.06, len(vals))
            ax.scatter(xj, vals, alpha=0.6, s=25, color='black', zorder=5, edgecolors='none')

        ax.set_ylabel(ml, fontsize=13, fontweight='bold')
        ax.set_title(ml, fontsize=14, fontweight='bold')
        ax.tick_params(labelsize=11)
        ax.grid(True, axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'boxplot_{mk}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # Display box plots in notebooks
    try:
        from IPython.display import display, Image
        for mk, _ in METRICS:
            bp_path = os.path.join(output_dir, f'boxplot_{mk}.png')
            if os.path.exists(bp_path):
                display(Image(filename=bp_path))
    except ImportError:
        pass

    print(f'\n  Box plots saved to {output_dir}')


if __name__ == '__main__':
    def upload_files():
        """Get data directory — works on any platform."""
        try:
            from google.colab import files
            print("Upload your .slk file(s):")
            uploaded = files.upload()
            return '.', list(uploaded.keys())
        except ImportError:
            pass

        import sys
        if len(sys.argv) > 1:
            data_dir = sys.argv[1]
        else:
            data_dir = input("Enter data directory path (e.g. ../data): ").strip()
        return data_dir, None

    data_dir, _ = upload_files()
    rows = run_batch(data_dir, output_dir='layer_statistics')
    if rows:
        compute_stats(rows, output_dir='layer_statistics')
