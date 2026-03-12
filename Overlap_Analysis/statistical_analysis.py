#!/usr/bin/env python3
"""
Statistical Analysis of Overlap Bead Geometric Parameters
Author: Gamar Ismayilova, TU Delft

Uses PRIMARY PEAK per sample (most prominent) to ensure equal weighting.
Produces: descriptive stats, normality tests, ANOVA, box plots, CSV tables.

Usage: python statistical_analysis.py all_measurements.csv -o stats_output
"""

import os, sys, csv, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from itertools import combinations

METRICS = [
    ('prominence', 'Prominence [mm]'), ('base_width', 'Base Width [mm]'),
    ('fwhm', 'FWHM [mm]'), ('area', 'Area [mm²]'),
    ('avg_slope', 'Avg Slope [°]'), ('asymmetry', 'Asymmetry Ratio'),
    ('height_z', 'Peak Height [mm]'),
]
TYPE_LABELS = {'1':'Type 1 (2-layer)','2':'Type 2 (2L+repass)','3':'Type 3 (3-layer)'}
BOX_COLORS = {'1':'#4a90d9','2':'#e8a838','3':'#d94a4a'}

def load_measurements(csv_path):
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            for k in row:
                try: row[k] = float(row[k])
                except: pass
            rows.append(row)
    return rows

def select_primary_peaks(rows):
    samples = {}
    for r in rows:
        samples.setdefault(r['sample'], []).append(r)
    return [max(v, key=lambda p: float(p['prominence'])) for v in sorted(samples.values(), key=lambda v: v[0]['sample'])]

def compute_descriptive(rows, gkey):
    groups = {}
    for r in rows:
        g = str(int(float(r[gkey])))
        groups.setdefault(g, []).append(r)
    res = {}
    for g in sorted(groups):
        res[g] = {}
        for mk, ml in METRICS:
            vals = [float(r[mk]) for r in groups[g] if r.get(mk) is not None]
            if vals:
                m, s = np.mean(vals), np.std(vals, ddof=1) if len(vals)>1 else 0
                res[g][mk] = dict(mean=m, std=s, cv=(s/m*100) if m else 0,
                                   mn=np.min(vals), mx=np.max(vals), n=len(vals), values=vals)
    return res

def print_stats(desc, label):
    print(f'\n{"="*85}\n  DESCRIPTIVE STATISTICS — {label} (primary peak per sample)\n{"="*85}')
    for mk, ml in METRICS:
        print(f'\n  {ml}:')
        print(f'  {"Group":<22} {"N":>3} {"Mean":>10} {"Std":>10} {"Min":>10} {"Max":>10} {"CV%":>7}')
        print(f'  {"-"*75}')
        for g in sorted(desc):
            s = desc[g].get(mk, {})
            if s:
                lab = TYPE_LABELS.get(g, g)
                print(f'  {lab:<22} {s["n"]:>3} {s["mean"]:>10.4f} {s["std"]:>10.4f} {s["mn"]:>10.4f} {s["mx"]:>10.4f} {s["cv"]:>7.1f}')

def run_normality(rows, gkey):
    groups = {}
    for r in rows:
        g = str(int(float(r[gkey])))
        groups.setdefault(g, []).append(r)
    print(f'\n{"="*85}\n  NORMALITY (Shapiro-Wilk)\n{"="*85}')
    for mk, ml in METRICS:
        print(f'\n  {ml}:')
        for g in sorted(groups):
            vals = [float(r[mk]) for r in groups[g]]
            if len(vals) >= 3:
                w, p = stats.shapiro(vals)
                print(f'    Type {g} (n={len(vals)}): W={w:.4f}, p={p:.4f} → {"Normal" if p>0.05 else "Not normal"}')

def run_anova(rows, gkey, label):
    groups = {}
    for r in rows:
        g = str(int(float(r[gkey])))
        groups.setdefault(g, []).append(r)
    gn = sorted(groups)
    print(f'\n{"="*85}\n  ONE-WAY ANOVA — {label}\n{"="*85}')
    print(f'\n  {"Metric":<25} {"F":>10} {"p-value":>12} {"Result":>15}')
    print(f'  {"-"*65}')
    res = {}
    for mk, ml in METRICS:
        gv = [[float(r[mk]) for r in groups[g]] for g in gn]
        valid = [v for v in gv if len(v)>=2]
        if len(valid) >= 2:
            f, p = stats.f_oneway(*valid)
            sig = p < 0.05
            print(f'  {ml:<25} {f:>10.3f} {p:>12.6f} {"Significant *" if sig else "Not significant":>15}')
            res[mk] = dict(f=f, p=p, sig=sig)
            if sig and len(gn) > 2:
                nc = len(list(combinations(range(len(gn)), 2)))
                for i, j in combinations(range(len(gn)), 2):
                    if len(gv[i])>=2 and len(gv[j])>=2:
                        _, tp = stats.ttest_ind(gv[i], gv[j])
                        cp = min(tp*nc, 1.0)
                        print(f'    Post-hoc: Type {gn[i]} vs Type {gn[j]}: p={cp:.4f}{" *" if cp<0.05 else ""}')
    return res

def create_boxplots(rows, out_dir):
    groups = {}
    for r in rows:
        g = str(int(float(r['specimen_type'])))
        groups.setdefault(g, []).append(r)
    gn = sorted(groups)
    for mk, ml in METRICS:
        data = [[float(r[mk]) for r in groups[g]] for g in gn]
        labels = [TYPE_LABELS.get(g, g) for g in gn]
        fig, ax = plt.subplots(figsize=(8, 6))
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.5,
                        showmeans=True, meanprops=dict(marker='D', markerfacecolor='white', markeredgecolor='black', markersize=7),
                        medianprops=dict(color='black', linewidth=2), whiskerprops=dict(linewidth=1.5), capprops=dict(linewidth=1.5))
        for patch, g in zip(bp['boxes'], gn):
            patch.set_facecolor(BOX_COLORS.get(g, '#ccc')); patch.set_alpha(0.6); patch.set_linewidth(1.5)
        for i, (vals, g) in enumerate(zip(data, gn)):
            xj = np.random.default_rng(42).normal(i+1, 0.06, len(vals))
            ax.scatter(xj, vals, alpha=0.6, s=25, color='black', zorder=5, edgecolors='none')
        ax.set_ylabel(ml, fontsize=13, fontweight='bold')
        ax.set_title(ml, fontsize=14, fontweight='bold')
        ax.tick_params(labelsize=11); ax.grid(True, axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'boxplot_{mk}.png'), dpi=300, bbox_inches='tight')
        plt.close()

def create_summary(rows, out_dir):
    groups = {}
    for r in rows:
        g = str(int(float(r['specimen_type'])))
        groups.setdefault(g, []).append(r)
    gn = sorted(groups)
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    axes = axes.flatten()
    for idx, (mk, ml) in enumerate(METRICS):
        ax = axes[idx]
        data = [[float(r[mk]) for r in groups[g]] for g in gn]
        labels = [TYPE_LABELS.get(g, g) for g in gn]
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.45, showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor='white', markeredgecolor='black', markersize=6),
                        medianprops=dict(color='black', linewidth=1.5))
        for patch, g in zip(bp['boxes'], gn):
            patch.set_facecolor(BOX_COLORS.get(g, '#ccc')); patch.set_alpha(0.6)
        for i, vals in enumerate(data):
            xj = np.random.default_rng(42).normal(i+1, 0.05, len(vals))
            ax.scatter(xj, vals, alpha=0.5, s=18, color='black', zorder=5, edgecolors='none')
        ax.set_ylabel(ml, fontsize=10, fontweight='bold')
        ax.set_title(ml, fontsize=11, fontweight='bold')
        ax.tick_params(labelsize=8); ax.grid(True, axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    for i in range(len(METRICS), len(axes)): axes[i].set_visible(False)
    fig.suptitle('Overlap Bead Geometry by Specimen Type (primary peak, n=32)', fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.savefig(os.path.join(out_dir, 'summary_boxplots.png'), dpi=300, bbox_inches='tight')
    plt.close()

def export_csv(desc, out_dir, fname):
    path = os.path.join(out_dir, fname)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Group','Metric','N','Mean','Std','Min','Max','CV%'])
        for g in sorted(desc):
            for mk, ml in METRICS:
                s = desc[g].get(mk, {})
                if s: w.writerow([TYPE_LABELS.get(g,g), ml, s['n'], f'{s["mean"]:.4f}', f'{s["std"]:.4f}', f'{s["mn"]:.4f}', f'{s["mx"]:.4f}', f'{s["cv"]:.1f}'])
    print(f'  Saved: {path}')

def run_full_analysis(csv_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rows = load_measurements(csv_path)
    print(f'Loaded {len(rows)} peak measurements from {len(set(r["sample"] for r in rows))} samples.')
    primary = select_primary_peaks(rows)
    print(f'Selected {len(primary)} primary peaks (one per sample).')
    
    desc = compute_descriptive(primary, 'specimen_type')
    print_stats(desc, 'Specimen Type')
    export_csv(desc, out_dir, 'descriptive_by_type.csv')
    
    run_normality(primary, 'specimen_type')
    anova = run_anova(primary, 'specimen_type', 'Specimen Type')
    run_anova(primary, 'position', 'Position')
    
    create_boxplots(primary, out_dir)
    create_summary(primary, out_dir)
    print(f'\n  All outputs saved to: {out_dir}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_file')
    parser.add_argument('-o', '--output', default='stats_output')
    args = parser.parse_args()
    run_full_analysis(args.csv_file, args.output)
