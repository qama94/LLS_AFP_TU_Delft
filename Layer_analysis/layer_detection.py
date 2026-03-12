#!/usr/bin/env python3
"""
AFP Layer Detection from LLS Cross-Section Profiles
Author:  Gamar Ismayilova
Affiliation: TU Delft, Dept. of Aerospace Structures and Materials
License: MIT

See README.md for full methodology and usage.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from scipy.signal import medfilt, find_peaks, savgol_filter
import re, os, argparse, glob

# ── Parameters ────────────────────────────────────────────────────────
SPIKE_KERNEL    = 51
MEDIAN_KERNEL   = 11
DETREND_FRAC    = 0.08
HIST_BINS       = 40
LINE_B_FLOOR    = 0.12
MIN_STEP_RATIO  = 0.50
SG_WIN          = 81
SG_POLY         = 3
SLOPE_EDGE      = 0.12
TOL_FRAC        = 0.08
MAX_PEAK_TRIES  = 10
MIN_PEAK_DZ     = 0.015
SEARCH_MARGIN   = 0.0
BOUNDARY_PAD    = 0.20
OUTER_ZONE_FRAC = 0.30
OUTER_ZONE_MAX  = 5.0
SYM_MARGIN      = 2.0
MERGE_RATIO     = 0.70
SUPP_PROM_RATIO = 0.20
SUPP_STEP_RATIO = 0.45

# ── Visual styling ────────────────────────────────────────────────────
LINE_COLORS  = ['#CC2222', '#1166CC', '#228833', '#AA6600']
LINE_LETTERS = ['B', 'C', 'D', 'E']
LAYER_COLORS = ['#CC2222', '#228833', '#7733BB', '#1166CC']
LAYER_NAMES  = ['Layer 1', 'Layer 2', 'Layer 3', 'Layer 4']
SUB_COLOR    = '#999999'

LABEL_MAP = {
    (0,'left','foot'): 'C1',  (0,'left','shoulder'): 'C2',
    (0,'right','shoulder'): 'C3', (0,'right','foot'): 'C4',
    (1,'left','foot'): 'D2',  (1,'left','shoulder'): 'D2_top',
    (1,'right','shoulder'): 'D3_top', (1,'right','foot'): 'D3',
    (2,'left','foot'): 'E2',  (2,'left','shoulder'): 'E2_top',
    (2,'right','shoulder'): 'E3_top', (2,'right','foot'): 'E3',
}


# ── Utilities ─────────────────────────────────────────────────────────

def format_sample_name(filepath):
    """Derive sample name from filename: Sample_2_1A_00010.slk → Sample 2_1A"""
    base = os.path.basename(filepath)
    name = re.sub(r'_\d{4,5}\.slk$', '', base)
    if name == base:
        name = base.replace('.slk', '')
    return re.sub(r'^Sample_', 'Sample ', name)


def versioned_path(base):
    """Avoid overwriting: append _v2, _v3, etc. if file exists."""
    if not os.path.exists(base):
        return base
    root, ext = os.path.splitext(base)
    v = 2
    while True:
        candidate = f"{root}_v{v}{ext}"
        if not os.path.exists(candidate):
            return candidate
        v += 1


def parse_slk(filepath):
    """Parse SYLK (.slk) file → (x, z) arrays in mm."""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    lines = text.replace('\r\n', '\n').replace('\r', '\n').split('\n')
    data = {}
    cur_row, cur_col = 1, 1
    for line in lines:
        line = line.strip()
        if line.startswith('F;'):
            my = re.search(r'Y(\d+)', line)
            mx = re.search(r'X(\d+)', line)
            if my: cur_row = int(my.group(1))
            if mx: cur_col = int(mx.group(1))
        elif line.startswith('C;') and 'K' in line:
            my = re.search(r'Y(\d+)', line)
            mx = re.search(r'X(\d+)', line)
            if my: cur_row = int(my.group(1))
            if mx: cur_col = int(mx.group(1))
            ki = line.index('K')
            try:
                data[(cur_row, cur_col)] = float(line[ki+1:].split(';')[0])
            except ValueError:
                pass
    mr = max(r for r, _ in data)
    xs, zs = [], []
    for row in range(1, mr + 1):
        if (row, 1) in data and (row, 2) in data:
            xs.append(data[(row, 1)])
            zs.append(data[(row, 2)])
    return np.array(xs), np.array(zs)


def preprocess(x_raw, z_raw):
    """Spike removal, median filter, linear detrend."""
    x = x_raw[::-1] - x_raw[::-1].min()
    z_raw = z_raw[::-1]
    n = len(x)

    # Spike removal
    zl = medfilt(z_raw, SPIKE_KERNEL)
    res = z_raw - zl
    sig = np.std(res)
    mask = np.abs(res) > 5 * sig
    zd = z_raw.copy()
    zd[mask] = zl[mask]

    # Median filter
    zf = medfilt(zd, MEDIAN_KERNEL)

    # Linear detrend from outer flat regions
    en = max(30, int(n * DETREND_FRAC))
    xs2 = np.concatenate([x[:en], x[-en:]])
    zs2 = np.concatenate([zf[:en], zf[-en:]])
    sl, it = np.polyfit(xs2, zs2, 1)
    return x, zf - (sl * x + it)


# ── Level detection ───────────────────────────────────────────────────

def get_levels(z):
    """Detect layer plateau heights from Z histogram (two-stage)."""
    z_range = z.max()
    counts, edges = np.histogram(z, bins=HIST_BINS)
    centres = 0.5 * (edges[:-1] + edges[1:])

    # Stage 1: standard histogram peaks
    pks, props = find_peaks(counts, prominence=counts.max() * 0.06, distance=2)
    above = [p for p in pks if centres[p] > z_range * LINE_B_FLOOR]
    if not above:
        return []

    max_p = max(props['prominences'][list(pks).index(p)] for p in above)
    cands = sorted([centres[p] for p in above
                    if props['prominences'][list(pks).index(p)] > max_p * 0.20])

    L1 = cands[0]
    filtered = [L1]
    prev_step = L1
    for lv in cands[1:]:
        cur_step = lv - filtered[-1]
        if cur_step >= MIN_STEP_RATIO * max(L1, prev_step):
            filtered.append(lv)
            prev_step = cur_step

    # Stage 2: supplement weak upper-layer peaks
    top = filtered[-1]
    prev_step = top - (filtered[-2] if len(filtered) > 1 else 0)
    supp_pks, supp_props = find_peaks(counts, prominence=counts.max() * 0.04, distance=2)
    above_top = [(centres[p], supp_props['prominences'][list(supp_pks).index(p)])
                 for p in supp_pks if centres[p] > top]
    for z_s, prom_s in sorted(above_top, key=lambda t: -t[1]):
        prom_ok = prom_s >= counts.max() * SUPP_PROM_RATIO
        step_ok = (z_s - top) >= SUPP_STEP_RATIO * prev_step
        if prom_ok and step_ok:
            filtered.append(z_s)
            break

    return filtered


def get_band_tape_layer(levels):
    """Map each band index to a tape-layer index for colouring."""
    N = len(levels)
    L1 = levels[0]
    threshold = MERGE_RATIO * L1
    mapping = [0]
    layer_count = 0
    for bi in range(1, N):
        prev_height = levels[bi-1] - (levels[bi-2] if bi > 1 else 0)
        if prev_height >= threshold:
            layer_count += 1
        mapping.append(layer_count)
    return mapping


def fit_line(xa, za):
    """Fit linear regression, return (slope, intercept)."""
    if len(xa) < 2:
        return 0.0, float(np.mean(za)) if len(za) else 0.0
    return np.polyfit(xa, za, 1)


# ── Vertex detection ──────────────────────────────────────────────────

def find_step_vertices(x, z, dz, z_lo, z_hi, side, x_lo_bound, x_hi_bound):
    """Find foot (at z_lo) and shoulder (at z_hi) for one tape step.
    Returns (foot, shoulder) as (x,z) tuples or None."""
    n = len(x)
    dx = float(x[1] - x[0])
    tol = (z_hi - z_lo) * TOL_FRAC

    # Search region
    search = ((x >= x_lo_bound) & (x <= x_hi_bound) &
              (z >= z_lo - tol) & (z <= z_hi + tol))
    active = np.where(search)[0]
    if len(active) == 0:
        return None, None

    # Find gradient peaks in search region
    dz_in = dz.copy()
    dz_in[~search] = 0.0
    sign = 1 if side == 'left' else -1
    raw_peaks, _ = find_peaks(sign * dz_in, distance=8, prominence=0)
    if len(raw_peaks) == 0:
        return None, None

    # Filter weak peaks
    strong = np.where(np.abs(dz[raw_peaks]) >= MIN_PEAK_DZ)[0]
    if len(strong) == 0:
        return None, None
    raw_peaks = raw_peaks[strong]

    # Rank by |dz| magnitude (most robust)
    order = np.argsort(np.abs(dz[raw_peaks]))[::-1]
    candidates = raw_peaks[order[:MAX_PEAK_TRIES]]

    hw = max(5, int(0.15 / dx))

    for peak_idx in candidates:
        peak_abs = abs(dz[peak_idx])
        if peak_abs < MIN_PEAK_DZ:
            continue
        thresh = SLOPE_EDGE * peak_abs

        # Expand slope window where |dz| > threshold
        lo = peak_idx
        while lo > 0 and abs(dz[lo-1]) > thresh and search[lo-1]:
            lo -= 1
        hi = peak_idx
        while hi < n-1 and abs(dz[hi+1]) > thresh and search[hi+1]:
            hi += 1
        lo = max(0, min(lo, peak_idx - hw))
        hi = min(n-1, max(hi, peak_idx + hw))

        # Fit slope line
        s_sl, b_sl = fit_line(x[lo:hi+1], z[lo:hi+1])
        if abs(s_sl) < 1e-10:
            continue

        # Intersect slope line with reference levels
        foot_x = (z_lo - b_sl) / s_sl
        shoulder_x = (z_hi - b_sl) / s_sl
        foot = (foot_x, z_lo)
        shoulder = (shoulder_x, z_hi)

        # Validate: intersection must be near the slope window
        x_margin = (x[hi] - x[lo]) * 2.5 + 0.5
        z_margin = (z_hi - z_lo) * 0.60

        def validate(v, z_ref):
            if v is None:
                return None
            vx, vz = v
            if not (x[lo] - x_margin <= vx <= x[hi] + x_margin):
                return None
            if not (z_ref - z_margin <= vz <= z_ref + z_margin):
                return None
            if not (x.min() - 1 <= vx <= x.max() + 1):
                return None
            return v

        foot = validate(foot, z_lo)
        shoulder = validate(shoulder, z_hi)

        if foot is not None or shoulder is not None:
            return foot, shoulder

    return None, None


# ── Main pipeline ─────────────────────────────────────────────────────

def run(filepath, sample_name=None, output_dir=None):
    """Run layer detection on a single .slk file.

    Parameters
    ----------
    filepath : str
        Path to the .slk file.
    sample_name : str, optional
        Display label. Auto-derived from filename if not given.
    output_dir : str, optional
        Where to save the PNG. Default: same folder as input.

    Returns
    -------
    str : path to saved figure.
    """
    # Auto-derive names
    if sample_name is None:
        sample_name = format_sample_name(filepath)
    if output_dir is None:
        output_dir = os.path.dirname(filepath) or '.'
    os.makedirs(output_dir, exist_ok=True)

    # Safe filename from sample name
    safe_name = sample_name.replace(' ', '_')
    out_path = os.path.join(output_dir, f'{safe_name}_layer_detection.png')

    # Load and preprocess
    x_raw, z_raw = parse_slk(filepath)
    x, z = preprocess(x_raw, z_raw)
    n = len(x)
    dx = float(x[1] - x[0])

    # Detect levels
    levels = get_levels(z)
    N = len(levels)
    z_all = [0.0] + levels

    # Band → tape-layer mapping
    band_tape_layer = get_band_tape_layer(levels)
    n_tape_layers = max(band_tape_layer) + 1

    print(f"\n{'='*60}")
    print(f"  {sample_name}  →  {N} band(s)  /  {n_tape_layers} tape layer(s)")
    for i, lv in enumerate(levels):
        print(f"    Line {LINE_LETTERS[i]} = {lv*1000:.1f} µm")
    print(f"{'='*60}")

    # Gradient via Savitzky-Golay
    dz = savgol_filter(z, window_length=SG_WIN, polyorder=SG_POLY,
                       deriv=1, delta=dx)

    # Tape extent
    tape_pts = np.where(z > 0.05 * z.max())[0]
    x_l = x[tape_pts[0]] + SEARCH_MARGIN
    x_r = x[tape_pts[-1]] - SEARCH_MARGIN
    x_mid = (x_l + x_r) / 2.0
    tape_half = (x_r - x_l) / 2.0
    outer_zone = min(OUTER_ZONE_MAX, OUTER_ZONE_FRAC * tape_half)

    band_results = [dict(band=i, z_lo=z_all[i], z_hi=z_all[i+1],
                         lf=None, ls=None, rf=None, rs=None)
                    for i in range(N)]

    # Left side: outer → inner
    left_boundary = x_l
    for bi in range(N):
        z_lo, z_hi = z_all[bi], z_all[bi+1]
        x_lo = x_l if bi == 0 else left_boundary
        x_hi = x_l + outer_zone if bi == 0 else x_mid
        lf, ls = find_step_vertices(x, z, dz, z_lo, z_hi, 'left', x_lo, x_hi)
        band_results[bi].update(lf=lf, ls=ls)
        ref = ls[0] if ls else (lf[0] if lf else left_boundary)
        left_boundary = ref + BOUNDARY_PAD

    # Right side: outer → inner (symmetric window for Band 1+)
    right_boundary = x_r
    for bi in range(N):
        z_lo, z_hi = z_all[bi], z_all[bi+1]
        if bi == 0:
            x_lo_r = x_r - outer_zone
            x_hi_r = x_r
        else:
            lf_x = band_results[bi]['lf'][0] if band_results[bi]['lf'] else x_mid
            sym_center = 2.0 * x_mid - lf_x
            x_lo_r = max(x_mid, sym_center - SYM_MARGIN)
            x_hi_r = min(right_boundary, sym_center + SYM_MARGIN)
        rf, rs = find_step_vertices(x, z, dz, z_lo, z_hi, 'right', x_lo_r, x_hi_r)
        band_results[bi].update(rf=rf, rs=rs)
        ref = rs[0] if rs else (rf[0] if rf else right_boundary)
        right_boundary = ref - BOUNDARY_PAD

    # Print results
    print(f"\n  Vertices:")
    for br in band_results:
        bi = br['band']
        k = band_tape_layer[bi]
        print(f"\n    Band {bi} ({br['z_lo']*1000:.0f}→{br['z_hi']*1000:.0f} µm)"
              f"  tape layer {k+1}")
        for lbl, v in [
            (LABEL_MAP.get((bi, 'left', 'foot'), '?'), br['lf']),
            (LABEL_MAP.get((bi, 'left', 'shoulder'), '?'), br['ls']),
            (LABEL_MAP.get((bi, 'right', 'shoulder'), '?'), br['rs']),
            (LABEL_MAP.get((bi, 'right', 'foot'), '?'), br['rf']),
        ]:
            if v:
                print(f"      {lbl:10s}: x={v[0]:.3f} mm  z={v[1]*1000:.1f} µm")

    print(f"\n  Widths:")
    for br in band_results:
        if br['lf'] and br['rf']:
            W = br['rf'][0] - br['lf'][0]
            seg = z[(x >= br['lf'][0]) & (x <= br['rf'][0])]
            avg = float(np.mean(seg)) if len(seg) > 0 else 0.0
            k = band_tape_layer[br['band']]
            print(f"    {LAYER_NAMES[k]}: W={W:.3f} mm  avg={avg*1000:.1f} µm")

    # ── Build coloured segments ───────────────────────────────────────
    events = sorted(
        [(int(np.argmin(np.abs(x - br['lf'][0]))), +1)
         for br in band_results if br['lf']] +
        [(int(np.argmin(np.abs(x - br['rf'][0]))), -1)
         for br in band_results if br['rf']],
        key=lambda t: t[0])
    seg_idxs = [0] + [e[0] for e in events] + [n-1]
    layer_at = 0
    lay_seq = []
    for _, s in events:
        lay_seq.append(layer_at)
        layer_at = max(0, layer_at + s)
    lay_seq.append(layer_at)

    def seg_color(val):
        if val == 0:
            return SUB_COLOR
        bi = min(val - 1, N - 1)
        tl = band_tape_layer[bi]
        return LAYER_COLORS[min(tl, 3)]

    # ── Plot ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 12), facecolor='white')
    ax = fig.add_axes([0.09, 0.38, 0.88, 0.55])
    axb = fig.add_axes([0.09, 0.06, 0.88, 0.26])
    for a in [ax, axb]:
        a.set_facecolor('white')

    # Coloured profile segments
    for k in range(len(seg_idxs) - 1):
        i0, i1 = seg_idxs[k], seg_idxs[k+1]
        ax.plot(x[i0:i1+1], z[i0:i1+1], color=seg_color(lay_seq[k]),
                lw=2.2, solid_capstyle='round', zorder=2)

    # Reference lines
    ax.axhline(0, color='#cccccc', lw=1.0)
    for i, lv in enumerate(levels):
        ax.axhline(lv, color=LINE_COLORS[i % 4], lw=1.4, ls='--', alpha=0.55)
        ax.text(x.max() + 0.15, lv, f'Line {LINE_LETTERS[i]}',
                va='center', ha='left', fontsize=9,
                color=LINE_COLORS[i % 4], fontweight='bold')

    # Vertex dots and labels
    for br in band_results:
        bi = br['band']
        spot_col = LINE_COLORS[bi % 4]
        for v_type, v, z_ref, is_left in [
            ('foot', br['lf'], br['z_lo'], True),
            ('shoulder', br['ls'], br['z_hi'], True),
            ('shoulder', br['rs'], br['z_hi'], False),
            ('foot', br['rf'], br['z_lo'], False),
        ]:
            if v is None:
                continue
            side_str = 'left' if is_left else 'right'
            lbl = LABEL_MAP.get((bi, side_str, v_type), '')
            if lbl.endswith('_top'):
                continue

            ax.plot(v[0], v[1], 'o', color=spot_col, ms=12, zorder=10,
                    markeredgecolor='white', markeredgewidth=1.5)
            ax.axvline(v[0], color=spot_col, lw=1.2, ls='--', alpha=0.50)
            if lbl:
                ax.text(v[0], v[1] + (-0.055 if z_ref < 0.001 else 0.020),
                        lbl, ha='center', va='bottom', fontsize=10,
                        color='white', zorder=11,
                        bbox=dict(fc=spot_col, ec='none', pad=3,
                                  boxstyle='round,pad=0.3'))

    # Axis formatting
    ax.set_ylabel('Z − Height (mm)', fontsize=12)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))
    ax.tick_params(axis='x', labelbottom=False)
    ax.tick_params(axis='both', which='major', labelsize=13, length=6)
    ax.tick_params(axis='both', which='minor', length=3)
    ax.grid(which='major', alpha=0.20)
    ax.grid(which='minor', alpha=0.08)
    ax.set_xlim(x.min() - 0.3, x.max() + 0.6)
    ax.set_ylim(z.min() - 0.08, z.max() + 0.15)
    ax.set_title(f'{sample_name} — Layer Detection  '
                 f'({n_tape_layers} tape layer{"s" if n_tape_layers > 1 else ""})',
                 fontsize=13, fontweight='bold', pad=8)
    for sp in ax.spines.values():
        sp.set_color('#cccccc')

    # Legend
    leg = [Line2D([0], [0], color=SUB_COLOR, lw=3, label='Substrate')]
    for tl in range(n_tape_layers):
        leg.append(Line2D([0], [0], color=LAYER_COLORS[tl], lw=3,
                          label=LAYER_NAMES[tl]))
    for i, lv in enumerate(levels):
        leg.append(Line2D([0], [0], color=LINE_COLORS[i % 4], lw=1.4, ls='--',
                          label=f'Line {LINE_LETTERS[i]} = {lv*1000:.1f} µm'))
    leg.append(Line2D([0], [0], color='#cccccc', lw=1.0, label='Baseline (z=0)'))
    ax.legend(handles=leg, loc='upper right', fontsize=9,
              facecolor='white', edgecolor='#cccccc', framealpha=0.95, ncol=2)

    # Width brackets (bottom panel)
    axb.set_xlim(ax.get_xlim())
    axb.set_ylim(0, 1)
    axb.axis('off')

    def bracket(x0, x1, y, color, label):
        axb.annotate('', xy=(x1, y), xytext=(x0, y),
                     arrowprops=dict(arrowstyle='<->', color=color,
                                     lw=2.0, mutation_scale=12))
        for xp in [x0, x1]:
            axb.plot([xp, xp], [y - 0.07, y + 0.07], color=color, lw=1.8)
        axb.text((x0 + x1) / 2, y + 0.10, label,
                 ha='center', va='bottom', fontsize=10, color='white',
                 bbox=dict(fc=color, ec='none', pad=3.5,
                           boxstyle='round,pad=0.4'))

    y_br = 0.80
    seen_layers = set()
    for br in band_results:
        if not (br['lf'] and br['rf']):
            continue
        tl = band_tape_layer[br['band']]
        if tl in seen_layers:
            continue
        W = br['rf'][0] - br['lf'][0]
        seg = z[(x >= br['lf'][0]) & (x <= br['rf'][0])]
        avg = float(np.mean(seg)) if len(seg) > 0 else 0.0
        bracket(br['lf'][0], br['rf'][0], y_br,
                LAYER_COLORS[tl % 4],
                f"{LAYER_NAMES[tl]}   W = {W:.2f} mm  |  avg = {avg*1000:.1f} µm")
        y_br -= 0.36
        seen_layers.add(tl)

    axb.set_xlabel('Width (mm)', fontsize=12)
    axb.xaxis.set_major_locator(ticker.MultipleLocator(1))
    axb.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    axb.tick_params(axis='x', which='major', labelsize=13, length=6)
    axb.tick_params(axis='x', which='minor', length=3)
    axb.set_xlim(ax.get_xlim())

    # Save
    out = versioned_path(out_path)
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n  Saved: {out}")

    # Build results dict
    layer_widths = {}
    layer_heights = {}
    for br in band_results:
        if br['lf'] and br['rf']:
            k = band_tape_layer[br['band']]
            W = br['rf'][0] - br['lf'][0]
            seg = z[(x >= br['lf'][0]) & (x <= br['rf'][0])]
            avg = float(np.mean(seg)) if len(seg) > 0 else 0.0
            layer_widths[LAYER_NAMES[k]] = W
            layer_heights[LAYER_NAMES[k]] = avg

    return {
        'sample': sample_name,
        'n_bands': N,
        'n_tape_layers': n_tape_layers,
        'levels': levels,
        'band_results': band_results,
        'band_tape_layer': band_tape_layer,
        'layer_widths': layer_widths,
        'layer_heights': layer_heights,
        'image_path': out,
    }


# ── Run ───────────────────────────────────────────────────────────────

if __name__ == '__main__':

    def display_result(image_path):
        """Show the result image — works in Colab, Jupyter, or terminal."""
        try:
            from IPython.display import display, Image
            display(Image(filename=image_path))
            return
        except ImportError:
            pass
        print(f"  Output saved: {image_path}")

    def upload_files():
        """Upload .slk files — works on Colab, Jupyter, or terminal."""
        try:
            from google.colab import files
            print("Upload your .slk file(s):")
            uploaded = files.upload()
            return list(uploaded.keys())
        except ImportError:
            pass

        try:
            import ipywidgets
            from IPython.display import display
            uploader = ipywidgets.FileUpload(accept='.slk', multiple=True)
            display(uploader)
            print("Click the upload button above, then re-run this cell.")
            return []
        except ImportError:
            pass

        import sys
        if len(sys.argv) > 1:
            return sys.argv[1:]

        path = input("Enter .slk file path (or glob pattern like data/*.slk): ").strip()
        matches = glob.glob(path)
        return matches if matches else [path]

    slk_files = upload_files()
    if slk_files:
        for fpath in slk_files:
            result = run(fpath, output_dir="results")
            display_result(result['image_path'])
