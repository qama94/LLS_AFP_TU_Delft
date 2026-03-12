"""
AFP LLS — Layer Detection
==========================
Detects layer count, reference lines, and geometric vertices
(feet and shoulders) for any N-layer AFP tape staircase profile
from SYLK (.slk) files.

Algorithm:
  1. Parse + preprocess SYLK  →  x, z arrays
  2. Histogram peaks + high-z supplement  →  N layer plateau heights
  3. Band-to-tape-layer mapping  →  corrects spurious intermediate levels
     (e.g. when two histogram levels together represent one physical tape)
  4. For each band (z_lo → z_hi), find slope on LEFT and RIGHT:
       • Band 0 (outermost tape edge): outer zone only, peaks ranked by
         |dz| magnitude (most robust against boundary-proximity artefacts)
       • Band 1+: search window bounded by previous band's shoulder.
         Peaks also ranked by |dz| for the same robustness reason.
  5. Foot   = slope line @ z_lo  (direct reference-level intersection)
     Shoulder = slope line @ z_hi
  6. Widths and average heights from foot-to-foot distances.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from scipy.signal import medfilt, find_peaks, savgol_filter
import re, os

# ══════════════════════════════════════════════════════════════════════
# PARAMETERS
# ══════════════════════════════════════════════════════════════════════
SPIKE_KERNEL    = 51
MEDIAN_KERNEL   = 11
DETREND_FRAC    = 0.08
HIST_BINS       = 40
LINE_B_FLOOR    = 0.12      # levels must be above 12 % of z_range (captures low first layers)
MIN_STEP_RATIO  = 0.50      # each new step ≥ 50 % of max(L1, prev_step)
SG_WIN          = 81
SG_POLY         = 3
SLOPE_EDGE      = 0.12
TOL_FRAC        = 0.08
MAX_PEAK_TRIES  = 10
MIN_PEAK_DZ     = 0.015
SEARCH_MARGIN   = 0.0
BOUNDARY_PAD    = 0.20
OUTER_ZONE_FRAC = 0.30      # Band 0 search = this fraction of tape half-width
OUTER_ZONE_MAX  = 5.0       # mm: absolute cap on outer zone
SYM_MARGIN      = 2.0       # mm: Band 1+ right search restricted to ±SYM_MARGIN of
                            #     the symmetric position (2*x_mid − left_foot_x).
                            #     Prevents bump inner-slopes from being mistaken for
                            #     the correct outer edge.
# Tape-layer color grouping: if a band's height (z_hi - z_lo) is less than
# MERGE_RATIO * L1, it does NOT count as a new tape layer for coloring.
MERGE_RATIO     = 0.70
# Stage-2 supplement parameters (see get_levels docstring)
SUPP_PROM_RATIO = 0.20      # peak prom ≥ this fraction of histogram max count
SUPP_STEP_RATIO = 0.45      # new step ≥ this fraction of the previous inter-level step

LINE_COLORS  = ['#CC2222', '#1166CC', '#228833', '#AA6600']
LINE_LETTERS = ['B', 'C', 'D', 'E']
LAYER_COLORS = ['#CC2222', '#228833', '#7733BB', '#1166CC']
LAYER_NAMES  = ['Layer 1', 'Layer 2', 'Layer 3', 'Layer 4']
SUB_COLOR    = '#999999'

LABEL_MAP = {
    (0,'left', 'foot'):      'C1',     (0,'left', 'shoulder'):  'C2',
    (0,'right','shoulder'):  'C3',     (0,'right','foot'):      'C4',
    (1,'left', 'foot'):      'D2',     (1,'left', 'shoulder'):  'D2_top',
    (1,'right','shoulder'):  'D3_top', (1,'right','foot'):      'D3',
    (2,'left', 'foot'):      'E2',     (2,'left', 'shoulder'):  'E2_top',
    (2,'right','shoulder'):  'E3_top', (2,'right','foot'):      'E3',
}


# ══════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════
def versioned_path(base):
    if not os.path.exists(base): return base
    root, ext = os.path.splitext(base); v = 2
    while True:
        c = f"{root}_v{v}{ext}"
        if not os.path.exists(c): return c
        v += 1

def parse_slk(filepath):
    with open(filepath,'r',encoding='utf-8',errors='ignore') as f:
        text = f.read()
    lines = text.replace('\r\n','\n').replace('\r','\n').split('\n')
    data = {}; cur_row, cur_col = 1, 1
    for line in lines:
        line = line.strip()
        if line.startswith('F;'):
            my=re.search(r'Y(\d+)',line); mx=re.search(r'X(\d+)',line)
            if my: cur_row=int(my.group(1))
            if mx: cur_col=int(mx.group(1))
        elif line.startswith('C;') and 'K' in line:
            my=re.search(r'Y(\d+)',line); mx=re.search(r'X(\d+)',line)
            if my: cur_row=int(my.group(1))
            if mx: cur_col=int(mx.group(1))
            ki = line.index('K')
            try: data[(cur_row,cur_col)]=float(line[ki+1:].split(';')[0])
            except ValueError: pass
    mr = max(r for r,_ in data); xs, zs = [], []
    for row in range(1, mr+1):
        if (row,1) in data and (row,2) in data:
            xs.append(data[(row,1)]); zs.append(data[(row,2)])
    return np.array(xs), np.array(zs)

def preprocess(x_raw, z_raw):
    x = x_raw[::-1]-x_raw[::-1].min(); z_raw=z_raw[::-1]; n=len(x)
    zl = medfilt(z_raw,SPIKE_KERNEL); res=z_raw-zl; sig=np.std(res)
    mask = np.abs(res)>5*sig; zd=z_raw.copy(); zd[mask]=zl[mask]
    zf = medfilt(zd,MEDIAN_KERNEL)
    en = max(30,int(n*DETREND_FRAC))
    xs2=np.concatenate([x[:en],x[-en:]]); zs2=np.concatenate([zf[:en],zf[-en:]])
    sl,it=np.polyfit(xs2,zs2,1)
    return x, zf-(sl*x+it)

def get_levels(z):
    """
    Detect layer plateau heights from histogram.

    Two-stage approach:
      Stage 1 (standard): find histogram peaks above LINE_B_FLOOR with
        prominence ≥ 6 % of max, then apply step-size filter.
      Stage 2 (supplement): scan for weaker peaks ABOVE the highest
        detected level that were rejected only by the step-size filter.
        A supplemental peak is accepted when:
          (a) its histogram prominence ≥ SUPP_PROM_RATIO × counts.max()
              (ensures it is a real plateau, not just histogram noise), AND
          (b) its step from the top level is ≥ SUPP_STEP_RATIO × the
              step that produced the top level (geometric consistency).
        This combination is calibrated to fire exclusively for samples
        where a dome-shaped innermost layer creates no flat histogram
        peak yet does have a meaningful secondary plateau band.
    """
    z_range = z.max()
    counts, edges = np.histogram(z, bins=HIST_BINS)
    centres = 0.5*(edges[:-1]+edges[1:])

    # ── Stage 1 ───────────────────────────────────────────────────────
    pks, props = find_peaks(counts, prominence=counts.max()*0.06, distance=2)
    above = [p for p in pks if centres[p] > z_range*LINE_B_FLOOR]
    if not above: return []
    max_p = max(props['prominences'][list(pks).index(p)] for p in above)
    cands = sorted([centres[p] for p in above
                    if props['prominences'][list(pks).index(p)] > max_p*0.20])
    L1 = cands[0]; filtered = [L1]
    # Compare each candidate step to MIN_STEP_RATIO × max(L1, prev_step).
    # Using max() guards against two failure modes:
    #   • Small L1 (e.g. 61 µm in 3_3B) making the threshold too loose → spurious
    #     bump levels would slip through a pure L1-based test.
    #   • A sudden large drop in step size (e.g. 2_2B 76→54 µm) making a
    #     pure prev-step-based test too loose.
    prev_step = L1
    for lv in cands[1:]:
        cur_step = lv - filtered[-1]
        if cur_step >= MIN_STEP_RATIO * max(L1, prev_step):
            filtered.append(lv)
            prev_step = cur_step

    # ── Stage 2 (supplement) ──────────────────────────────────────────
    # Scan for any histogram peak strictly above the highest detected
    # level. Accept if BOTH prominence and geometric-step criteria pass.
    top      = filtered[-1]
    prev_step = top - (filtered[-2] if len(filtered) > 1 else 0)
    supp_pks, supp_props = find_peaks(counts,
                                       prominence=counts.max()*0.04,
                                       distance=2)
    above_top = [(centres[p], supp_props['prominences'][list(supp_pks).index(p)])
                 for p in supp_pks if centres[p] > top]
    # Sort by descending prominence; take the first that passes both gates
    for z_s, prom_s in sorted(above_top, key=lambda t: -t[1]):
        prom_ok = prom_s >= counts.max() * SUPP_PROM_RATIO
        step_ok = (z_s - top) >= SUPP_STEP_RATIO * prev_step
        if prom_ok and step_ok:
            filtered.append(z_s)
            break

    return filtered

def get_band_tape_layer(levels):
    """
    Map each band index to a tape-layer index for coloring.
    A band increments the tape-layer counter only if the PREVIOUS
    band's height was at least MERGE_RATIO * L1. Otherwise the band
    is considered a spurious sub-level of the previous tape layer.

    Example for 2_2B (L1=141.7µm, MERGE_RATIO=0.70 → threshold=99.2µm):
      Band 0: height=141.7  ≥ 99.2  prev_band_height N/A → tape_layer=0 (red)
      Band 1: height=76.2   < 99.2  prev_band_height=141.7 ≥ 99.2 → tape_layer=1 (green)
      Band 2: height=141.6  ≥ 99.2  prev_band_height=76.2  < 99.2 → tape_layer=1 (green, same!)
    """
    N  = len(levels)
    L1 = levels[0]
    threshold = MERGE_RATIO * L1
    mapping = [0]   # Band 0 always = tape layer 0
    layer_count = 0
    for bi in range(1, N):
        prev_height = levels[bi-1] - (levels[bi-2] if bi > 1 else 0)
        if prev_height >= threshold:
            layer_count += 1
        mapping.append(layer_count)
    return mapping

def fit_line(xa, za):
    if len(xa)<2: return 0.0, float(np.mean(za)) if len(za) else 0.0
    return np.polyfit(xa, za, 1)


# ══════════════════════════════════════════════════════════════════════
# CORE: FIND VERTICES FOR ONE STEP
# ══════════════════════════════════════════════════════════════════════
def find_step_vertices(x, z, dz, z_lo, z_hi, side, x_lo_bound, x_hi_bound):
    """
    Find FOOT (at z_lo) and SHOULDER (at z_hi) for one tape step.

    Candidates are ranked by |dz| magnitude (strongest first).
    This is more robust than prominence because proximity to the search-window
    boundary artificially inflates prominence (zeroed-out points outside the
    window act as a high baseline).

    For Band 0 (outer zone), the real tape-edge slope consistently has the
    largest |dz|.

    For Band 1+ the correct search window is pre-restricted by the caller to a
    symmetric region around 2*x_mid − left_foot_x, so that inner bump-slopes
    (which can have larger |dz|) lie outside the window and cannot be picked.

    For each candidate:
      • Slope window expanded where |dz| > SLOPE_EDGE × peak |dz|.
      • Slope line fitted through slope window.
      • Foot     = slope line intersected with z_lo (known level).
      • Shoulder = slope line intersected with z_hi (known level).
      • Validated: intersection within generous bounds of slope window.
    """
    n  = len(x); dx = float(x[1]-x[0])
    tol = (z_hi-z_lo)*TOL_FRAC

    search = ((x>=x_lo_bound)&(x<=x_hi_bound)
              &(z>=z_lo-tol)&(z<=z_hi+tol))
    active = np.where(search)[0]
    if len(active)==0: return None, None

    dz_in = dz.copy(); dz_in[~search] = 0.0
    sign  = 1 if side=='left' else -1
    raw_peaks, _ = find_peaks(sign*dz_in, distance=8, prominence=0)
    if len(raw_peaks)==0: return None, None

    strong = np.where(np.abs(dz[raw_peaks]) >= MIN_PEAK_DZ)[0]
    if len(strong)==0: return None, None
    raw_peaks = raw_peaks[strong]

    # ── Rank by |dz| magnitude ───────────────────────────────────────
    # More robust than prominence (boundary effects inflate prominence).
    # For Band 1+ the caller has already restricted x_lo_bound/x_hi_bound to
    # a symmetric window, so inner bump-slopes lie outside and cannot be picked.
    order     = np.argsort(np.abs(dz[raw_peaks]))[::-1]
    candidates = raw_peaks[order[:MAX_PEAK_TRIES]]

    hw = max(5, int(0.15/dx))

    for peak_idx in candidates:
        peak_abs = abs(dz[peak_idx])
        if peak_abs < MIN_PEAK_DZ: continue
        thresh = SLOPE_EDGE * peak_abs

        lo = peak_idx
        while lo>0 and abs(dz[lo-1])>thresh and search[lo-1]: lo -= 1
        hi = peak_idx
        while hi<n-1 and abs(dz[hi+1])>thresh and search[hi+1]: hi += 1
        lo = max(0,   min(lo, peak_idx-hw))
        hi = min(n-1, max(hi, peak_idx+hw))

        s_sl, b_sl = fit_line(x[lo:hi+1], z[lo:hi+1])
        if abs(s_sl)<1e-10: continue

        foot_x     = (z_lo-b_sl)/s_sl
        shoulder_x = (z_hi-b_sl)/s_sl
        foot       = (foot_x,     z_lo)
        shoulder   = (shoulder_x, z_hi)

        x_margin = (x[hi]-x[lo])*2.5 + 0.5
        z_margin = (z_hi-z_lo)*0.60

        def validate(v, z_ref):
            if v is None: return None
            vx, vz = v
            if not (x[lo]-x_margin <= vx <= x[hi]+x_margin): return None
            if not (z_ref-z_margin <= vz <= z_ref+z_margin): return None
            if not (x.min()-1 <= vx <= x.max()+1): return None
            return v

        foot     = validate(foot,     z_lo)
        shoulder = validate(shoulder, z_hi)

        if foot is not None or shoulder is not None:
            return foot, shoulder

    return None, None


# ══════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════
def run(filepath, sample_name, out_base):
    x_raw, z_raw = parse_slk(filepath)
    x, z = preprocess(x_raw, z_raw)
    n = len(x); dx = float(x[1]-x[0])

    levels = get_levels(z); N = len(levels)
    z_all  = [0.0]+levels

    # Band → tape-layer mapping (handles spurious intermediate levels)
    band_tape_layer = get_band_tape_layer(levels)
    n_tape_layers   = max(band_tape_layer)+1

    print(f"\n{'='*60}")
    print(f"{sample_name}  →  {N} band(s)  /  {n_tape_layers} tape layer(s)")
    for i,lv in enumerate(levels):
        print(f"  Line {LINE_LETTERS[i]} = {lv*1000:.1f} µm")
    if N != n_tape_layers:
        print(f"  [band→layer: {band_tape_layer}]")

    dz = savgol_filter(z, window_length=SG_WIN, polyorder=SG_POLY,
                       deriv=1, delta=dx)

    tape_pts = np.where(z>0.05*z.max())[0]
    x_l   = x[tape_pts[0]]  + SEARCH_MARGIN
    x_r   = x[tape_pts[-1]] - SEARCH_MARGIN
    x_mid = (x_l+x_r)/2.0

    tape_half  = (x_r-x_l)/2.0
    outer_zone = min(OUTER_ZONE_MAX, OUTER_ZONE_FRAC*tape_half)

    band_results = [dict(band=i, z_lo=z_all[i], z_hi=z_all[i+1],
                         lf=None, ls=None, rf=None, rs=None)
                    for i in range(N)]

    # ── LEFT SIDE: outer → inner, |dz| ranking ───────────────────────
    left_boundary = x_l
    for bi in range(N):
        z_lo, z_hi = z_all[bi], z_all[bi+1]
        x_lo = x_l           if bi==0 else left_boundary
        x_hi = x_l+outer_zone if bi==0 else x_mid

        lf, ls = find_step_vertices(x, z, dz, z_lo, z_hi, 'left', x_lo, x_hi)
        band_results[bi].update(lf=lf, ls=ls)
        ref = ls[0] if ls else (lf[0] if lf else left_boundary)
        left_boundary = ref + BOUNDARY_PAD

    # ── RIGHT SIDE: outer → inner ─────────────────────────────────────
    # Band 0: outer zone only, |dz| ranking (same as left).
    # Band 1+: restrict window to a symmetric region about 2*x_mid − left_foot_x.
    #   Physical justification: each band's tape is approximately symmetric about
    #   the scan centerline. By anchoring the right search on the mirror of the
    #   (correctly found) left foot, bumps whose inner slopes lie inward of the
    #   symmetric position are excluded from the window. |dz| ranking then
    #   reliably selects the real outer edge slope.
    right_boundary = x_r
    for bi in range(N):
        z_lo, z_hi = z_all[bi], z_all[bi+1]

        if bi == 0:
            x_lo_r = x_r - outer_zone
            x_hi_r = x_r
        else:
            # Symmetric window: center on mirror of left foot
            lf_x = band_results[bi]['lf'][0] if band_results[bi]['lf'] else x_mid
            sym_center = 2.0*x_mid - lf_x
            x_lo_r = max(x_mid,            sym_center - SYM_MARGIN)
            x_hi_r = min(right_boundary,   sym_center + SYM_MARGIN)

        rf, rs = find_step_vertices(x, z, dz, z_lo, z_hi, 'right', x_lo_r, x_hi_r)
        band_results[bi].update(rf=rf, rs=rs)
        ref = rs[0] if rs else (rf[0] if rf else right_boundary)
        right_boundary = ref - BOUNDARY_PAD

    # ── Print results ─────────────────────────────────────────────────
    print(f"\n  Vertices:")
    for br in band_results:
        bi = br['band']; z_lo, z_hi = br['z_lo'], br['z_hi']
        print(f"\n  Band {bi} [{LINE_LETTERS[bi-1] if bi>0 else'0'}→{LINE_LETTERS[bi]}]"
              f"  ({z_lo*1000:.0f}→{z_hi*1000:.0f}µm)  tape layer {band_tape_layer[bi]+1}")
        for lbl, v in [
            (LABEL_MAP.get((bi,'left', 'foot'),    '?'), br['lf']),
            (LABEL_MAP.get((bi,'left','shoulder'),  '?'), br['ls']),
            (LABEL_MAP.get((bi,'right','shoulder'), '?'), br['rs']),
            (LABEL_MAP.get((bi,'right','foot'),     '?'), br['rf']),
        ]:
            if v: print(f"    {lbl:10s}: x={v[0]:.3f} mm  z={v[1]*1000:.1f}µm")
            else: print(f"    {lbl:10s}: not found")

    print(f"\n  Widths:")
    for br in band_results:
        if br['lf'] and br['rf']:
            W   = br['rf'][0]-br['lf'][0]
            seg = z[(x>=br['lf'][0])&(x<=br['rf'][0])]
            avg = float(np.mean(seg)) if len(seg)>0 else 0.0
            k   = band_tape_layer[br['band']]
            print(f"    {LAYER_NAMES[k]}: W={W:.3f} mm  avg={avg*1000:.1f}µm")

    # ── Coloured segments ─────────────────────────────────────────────
    events = sorted(
        [(int(np.argmin(np.abs(x-br['lf'][0]))), +1) for br in band_results if br['lf']] +
        [(int(np.argmin(np.abs(x-br['rf'][0]))), -1) for br in band_results if br['rf']],
        key=lambda t: t[0])
    seg_idxs = [0]+[e[0] for e in events]+[n-1]
    layer_at = 0; lay_seq = []
    for _,s in events:
        lay_seq.append(layer_at); layer_at = max(0,layer_at+s)
    lay_seq.append(layer_at)

    # Map segment's "band count" (layer_at) to tape_layer index for colouring
    # layer_at = how many band-feet we've passed through from the left.
    # We need: the band whose Band index = layer_at - 1 gives the tape layer.
    def seg_color(layer_at_val):
        if layer_at_val == 0:
            return SUB_COLOR
        bi = layer_at_val - 1   # band index of the innermost active band
        bi = min(bi, N-1)
        tl = band_tape_layer[bi]
        return LAYER_COLORS[min(tl, 3)]

    # ── Plot ──────────────────────────────────────────────────────────
    BG = 'white'
    fig = plt.figure(figsize=(18,12), facecolor=BG)
    ax  = fig.add_axes([0.09, 0.38, 0.88, 0.55])
    axb = fig.add_axes([0.09, 0.06, 0.88, 0.26])
    for a in [ax,axb]: a.set_facecolor(BG)

    for k in range(len(seg_idxs)-1):
        i0, i1 = seg_idxs[k], seg_idxs[k+1]
        col = seg_color(lay_seq[k])
        ax.plot(x[i0:i1+1], z[i0:i1+1], color=col, lw=2.2,
                solid_capstyle='round', zorder=2)

    ax.axhline(0, color='#cccccc', lw=1.0)
    for i,lv in enumerate(levels):
        ax.axhline(lv, color=LINE_COLORS[i%4], lw=1.4, ls='--', alpha=0.55)
        ax.text(x.max()+0.15, lv, f'Line {LINE_LETTERS[i]}',
                va='center', ha='left', fontsize=9,
                color=LINE_COLORS[i%4], fontweight='bold')

    for br in band_results:
        bi      = br['band']
        tl      = band_tape_layer[bi]
        spot_col = LINE_COLORS[bi%4]
        for v_type, v, z_ref, is_left in [
            ('foot',     br['lf'], br['z_lo'], True),
            ('shoulder', br['ls'], br['z_hi'], True),
            ('shoulder', br['rs'], br['z_hi'], False),
            ('foot',     br['rf'], br['z_lo'], False),
        ]:
            if v is None: continue
            side_str = 'left' if is_left else 'right'
            lbl = LABEL_MAP.get((bi,side_str,v_type),'')
            if lbl.endswith('_top'): continue

            ax.plot(v[0], v[1], 'o', color=spot_col, ms=12, zorder=10,
                    markeredgecolor='white', markeredgewidth=1.5)
            ax.axvline(v[0], color=spot_col, lw=1.2, ls='--', alpha=0.50)
            if lbl:
                ax.text(v[0], v[1]+(-0.055 if z_ref<0.001 else 0.020),
                        lbl, ha='center', va='bottom', fontsize=10,
                        color='white', zorder=11,
                        bbox=dict(fc=spot_col, ec='none', pad=3,
                                  boxstyle='round,pad=0.3'))

    ax.set_ylabel('Z − Height (mm)', fontsize=12)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))
    ax.tick_params(axis='x', labelbottom=False)
    ax.tick_params(axis='both', which='major', labelsize=13, length=6)
    ax.tick_params(axis='both', which='minor', length=3)
    ax.grid(which='major', alpha=0.20); ax.grid(which='minor', alpha=0.08)
    ax.set_xlim(x.min()-0.3, x.max()+0.6)
    ax.set_ylim(z.min()-0.08, z.max()+0.15)
    ax.set_title(f'{sample_name} — Layer Detection  '
                 f'({n_tape_layers} tape layer{"s" if n_tape_layers>1 else ""})',
                 fontsize=13, fontweight='bold', pad=8)
    for sp in ax.spines.values(): sp.set_color('#cccccc')

    # Legend uses tape-layer colours, not band colours
    leg = [Line2D([0],[0], color=SUB_COLOR, lw=3, label='Substrate')]
    for tl in range(n_tape_layers):
        leg.append(Line2D([0],[0], color=LAYER_COLORS[tl], lw=3,
                          label=LAYER_NAMES[tl]))
    for i,lv in enumerate(levels):
        leg.append(Line2D([0],[0], color=LINE_COLORS[i%4], lw=1.4, ls='--',
                          label=f'Line {LINE_LETTERS[i]} = {lv*1000:.1f} µm'))
    leg.append(Line2D([0],[0], color='#cccccc', lw=1.0, label='Baseline (z=0)'))
    ax.legend(handles=leg, loc='upper right', fontsize=9,
              facecolor='white', edgecolor='#cccccc', framealpha=0.95, ncol=2)

    axb.set_xlim(ax.get_xlim()); axb.set_ylim(0,1); axb.axis('off')

    def bracket(x0, x1, y, color, label):
        axb.annotate('', xy=(x1,y), xytext=(x0,y),
                     arrowprops=dict(arrowstyle='<->',color=color,
                                     lw=2.0,mutation_scale=12))
        for xp in [x0,x1]:
            axb.plot([xp,xp],[y-0.07,y+0.07], color=color, lw=1.8)
        axb.text((x0+x1)/2, y+0.10, label,
                 ha='center', va='bottom', fontsize=10, color='white',
                 bbox=dict(fc=color, ec='none', pad=3.5, boxstyle='round,pad=0.4'))

    y_br = 0.80; seen_layers = set()
    for br in band_results:
        if not (br['lf'] and br['rf']): continue
        tl = band_tape_layer[br['band']]
        if tl in seen_layers: continue        # don't double-bracket same tape layer
        # find the outermost (lf) and innermost band for this tape layer
        # — use the current band's foot only
        W   = br['rf'][0]-br['lf'][0]
        seg = z[(x>=br['lf'][0])&(x<=br['rf'][0])]
        avg = float(np.mean(seg)) if len(seg)>0 else 0.0
        bracket(br['lf'][0], br['rf'][0], y_br,
                LAYER_COLORS[tl%4],
                f"{LAYER_NAMES[tl]}   W = {W:.2f} mm  |  avg = {avg*1000:.1f} µm")
        y_br -= 0.36; seen_layers.add(tl)

    axb.set_xlabel('Width (mm)', fontsize=12)
    axb.xaxis.set_major_locator(ticker.MultipleLocator(1))
    axb.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    axb.tick_params(axis='x', which='major', labelsize=13, length=6)
    axb.tick_params(axis='x', which='minor', length=3)
    axb.set_xlim(ax.get_xlim())

    out = versioned_path(out_base)
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"\n  → saved: {out}")
    return out


# ══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    samples = [
        ('/mnt/user-data/uploads/Sample_1_3C_00010.slk', 'Sample 1.3C',
         '/mnt/user-data/outputs/Sample_1_3C_layer_detection.png'),
        ('/mnt/user-data/uploads/Sample_2_1A_00001.slk', 'Sample 2_1A',
         '/mnt/user-data/outputs/Sample_2_1A_layer_detection.png'),
        ('/mnt/user-data/uploads/Sample_3_1A_00001.slk', 'Sample 3_1A',
         '/mnt/user-data/outputs/Sample_3_1A_layer_detection.png'),
        ('/mnt/user-data/uploads/Sample_1_1B_00004.slk', 'Sample 1_1B',
         '/mnt/user-data/outputs/Sample_1_1B_layer_detection.png'),
        ('/mnt/user-data/uploads/Sample_2_2B_00004.slk', 'Sample 2_2B',
         '/mnt/user-data/outputs/Sample_2_2B_layer_detection.png'),
        ('/mnt/user-data/uploads/Sample_3_3B_00004.slk', 'Sample 3_3B',
         '/mnt/user-data/outputs/Sample_3_3B_layer_detection.png'),
    ]
    for fp, name, out in samples:
        run(fp, name, out)
