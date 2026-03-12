#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Overlap Bead Peak Analysis for Laser Line Scanner Profiles
===========================================================
Author:  Gamar Ismayilova
Affiliation: TU Delft, Dept. of Aerospace Structures and Materials
Contact: gamar.ismayilova@gmail.com
License: MIT

Pipeline: SYLK loading → spike removal (MAD) → median filter →
          substrate identification → peak detection → geometric
          characterisation → visualisation + report.

See README.md for full mathematical methodology and usage examples.
"""

import os
import re

# ---------------------------------------------------------------------------
# Third-party numerical and scientific computing
# ---------------------------------------------------------------------------
import numpy as np
import matplotlib
matplotlib.use('Agg')                       # Non-interactive backend
import matplotlib.pyplot as plt
from scipy.signal import medfilt, find_peaks
from scipy.ndimage import gaussian_filter1d


# ============================================================================
# MODULE-LEVEL CONSTANTS
# ============================================================================

#: Conversion factor from the Median Absolute Deviation (MAD) to the
#: equivalent standard deviation of a normal distribution. Derived as
#: 1 / Phi^{-1}(3/4) where Phi^{-1} is the quantile function of the
#: standard normal distribution. See Leys et al. (2013), Ref. [3].
MAD_TO_SIGMA = 1.4826


# ============================================================================
# SECTION 1 — DATA INGESTION
# ============================================================================

def load_slk_profile(filepath):
    """Parse a SYLK (.slk) file and return (x, z) arrays in mm."""
    x_values = []
    z_values = []

    with open(filepath, 'r') as fh:
        lines = fh.readlines()

    pending_x = None
    active_col = None

    for line in lines:
        line = line.strip()

        # Format directive: indicates the column of the next cell value.
        if line.startswith('F;Y'):
            match = re.search(r'F;Y(\d+);X(\d+)', line)
            if match:
                active_col = int(match.group(2))

        # Cell value record.
        elif line.startswith('C;K'):
            value = float(line.replace('C;K', ''))
            if active_col == 1:
                pending_x = value
            elif active_col == 2 and pending_x is not None:
                x_values.append(pending_x)
                z_values.append(value)
                pending_x = None

    if len(x_values) == 0:
        raise ValueError(
            f"No valid profile data extracted from '{filepath}'. "
            "Verify that the file conforms to the expected SYLK format."
        )

    return np.array(x_values), np.array(z_values)


# ============================================================================
# SECTION 2 — SIGNAL CONDITIONING
# ============================================================================

def remove_spikes(z_raw, window_size=15, threshold_factor=3.0):
    """
    Detect and replace anomalous spikes in the height signal using a
    robust, locally-adaptive outlier criterion based on the Median
    Absolute Deviation (MAD).

    Methodology
    -----------
    For each sample index *i*, a symmetric neighbourhood of
    ``window_size`` points centred on *i* is extracted. The local
    median m_i and the local MAD are computed within this window:

        MAD_i = median( | z_j - m_i | )    for j in neighbourhood

    The MAD is converted to an equivalent Gaussian standard deviation
    via the consistency constant (see Leys et al., 2013):

        sigma_i = MAD_TO_SIGMA * MAD_i

    A data point z_i is classified as a spike if:

        | z_i - m_i | > threshold_factor * sigma_i

    Detected spikes are replaced by the local median m_i, which is the
    minimax-optimal estimator of location under the Hampel (1974)
    breakdown framework.

    Parameters
    ----------
    z_raw : ndarray, shape (N,)
        Raw height signal.
    window_size : int, optional
        Size of the sliding neighbourhood (must be odd, default 15).
    threshold_factor : float, optional
        Rejection threshold in units of robust standard deviations
        (default 3.0, corresponding to a nominal 0.27% false-positive
        rate under Gaussianity).

    Returns
    -------
    z_clean : ndarray, shape (N,)
        Height signal with spikes replaced by local medians.
    n_spikes : int
        Number of samples identified and corrected as spikes.
    """
    z_clean = z_raw.copy()
    half_w = window_size // 2

    for i in range(len(z_clean)):
        lo = max(0, i - half_w)
        hi = min(len(z_clean), i + half_w + 1)
        neighbourhood = z_raw[lo:hi]

        local_median = np.median(neighbourhood)
        local_mad = np.median(np.abs(neighbourhood - local_median))

        if local_mad > 0:
            robust_z = abs(z_clean[i] - local_median) / (MAD_TO_SIGMA * local_mad)
            if robust_z > threshold_factor:
                z_clean[i] = local_median

    n_spikes = int(np.sum(z_clean != z_raw))
    return z_clean, n_spikes


def apply_median_filter(z_data, kernel_size=5):
    """
    Suppress high-frequency measurement noise via a running median filter.

    The median filter is a rank-order, non-linear operator that replaces
    each sample with the median of its ``kernel_size``-point
    neighbourhood. Unlike linear (e.g. Gaussian) smoothing, the median
    filter preserves sharp step edges and does not introduce ringing
    artefacts near discontinuities (Tukey, 1977). This property is
    essential for retaining the true geometric transitions at the bead
    boundaries.

    Parameters
    ----------
    z_data : ndarray, shape (N,)
        Height signal after spike removal.
    kernel_size : int, optional
        Width of the median window in samples (default 5). Automatically
        incremented to the next odd integer if an even value is supplied.

    Returns
    -------
    z_filtered : ndarray, shape (N,)
        Smoothed height signal.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    return medfilt(z_data, kernel_size=kernel_size)


def normalise_to_origin(x_data, z_data):
    """
    Translate the coordinate system so that both the lateral position
    and the height axes originate at zero.

    This normalisation removes the arbitrary offset inherent to the
    scanner coordinate frame and enables direct comparison of geometric
    features across different specimens and measurement sessions.

    Parameters
    ----------
    x_data : ndarray, shape (N,)
        Lateral position values.
    z_data : ndarray, shape (N,)
        Height values.

    Returns
    -------
    x_norm : ndarray, shape (N,)
        Position values with min(x) = 0.
    z_norm : ndarray, shape (N,)
        Height values with min(z) = 0.
    """
    return x_data - x_data.min(), z_data - z_data.min()


def preprocess_profile(x_raw, z_raw, median_kernel=5,
                       spike_window=15, spike_threshold=3.0):
    """
    Execute the full signal conditioning pipeline.

    The processing order is chosen deliberately:

        1. **Spike removal first**, because isolated extreme values
           (instrument artefacts, reflectance anomalies) would otherwise
           bias the median filter output over the width of its kernel.
        2. **Median filtering second**, to suppress broadband noise
           remaining after despiking.
        3. **Origin normalisation last**, so that the zero-reference is
           computed on the clean, filtered signal.

    Parameters
    ----------
    x_raw, z_raw : ndarray
        Raw profile coordinates from the scanner.
    median_kernel : int
        Median filter kernel width (samples).
    spike_window : int
        Neighbourhood size for spike detection (samples).
    spike_threshold : float
        MAD-based rejection threshold for spike classification.

    Returns
    -------
    x_clean : ndarray
        Normalised lateral positions (starting from zero).
    z_clean : ndarray
        Conditioned and normalised height values (starting from zero).
    n_spikes : int
        Number of spikes removed during preprocessing.
    """
    # Stage 1: Spike removal
    z_despiked, n_spikes = remove_spikes(z_raw, spike_window, spike_threshold)

    # Stage 2: Median filtering
    z_filtered = apply_median_filter(z_despiked, median_kernel)

    # Stage 3: Origin normalisation
    x_clean, z_clean = normalise_to_origin(x_raw, z_filtered)

    return x_clean, z_clean, n_spikes


# ============================================================================
# SECTION 3 — SUBSTRATE IDENTIFICATION AND PEAK DETECTION
# ============================================================================

def identify_substrate_region(z_data, percentile=67, sigma=3.0):
    """
    Delineate the elevated substrate region within the full-width profile.

    The substrate—the deposited weld bead material—occupies the upper
    portion of the height distribution. The boundary is established by
    applying a percentile-based threshold to a Gaussian-smoothed copy
    of the height signal:

        threshold = P_{percentile}( G_{sigma} * z )

    where G_{sigma} denotes a Gaussian kernel of standard deviation
    ``sigma`` and P_{q} is the q-th percentile operator. All indices
    where the smoothed signal exceeds this threshold are classified as
    substrate.

    Parameters
    ----------
    z_data : ndarray, shape (N,)
        Normalised height signal.
    percentile : float
        Percentile rank defining the substrate boundary (default 67,
        i.e. the upper third of the height distribution).
    sigma : float
        Standard deviation of the Gaussian smoothing kernel applied
        prior to thresholding (default 3.0 samples).

    Returns
    -------
    substrate_indices : ndarray of int
        Array indices corresponding to the substrate region.
    """
    z_smooth = gaussian_filter1d(z_data, sigma=sigma)
    threshold = np.percentile(z_smooth, percentile)
    return np.where(z_smooth > threshold)[0]


def detect_peaks(x_data, z_data, substrate_indices,
                 smooth_sigma=2.0, prominence_min=0.04,
                 width_min=5, distance_min=10,
                 min_base_width_mm=0.5,
                 spatial_outlier_factor=2.0):
    """
    Locate overlap bead peaks within the substrate region using
    topographic prominence as the primary detection criterion,
    followed by geometric and spatial validation to reject artefacts.

    The detection proceeds in two stages:

    **Stage 1 — Candidate identification.**
    The substrate height signal is smoothed with a Gaussian kernel
    to suppress residual noise. Local maxima satisfying prominence,
    width, and inter-peak distance thresholds are identified as
    candidate peaks.

    **Stage 2 — Geometric and spatial validation.**
    Candidates are subjected to two additional filters:

        (a) *Minimum base width*: peaks whose base width (the lateral
            distance between the left and right prominence saddle
            points) falls below ``min_base_width_mm`` are rejected.
            Physically, an overlap bead cannot be narrower than the
            scanner lateral resolution; sub-millimetre features are
            surface roughness artefacts, not deposited material.

        (b) *Spatial coherence*: if multiple peaks are detected, any
            peak whose position lies more than
            ``spatial_outlier_factor`` × (median inter-peak spacing)
            from its nearest neighbour is flagged as an isolated
            outlier and removed. This eliminates edge artefacts and
            noise features far from the actual overlap zone on the
            substrate surface.

    Parameters
    ----------
    x_data, z_data : ndarray
        Full normalised profile coordinates.
    substrate_indices : ndarray of int
        Indices defining the substrate region.
    smooth_sigma : float
        Gaussian smoothing sigma for peak detection (samples).
    prominence_min : float
        Minimum topographic prominence [mm].
    width_min : int
        Minimum peak width [samples].
    distance_min : int
        Minimum inter-peak distance [samples].
    min_base_width_mm : float
        Minimum acceptable base width in millimetres (default 0.5).
        Peaks narrower than this are classified as roughness artefacts.
    spatial_outlier_factor : float
        Multiplier for spatial outlier rejection (default 2.0).
        A peak is rejected if its distance to the nearest neighbour
        exceeds this factor times the median inter-peak spacing.

    Returns
    -------
    peak_indices : ndarray of int
        Validated peak indices within the substrate sub-array.
    peak_properties : dict
        Properties corresponding to validated peaks only.
    n_rejected : int
        Number of candidate peaks rejected during validation.
    """
    x_sub = x_data[substrate_indices]
    z_sub = z_data[substrate_indices]
    z_sub_smooth = gaussian_filter1d(z_sub, sigma=smooth_sigma)

    # ------------------------------------------------------------------
    # Stage 1: Candidate identification via scipy.signal.find_peaks.
    # ------------------------------------------------------------------
    candidate_indices, candidate_props = find_peaks(
        z_sub_smooth,
        prominence=prominence_min,
        width=width_min,
        distance=distance_min
    )

    if len(candidate_indices) == 0:
        return candidate_indices, candidate_props, 0

    # ------------------------------------------------------------------
    # Stage 2a: Minimum base width filter.
    # Reject peaks whose prominence-defined base span is below the
    # physical plausibility threshold.
    # ------------------------------------------------------------------
    keep_mask = np.ones(len(candidate_indices), dtype=bool)

    for i in range(len(candidate_indices)):
        left_idx = int(candidate_props['left_bases'][i])
        right_idx = int(candidate_props['right_bases'][i])
        base_w = abs(x_sub[right_idx] - x_sub[left_idx])
        if base_w < min_base_width_mm:
            keep_mask[i] = False

    # ------------------------------------------------------------------
    # Stage 2b: Spatial coherence filter.
    # Reject peaks that are spatially isolated from the main overlap
    # zone on the substrate surface. Two sub-checks are applied:
    #
    #   (i)  Substrate centroid test: the substrate region has a
    #        well-defined lateral centre of mass. Peaks located far
    #        from this centroid (beyond the central 80% of the
    #        substrate extent) are likely edge artefacts, not beads
    #        situated on the deposited material surface.
    #
    #   (ii) Nearest-neighbour test (3+ peaks): if multiple peaks
    #        remain, any peak whose distance to its nearest neighbour
    #        exceeds spatial_outlier_factor × median inter-peak spacing
    #        is removed as a spatial outlier.
    # ------------------------------------------------------------------
    surviving_indices = candidate_indices[keep_mask]

    if len(surviving_indices) >= 1:
        # (i) Substrate centroid test.
        # Compute the central region of the substrate (middle 80%).
        x_sub_range = x_sub.max() - x_sub.min()
        x_sub_centre = (x_sub.max() + x_sub.min()) / 2.0
        margin = 0.10 * x_sub_range  # 10% margin on each side
        x_sub_lower = x_sub.min() + margin
        x_sub_upper = x_sub.max() - margin

        spatial_mask = np.ones(len(surviving_indices), dtype=bool)
        for i in range(len(surviving_indices)):
            pos_i = x_sub[surviving_indices[i]]
            if pos_i < x_sub_lower or pos_i > x_sub_upper:
                spatial_mask[i] = False

        # (ii) Nearest-neighbour test for 3+ surviving peaks.
        surviving_after_centroid = surviving_indices[spatial_mask]
        if len(surviving_after_centroid) > 2:
            peak_positions = x_sub[surviving_after_centroid]
            sorted_pos = np.sort(peak_positions)
            spacings = np.diff(sorted_pos)
            median_spacing = np.median(spacings)

            nn_mask = np.ones(len(surviving_after_centroid), dtype=bool)
            for i in range(len(surviving_after_centroid)):
                pos_i = x_sub[surviving_after_centroid[i]]
                other_positions = np.array([
                    x_sub[surviving_after_centroid[j]]
                    for j in range(len(surviving_after_centroid)) if j != i
                ])
                min_dist = np.min(np.abs(pos_i - other_positions))

                if min_dist > spatial_outlier_factor * median_spacing:
                    nn_mask[i] = False

            # Map nn_mask back to spatial_mask.
            centroid_survivors = np.where(spatial_mask)[0]
            for i, sm_idx in enumerate(centroid_survivors):
                if not nn_mask[i]:
                    spatial_mask[sm_idx] = False

        # Map spatial_mask back to keep_mask.
        surviving_in_original = np.where(keep_mask)[0]
        for i, orig_idx in enumerate(surviving_in_original):
            if not spatial_mask[i]:
                keep_mask[orig_idx] = False

    # ------------------------------------------------------------------
    # Stage 3: Recompute properties for validated peaks.
    # After removing artefact peaks, the prominence and base boundaries
    # of the surviving peaks may have been influenced by the rejected
    # neighbours (scipy's find_peaks computes bases relative to all
    # detected peaks). We therefore re-run find_peaks on a signal where
    # rejected peak regions have been suppressed, yielding correct base
    # boundaries for the validated peaks only.
    # ------------------------------------------------------------------
    n_rejected = int(np.sum(~keep_mask))
    valid_indices = candidate_indices[keep_mask]

    if n_rejected > 0 and len(valid_indices) > 0:
        # Create a modified signal where rejected peak regions are
        # replaced by local baseline values, so they no longer
        # influence the prominence computation of surviving peaks.
        z_modified = z_sub_smooth.copy()
        rejected_indices = candidate_indices[~keep_mask]
        for ri in range(len(rejected_indices)):
            # Find the original left/right bases of the rejected peak.
            orig_idx = np.where(~keep_mask)[0][ri]
            rl = int(candidate_props['left_bases'][orig_idx])
            rr = int(candidate_props['right_bases'][orig_idx])
            lo, hi = min(rl, rr), max(rl, rr)
            # Replace that region with a linear interpolation (flatten).
            if hi > lo:
                z_modified[lo:hi + 1] = np.linspace(
                    z_modified[lo], z_modified[hi], hi - lo + 1
                )

        # Re-detect on the modified signal with relaxed thresholds
        # to recover the validated peaks with corrected bases.
        recomp_indices, recomp_props = find_peaks(
            z_modified,
            prominence=prominence_min * 0.5,
            width=width_min,
            distance=distance_min
        )

        # Match recomputed peaks to validated peaks by nearest position.
        if len(recomp_indices) > 0:
            matched_props = {
                'prominences': [],
                'left_bases': [],
                'right_bases': [],
                'widths': [],
                'width_heights': [],
                'left_ips': [],
                'right_ips': []
            }
            matched_indices = []

            for vi in valid_indices:
                distances = np.abs(recomp_indices - vi)
                closest = np.argmin(distances)
                if distances[closest] <= distance_min:
                    matched_indices.append(recomp_indices[closest])
                    for key in matched_props:
                        if key in recomp_props:
                            matched_props[key].append(
                                recomp_props[key][closest]
                            )
                else:
                    # Fallback: keep the original if no match found.
                    matched_indices.append(vi)
                    orig_pos = np.where(
                        candidate_indices == vi
                    )[0][0]
                    for key in matched_props:
                        if key in candidate_props:
                            matched_props[key].append(
                                candidate_props[key][orig_pos]
                            )

            valid_indices = np.array(matched_indices)
            valid_props = {}
            for key in matched_props:
                if len(matched_props[key]) > 0:
                    valid_props[key] = np.array(matched_props[key])
        else:
            # Fallback: use original properties.
            valid_props = {}
            for key, val in candidate_props.items():
                valid_props[key] = val[keep_mask]
    else:
        valid_props = {}
        for key, val in candidate_props.items():
            valid_props[key] = val[keep_mask]

    return valid_indices, valid_props, n_rejected


# ============================================================================
# SECTION 4 — GEOMETRIC CHARACTERISATION OF INDIVIDUAL PEAKS
# ============================================================================

def characterise_peak(x_data, z_data, substrate_indices,
                      peak_idx, properties, peak_number):
    """
    Compute a comprehensive set of geometric descriptors for a single
    overlap bead peak.

    The following quantities are extracted:

    **Position and Height**
        - Peak apex coordinates (X_peak, Z_peak).
        - Topographic prominence: the height of the peak above the
          higher of the two bounding saddle points.

    **Width Metrics**
        - Base width (W_base): lateral distance between left and right
          base points as determined by the prominence calculation.
        - Full Width at Half Maximum (FWHM): lateral extent at 50%
          of the prominence above the baseline, a standard measure
          of peak breadth insensitive to tailing effects.
        - Width at 25% prominence (W_25): provides additional shape
          information about the lower flanks.

    **Cross-Sectional Area**
        Area of the peak above a linear baseline connecting the two
        base points, computed via trapezoidal numerical integration.
        This quantity is proportional to the excess material volume
        per unit bead length.

    **Slope Angles**
        Linear regression of the left and right flanks (base to apex)
        yields representative slope angles theta_L and theta_R
        [degrees]. The average |theta_L| + |theta_R|) / 2 provides
        a single steepness metric.

    **Asymmetry**
        Width asymmetry ratio = W_left / W_right, where W_left and
        W_right are the lateral distances from the apex to the left
        and right base points respectively. A ratio of 1.0 indicates
        perfect symmetry; values below 0.9 indicate right-skew, and
        above 1.1 indicate left-skew.

    **Curvature**
        The discrete second derivative of the smoothed height signal
        at the apex, d^2z/dx^2, evaluated by central finite
        differences. A high absolute value indicates a sharp, pointed
        peak; a low value indicates a flat, plateau-like summit.

    Parameters
    ----------
    x_data, z_data : ndarray
        Full normalised profile coordinates.
    substrate_indices : ndarray of int
        Substrate region indices.
    peak_idx : int
        Index of the peak within the substrate sub-array.
    properties : dict
        Peak properties from ``scipy.signal.find_peaks``.
    peak_number : int
        Sequential peak identifier (1-indexed).

    Returns
    -------
    dimensions : dict
        Dictionary containing all computed geometric descriptors.
    """
    x_sub = x_data[substrate_indices]
    z_sub = z_data[substrate_indices]
    z_smooth = gaussian_filter1d(z_sub, sigma=2)

    # --- Apex coordinates ---
    peak_x = x_sub[peak_idx]
    peak_z = z_sub[peak_idx]

    # --- Prominence and base indices ---
    prominence = properties['prominences'][peak_number - 1]
    left_base_idx = int(properties['left_bases'][peak_number - 1])
    right_base_idx = int(properties['right_bases'][peak_number - 1])

    # --- Base width using LOCAL BASELINE ---
    # The overlap peak sits on top of the local surface. We find the
    # local baseline by taking the median height in the flat regions
    # flanking the peak (2-6 mm on each side). The base boundaries
    # are where the profile crosses this local baseline.
    search_inner = 2.0  # mm from peak to start looking for flat
    search_outer = 6.0  # mm from peak to stop looking

    left_flat_mask = (x_sub > peak_x - search_outer) & (x_sub < peak_x - search_inner)
    right_flat_mask = (x_sub > peak_x + search_inner) & (x_sub < peak_x + search_outer)

    left_flat = z_smooth[left_flat_mask] if np.any(left_flat_mask) else z_smooth[:20]
    right_flat = z_smooth[right_flat_mask] if np.any(right_flat_mask) else z_smooth[-20:]
    local_baseline = (np.median(left_flat) + np.median(right_flat)) / 2.0

    # Recompute prominence relative to local baseline
    local_prominence = peak_z - local_baseline
    if local_prominence < 0.01:
        # Fallback: use scipy prominence if local baseline is above peak
        local_prominence = prominence
        local_baseline = z_smooth[peak_idx] - prominence

    # Transition level at 5% of local prominence above baseline
    transition_level = local_baseline + 0.05 * local_prominence

    # Scan outward from peak until signal drops below transition level
    lo_end = min(left_base_idx, right_base_idx)
    hi_end = max(left_base_idx, right_base_idx)

    # Scan outward from peak toward the low-index end.
    refined_left = lo_end
    for idx_scan in range(peak_idx, lo_end - 1, -1):
        if z_smooth[idx_scan] < transition_level:
            refined_left = idx_scan
            break

    # Scan outward from peak toward the high-index end.
    refined_right = hi_end
    for idx_scan in range(peak_idx, hi_end + 1):
        if z_smooth[idx_scan] < transition_level:
            refined_right = idx_scan
            break

    base_left_x = x_sub[refined_left]
    base_right_x = x_sub[refined_right]
    # Canonical ordering: ensure left < right for plotting consistency.
    if base_left_x > base_right_x:
        base_left_x, base_right_x = base_right_x, base_left_x

    # Update base indices for downstream calculations.
    left_base_idx = refined_left
    right_base_idx = refined_right

    base_width = abs(base_right_x - base_left_x)

    # --- Full Width at Half Maximum (FWHM) ---
    # Use LOCAL baseline and LOCAL prominence for all width/area calculations
    baseline_z = local_baseline
    half_height_z = local_baseline + local_prominence / 2.0

    peak_region = z_smooth[left_base_idx:right_base_idx + 1]

    # Left crossing of the half-prominence contour.
    left_crossings = np.where(
        peak_region[:peak_idx - left_base_idx] <= half_height_z
    )[0]
    left_fwhm_x = (
        x_sub[left_base_idx + left_crossings[-1]]
        if len(left_crossings) > 0 else base_left_x
    )

    # Right crossing of the half-prominence contour.
    right_crossings = np.where(
        peak_region[peak_idx - left_base_idx:] <= half_height_z
    )[0]
    right_fwhm_x = (
        x_sub[peak_idx + right_crossings[0]]
        if len(right_crossings) > 0 else base_right_x
    )

    fwhm = abs(right_fwhm_x - left_fwhm_x)
    if left_fwhm_x > right_fwhm_x:
        left_fwhm_x, right_fwhm_x = right_fwhm_x, left_fwhm_x

    # --- Width at 25% prominence ---
    quarter_height_z = local_baseline + local_prominence / 4.0
    left_q = np.where(
        peak_region[:peak_idx - left_base_idx] <= quarter_height_z
    )[0]
    right_q = np.where(
        peak_region[peak_idx - left_base_idx:] <= quarter_height_z
    )[0]
    width_25 = None
    if len(left_q) > 0 and len(right_q) > 0:
        width_25 = abs(
            x_sub[peak_idx + right_q[0]]
            - x_sub[left_base_idx + left_q[-1]]
        )

    # --- Cross-sectional area above local baseline ---
    idx_lo = min(left_base_idx, right_base_idx)
    idx_hi = max(left_base_idx, right_base_idx)
    x_region = x_sub[idx_lo:idx_hi + 1]
    z_region = z_sub[idx_lo:idx_hi + 1]
    sort_order = np.argsort(x_region)
    x_region = x_region[sort_order]
    z_region = z_region[sort_order]

    # Area above the local baseline (flat surface level)
    z_above = np.maximum(z_region - local_baseline, 0)
    area = abs(np.trapezoid(z_above, x_region))

    # --- Flank slope angles ---
    left_flank_x = x_sub[left_base_idx:peak_idx + 1]
    left_flank_z = z_sub[left_base_idx:peak_idx + 1]
    if len(left_flank_x) > 1:
        slope_left = np.polyfit(left_flank_x, left_flank_z, 1)[0]
        angle_left = np.degrees(np.arctan(slope_left))
    else:
        angle_left = 0.0

    right_flank_x = x_sub[peak_idx:right_base_idx + 1]
    right_flank_z = z_sub[peak_idx:right_base_idx + 1]
    if len(right_flank_x) > 1:
        slope_right = np.polyfit(right_flank_x, right_flank_z, 1)[0]
        angle_right = np.degrees(np.arctan(slope_right))
    else:
        angle_right = 0.0

    avg_slope_angle = (abs(angle_left) + abs(angle_right)) / 2.0

    # --- Width asymmetry ---
    left_width = abs(peak_x - base_left_x)
    right_width = abs(base_right_x - peak_x)
    asymmetry_ratio = (
        left_width / right_width if right_width > 0
        else float('inf')
    )

    # --- Apex curvature (discrete second derivative) ---
    if 0 < peak_idx < len(z_smooth) - 1:
        curvature = abs(
            z_smooth[peak_idx + 1]
            - 2.0 * z_smooth[peak_idx]
            + z_smooth[peak_idx - 1]
        )
    else:
        curvature = 0.0

    if curvature < 0.001:
        sharpness_class = "Flat plateau"
    elif curvature < 0.005:
        sharpness_class = "Rounded"
    else:
        sharpness_class = "Sharp"

    # --- Assemble output dictionary ---
    return {
        'peak_number':      peak_number,
        'position_x':       peak_x,
        'height_z':         peak_z,
        'prominence':       local_prominence,
        'local_baseline':   local_baseline,
        'base_width':       base_width,
        'base_left_x':      base_left_x,
        'base_right_x':     base_right_x,
        'fwhm':             fwhm,
        'fwhm_left_x':      left_fwhm_x,
        'fwhm_right_x':     right_fwhm_x,
        'half_height_z':    half_height_z,
        'width_25':         width_25,
        'area':             area,
        'left_slope_angle':  angle_left,
        'right_slope_angle': angle_right,
        'avg_slope_angle':   avg_slope_angle,
        'left_width':        left_width,
        'right_width':       right_width,
        'asymmetry_ratio':   asymmetry_ratio,
        'curvature':         curvature,
        'sharpness':         sharpness_class
    }


# ============================================================================
# SECTION 5 — OVERLAP ZONE COMPUTATION
# ============================================================================

def compute_overlap_zone(all_dimensions):
    """
    Determine the total lateral extent of the overlap zone.

    The overlap zone is defined using the **FWHM boundaries** of the
    outermost peaks rather than the base boundaries. This is because
    the base boundaries from the prominence calculation can extend
    well into the flat substrate region — particularly for broad,
    dome-shaped peaks — yielding overlap widths that overestimate
    the physically relevant bead footprint. The FWHM boundaries
    mark the region where the bead profile rises significantly above
    the substrate and thus provide a more conservative and physically
    meaningful delineation of the overlap zone.

    The overlap zone base width is defined as:

        W_overlap = max(fwhm_right_x, base_right_x)
                  - min(fwhm_left_x, base_left_x)

    taken over all detected peaks, using the **base boundaries** of
    the outermost peaks (not the FWHM boundaries) to capture the
    full bead footprint, but only considering peaks that passed
    validation (i.e., physically real beads on the substrate surface).

    For single-peak specimens, the overlap zone equals the individual
    peak's base width.

    Parameters
    ----------
    all_dimensions : list of dict
        List of dimension dictionaries as returned by
        ``characterise_peak``.

    Returns
    -------
    overlap_zone : dict
        Dictionary with keys 'left_x', 'right_x', 'width', and
        'base_z' (the minimum baseline height across all peaks).
    """
    # Use base boundaries from validated peaks.
    left = min(d['base_left_x'] for d in all_dimensions)
    right = max(d['base_right_x'] for d in all_dimensions)
    base_z = min(d.get('local_baseline', d['height_z'] - d['prominence'])
                 for d in all_dimensions)

    return {
        'left_x':  left,
        'right_x': right,
        'width':   right - left,
        'base_z':  base_z
    }


# ============================================================================
# SECTION 6 — VISUALISATION
# ============================================================================

def generate_visualisation(x_data, z_data, substrate_indices,
                           all_dimensions, overlap_zone,
                           sample_name, save_path):
    """
    Produce a multi-panel figure with an overview plot showing all
    detected peaks and the overlap zone, followed by individual
    detail plots for each peak with annotated dimensional features.

    Parameters
    ----------
    x_data, z_data : ndarray
        Full normalised profile data.
    substrate_indices : ndarray of int
        Substrate region indices.
    all_dimensions : list of dict
        Per-peak dimension dictionaries.
    overlap_zone : dict
        Overlap zone geometry.
    sample_name : str
        Specimen label for plot titles.
    save_path : str
        File path for the output image (PNG, 300 dpi).
    """
    x_sub = x_data[substrate_indices]
    z_sub = z_data[substrate_indices]

    n_peaks = len(all_dimensions)
    fig, axes = plt.subplots(
        n_peaks + 1, 1,
        figsize=(18, 6 * (n_peaks + 1)),
        squeeze=False
    )
    axes = axes.flatten()

    palette = ['red', 'green', 'orange', 'purple', 'brown']

    # ---- Panel 0: Overview ----
    ax0 = axes[0]
    ax0.plot(x_data, z_data, 'b-', linewidth=2, alpha=0.7,
             label='Full Profile')

    for idx, dims in enumerate(all_dimensions):
        colour = palette[idx % len(palette)]
        ax0.scatter(
            dims['position_x'], dims['height_z'],
            s=200, c=colour, marker='^', zorder=5,
            edgecolors='black', linewidths=2,
            label=f"B{dims['peak_number']}"
        )

    # Overlap zone base width annotation.
    oz = overlap_zone
    ax0.hlines(
        oz['base_z'], oz['left_x'], oz['right_x'],
        colors='darkgreen', linewidth=3, linestyles='--',
        label=f"Overlap Zone Width = {oz['width']:.3f}"
    )
    ax0.plot([oz['left_x']] * 2,
             [oz['base_z'], oz['base_z'] + 0.02],
             'darkgreen', linewidth=2, linestyle='--')
    ax0.plot([oz['right_x']] * 2,
             [oz['base_z'], oz['base_z'] + 0.02],
             'darkgreen', linewidth=2, linestyle='--')

    mid_x = (oz['left_x'] + oz['right_x']) / 2
    ax0.annotate(
        f"Overlap Zone Base Width = {oz['width']:.3f}",
        xy=(mid_x, oz['base_z']),
        xytext=(mid_x, oz['base_z'] - 0.03),
        fontsize=12, fontweight='bold', color='darkgreen',
        ha='center',
        bbox=dict(boxstyle='round,pad=0.4',
                  facecolor='lightgreen', alpha=0.8),
        arrowprops=dict(arrowstyle='->', color='darkgreen', lw=2)
    )

    ax0.set_xlabel('Position, X [mm]', fontsize=13, fontweight='bold')
    ax0.set_ylabel('Z (Height) [mm]', fontsize=13, fontweight='bold')
    ax0.set_title(f'{sample_name} — Overview: All Overlap Peaks',
                  fontsize=15, fontweight='bold')
    ax0.legend(loc='best', fontsize=12)
    ax0.xaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax0.xaxis.set_minor_locator(plt.MultipleLocator(0.25))
    ax0.yaxis.set_major_locator(plt.MultipleLocator(0.025))
    ax0.yaxis.set_minor_locator(plt.MultipleLocator(0.005))
    ax0.tick_params(axis='both', which='major', labelsize=11)
    ax0.grid(True, which='major', alpha=0.4)
    ax0.grid(True, which='minor', alpha=0.15)

    # ---- Panels 1..N: Individual peak details ----
    for idx, dims in enumerate(all_dimensions):
        ax = axes[idx + 1]
        colour = palette[idx % len(palette)]

        ax.plot(x_sub, z_sub, 'b-', linewidth=2, alpha=0.5,
                label='Profile')

        # Peak apex marker.
        ax.scatter(
            dims['position_x'], dims['height_z'],
            s=300, c=colour, marker='^', zorder=10,
            edgecolors='black', linewidths=3
        )

        base_z_line = dims.get('local_baseline', dims['height_z'] - dims['prominence'])

        # Base width indicator.
        ax.hlines(base_z_line,
                  dims['base_left_x'], dims['base_right_x'],
                  colors='green', linewidth=3,
                  label='Base Width', linestyles='--')
        ax.plot([dims['base_left_x']] * 2,
                [base_z_line, dims['height_z']], 'g--', linewidth=2)
        ax.plot([dims['base_right_x']] * 2,
                [base_z_line, dims['height_z']], 'g--', linewidth=2)

        # FWHM indicator.
        ax.hlines(dims['half_height_z'],
                  dims['fwhm_left_x'], dims['fwhm_right_x'],
                  colors='red', linewidth=3,
                  label='FWHM', linestyles='-')
        ax.plot([dims['fwhm_left_x']] * 2,
                [base_z_line, dims['half_height_z']], 'r:', linewidth=2)
        ax.plot([dims['fwhm_right_x']] * 2,
                [base_z_line, dims['half_height_z']], 'r:', linewidth=2)

        # Prominence line.
        ax.vlines(dims['position_x'],
                  base_z_line, dims['height_z'],
                  colors='purple', linewidth=3, label='Prominence')

        # Apex coordinate label.
        ax.text(
            dims['position_x'], dims['height_z'] + 0.005,
            f"B{dims['peak_number']}: "
            f"X={dims['position_x']:.3f}, Z={dims['height_z']:.3f}",
            ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5',
                      facecolor='yellow', alpha=0.8)
        )

        # Summary statistics inset.
        info = (
            f"Base Width: {dims['base_width']:.3f}\n"
            f"FWHM: {dims['fwhm']:.3f}\n"
            f"Area: {dims['area']:.4f}\n"
            f"Asymmetry: {dims['asymmetry_ratio']:.2f}\n"
            f"Avg Slope: {dims['avg_slope_angle']:.1f}\u00b0"
        )
        ax.text(
            0.02, 0.98, info,
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )

        ax.set_xlabel('Position, X [mm]', fontsize=12, fontweight='bold')
        ax.set_ylabel('Z (Height) [mm]', fontsize=12, fontweight='bold')
        ax.set_title(
            f'{sample_name} — Peak B{dims["peak_number"]} '
            f'Detailed Dimensions',
            fontsize=14, fontweight='bold'
        )
        ax.legend(loc='upper right', fontsize=10)
        ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.01))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(0.005))
        ax.tick_params(axis='both', which='major', labelsize=11)
        ax.grid(True, which='major', alpha=0.4)
        ax.grid(True, which='minor', alpha=0.15)

        margin = dims['base_width'] * 0.3
        ax.set_xlim(dims['base_left_x'] - margin,
                     dims['base_right_x'] + margin)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# SECTION 7 — REPORT GENERATION
# ============================================================================

def generate_report(all_dimensions, overlap_zone, sample_name, save_path):
    """
    Write a structured plain-text measurement report containing
    individual peak characterisation tables and a cross-peak
    comparison summary.

    Parameters
    ----------
    all_dimensions : list of dict
        Per-peak dimension dictionaries.
    overlap_zone : dict
        Overlap zone geometry.
    sample_name : str
        Specimen label.
    save_path : str
        Output file path (.txt).
    """
    with open(save_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write(f"OVERLAP BEAD DIMENSION ANALYSIS — {sample_name}\n")
        f.write("=" * 70 + "\n\n")

        # Overlap zone summary.
        oz = overlap_zone
        f.write("OVERLAP ZONE (ALL PEAKS)\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Left boundary:   X = {oz['left_x']:.4f} mm\n")
        f.write(f"  Right boundary:  X = {oz['right_x']:.4f} mm\n")
        f.write(f"  Total base width:    {oz['width']:.4f} mm\n\n")

        # Individual peak reports.
        for dims in all_dimensions:
            f.write(f"PEAK B{dims['peak_number']}\n")
            f.write("-" * 70 + "\n\n")

            f.write("1. POSITION AND HEIGHT\n")
            f.write(f"   X Position:  {dims['position_x']:.4f} mm\n")
            f.write(f"   Z Height:    {dims['height_z']:.4f} mm\n")
            f.write(f"   Prominence:  {dims['prominence']:.4f} mm\n\n")

            f.write("2. WIDTH MEASUREMENTS\n")
            f.write(f"   Base Width:  {dims['base_width']:.4f} mm\n")
            f.write(f"     Left edge:  {dims['base_left_x']:.4f} mm\n")
            f.write(f"     Right edge: {dims['base_right_x']:.4f} mm\n")
            f.write(f"   FWHM:        {dims['fwhm']:.4f} mm\n")
            f.write(f"     Half-height level: {dims['half_height_z']:.4f} mm\n")
            f.write(f"     Left FWHM:  {dims['fwhm_left_x']:.4f} mm\n")
            f.write(f"     Right FWHM: {dims['fwhm_right_x']:.4f} mm\n")
            if dims['width_25'] is not None:
                f.write(f"   Width at 25%: {dims['width_25']:.4f} mm\n")
            f.write("\n")

            f.write("3. CROSS-SECTIONAL AREA\n")
            f.write(f"   Area above baseline: {dims['area']:.6f} mm^2\n\n")

            f.write("4. FLANK SLOPE ANGLES\n")
            f.write(f"   Left flank:   {dims['left_slope_angle']:.2f} deg\n")
            f.write(f"   Right flank:  {dims['right_slope_angle']:.2f} deg\n")
            f.write(f"   Average:      {dims['avg_slope_angle']:.2f} deg\n\n")

            f.write("5. WIDTH ASYMMETRY\n")
            f.write(f"   Left width:   {dims['left_width']:.4f} mm\n")
            f.write(f"   Right width:  {dims['right_width']:.4f} mm\n")
            f.write(f"   Ratio (L/R):  {dims['asymmetry_ratio']:.4f}\n")
            if dims['asymmetry_ratio'] < 0.9:
                f.write("   Classification: RIGHT-skewed\n")
            elif dims['asymmetry_ratio'] > 1.1:
                f.write("   Classification: LEFT-skewed\n")
            else:
                f.write("   Classification: SYMMETRIC\n")
            f.write("\n")

            f.write("6. APEX CURVATURE\n")
            f.write(f"   Curvature value:  {dims['curvature']:.6f}\n")
            f.write(f"   Classification:   {dims['sharpness']}\n")
            f.write("\n" + "=" * 70 + "\n\n")

        # Comparison table.
        f.write("SUMMARY COMPARISON TABLE\n")
        f.write("-" * 70 + "\n")
        f.write(
            f"{'Peak':<8} {'X [mm]':<10} {'Z [mm]':<10} "
            f"{'Base W':<10} {'FWHM':<10} {'Area':<10} {'Asym':<8}\n"
        )
        f.write("-" * 70 + "\n")
        for dims in all_dimensions:
            f.write(
                f"B{dims['peak_number']:<7} "
                f"{dims['position_x']:<10.3f} "
                f"{dims['height_z']:<10.3f} "
                f"{dims['base_width']:<10.3f} "
                f"{dims['fwhm']:<10.3f} "
                f"{dims['area']:<10.4f} "
                f"{dims['asymmetry_ratio']:<8.2f}\n"
            )
        f.write("=" * 70 + "\n")


# ============================================================================
# SECTION 8 — ORCHESTRATION (SINGLE SPECIMEN)
# ============================================================================

def format_sample_name(filepath):
    """Derive a human-readable sample name from the .slk filename.

    Examples:
        Sample_2.1A_00001.slk  →  Sample 2.1A
        Sample_3_1A_00010.slk  →  Sample 3_1A
        my_profile.slk         →  my_profile
    """
    base = os.path.basename(filepath)
    # Remove trailing _NNNNN.slk or .slk
    name = re.sub(r'_\d{4,5}\.slk$', '', base)
    if name == base:
        name = base.replace('.slk', '')
    # Replace leading 'Sample_' with 'Sample ' for readability
    name = re.sub(r'^Sample_', 'Sample ', name)
    return name


def analyze_sample(filepath, sample_name=None, output_dir=None,
                   median_kernel=5, spike_window=15,
                   spike_threshold=3.0, substrate_percentile=67,
                   prominence_threshold=0.04,
                   min_peak_width=5, min_peak_distance=10,
                   min_base_width_mm=0.5,
                   spatial_outlier_factor=2.0):
    """
    Execute the complete analysis pipeline for a single specimen.

    Parameters
    ----------
    filepath : str
        Path to the ``.slk`` profile file,
        e.g. ``/content/Sample_2.1A_00001.slk``.
    sample_name : str, optional
        Human-readable label. If not provided, it is derived from
        the filename (e.g. ``Sample_2.1A_00001.slk`` → ``Sample 2.1A``).
    output_dir : str, optional
        Directory for output files. Default: same folder as input.
    median_kernel : int
        Median filter kernel width.
    spike_window : int
        Neighbourhood size for spike detection.
    spike_threshold : float
        MAD-based spike rejection threshold.
    substrate_percentile : float
        Percentile for substrate delineation.
    prominence_threshold : float
        Minimum topographic prominence for peak acceptance.
    min_peak_width : int
        Minimum peak width in samples.
    min_peak_distance : int
        Minimum inter-peak distance in samples.
    min_base_width_mm : float
        Minimum acceptable peak base width in mm (rejects roughness).
    spatial_outlier_factor : float
        Spatial coherence factor for rejecting isolated peaks.

    Returns
    -------
    result : dict
        Contains 'sample_name', 'all_dimensions', 'overlap_zone',
        'image_path', 'report_path'.
    """
    if sample_name is None:
        sample_name = format_sample_name(filepath)
    if output_dir is None:
        output_dir = os.path.dirname(filepath) or '.'
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'#' * 70}")
    print(f"  ANALYSING: {sample_name}")
    print(f"  Source:    {filepath}")
    print(f"{'#' * 70}\n")

    # Step 1: Data ingestion.
    print("  [1/7] Loading profile data...")
    x_raw, z_raw = load_slk_profile(filepath)
    print(f"         {len(x_raw)} data points loaded.")

    # Step 2: Signal conditioning.
    print("  [2/7] Signal conditioning (spike removal + median filter)...")
    x_data, z_data, n_spikes = preprocess_profile(
        x_raw, z_raw,
        median_kernel=median_kernel,
        spike_window=spike_window,
        spike_threshold=spike_threshold
    )
    print(f"         {n_spikes} spikes removed.")
    print(f"         X range: [0, {x_data.max():.4f}] mm")
    print(f"         Z range: [0, {z_data.max():.4f}] mm")

    # Step 3: Substrate identification.
    print("  [3/7] Identifying substrate region...")
    substrate_indices = identify_substrate_region(
        z_data, percentile=substrate_percentile
    )

    # Step 4: Peak detection with geometric and spatial validation.
    print("  [4/7] Detecting overlap peaks...")
    peak_indices, properties, n_rejected = detect_peaks(
        x_data, z_data, substrate_indices,
        prominence_min=prominence_threshold,
        width_min=min_peak_width,
        distance_min=min_peak_distance,
        min_base_width_mm=min_base_width_mm,
        spatial_outlier_factor=spatial_outlier_factor
    )
    print(f"         {len(peak_indices)} peaks accepted, "
          f"{n_rejected} rejected (artefacts).")

    # Step 5: Geometric characterisation.
    print("  [5/7] Characterising peak geometry...")
    all_dimensions = []
    for i, pidx in enumerate(peak_indices):
        dims = characterise_peak(
            x_data, z_data, substrate_indices,
            pidx, properties, i + 1
        )
        all_dimensions.append(dims)
        print(
            f"         B{i + 1}: X={dims['position_x']:.3f}, "
            f"Z={dims['height_z']:.3f}, "
            f"W_base={dims['base_width']:.3f}, "
            f"FWHM={dims['fwhm']:.3f}"
        )

    # Step 6: Overlap zone.
    overlap_zone = compute_overlap_zone(all_dimensions)
    print(
        f"         Overlap zone: "
        f"[{overlap_zone['left_x']:.3f}, {overlap_zone['right_x']:.3f}], "
        f"W = {overlap_zone['width']:.3f} mm"
    )

    # Step 7: Outputs.
    print("  [6/7] Generating visualisation...")
    img_path = os.path.join(output_dir, f"{sample_name}_peak_analysis.png")
    generate_visualisation(
        x_data, z_data, substrate_indices,
        all_dimensions, overlap_zone,
        sample_name, img_path
    )
    print(f"         Saved: {img_path}")

    print("  [7/7] Writing measurement report...")
    report_path = os.path.join(output_dir, f"{sample_name}_report.txt")
    generate_report(all_dimensions, overlap_zone, sample_name, report_path)
    print(f"         Saved: {report_path}")

    print(f"\n  Analysis complete: {len(peak_indices)} peaks characterised.\n")

    return {
        'sample_name':    sample_name,
        'all_dimensions': all_dimensions,
        'overlap_zone':   overlap_zone,
        'image_path':     img_path,
        'report_path':    report_path
    }


# ============================================================================
# SECTION 9 — BATCH PROCESSING
# ============================================================================

def analyze_multiple_samples(filepaths, sample_names, output_dir, **kwargs):
    """
    Execute the analysis pipeline sequentially on multiple specimens
    under identical algorithmic parameters, and produce a cross-specimen
    comparison summary.

    Parameters
    ----------
    filepaths : list of str
        Paths to ``.slk`` files.
    sample_names : list of str
        Specimen labels (same order as filepaths).
    output_dir : str
        Output directory.
    **kwargs
        Additional keyword arguments passed to ``analyze_sample``.

    Returns
    -------
    all_results : list of dict
        Per-specimen result dictionaries.
    """
    all_results = []

    print("\n" + "=" * 70)
    print(f"  BATCH PROCESSING: {len(filepaths)} specimens")
    print("=" * 70)

    for fpath, sname in zip(filepaths, sample_names):
        result = analyze_sample(fpath, sname, output_dir, **kwargs)
        all_results.append(result)

    # Cross-specimen comparison.
    print("\n" + "=" * 70)
    print("  CROSS-SPECIMEN COMPARISON")
    print("=" * 70)
    header = (
        f"\n  {'Specimen':<15} {'Peaks':<8} "
        f"{'Overlap Width [mm]':<22} {'Max Z [mm]':<18}"
    )
    print(header)
    print("  " + "-" * 63)

    for r in all_results:
        n = len(r['all_dimensions'])
        oz_w = r['overlap_zone']['width']
        max_z = max(d['height_z'] for d in r['all_dimensions'])
        print(
            f"  {r['sample_name']:<15} {n:<8} "
            f"{oz_w:<22.4f} {max_z:<18.4f}"
        )

    print("  " + "=" * 63 + "\n")

    return all_results


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    specimens = [
        ("/mnt/user-data/uploads/Sample_1_1A_00001.slk", "Sample_1"),
        ("/mnt/user-data/uploads/Sample_2_1A_00001.slk", "Sample_2"),
        ("/mnt/user-data/uploads/Sample_3_1A_00001.slk", "Sample_3"),
    ]

    paths = [s[0] for s in specimens]
    names = [s[1] for s in specimens]

    results = analyze_multiple_samples(
        paths, names,
        output_dir="/mnt/user-data/outputs",
        median_kernel=5,
        spike_window=15,
        spike_threshold=3.0,
        prominence_threshold=0.04,
        min_peak_width=5,
        min_peak_distance=10,
        min_base_width_mm=0.5,
        spatial_outlier_factor=2.0
    )
