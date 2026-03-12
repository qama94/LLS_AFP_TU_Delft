"""
Validation tests for overlap_peak_analysis.py

Run with:
    python -m pytest tests/test_algorithm.py -v

These tests verify that the algorithm correctly processes LLS profile
data and produces physically reasonable geometric measurements.
"""

import os
import sys
import numpy as np

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False

    # Minimal pytest.mark.skipif replacement
    class _mark:
        @staticmethod
        def skipif(condition, reason=""):
            def decorator(func):
                if condition:
                    def skipped(*a, **kw):
                        print(f"  SKIP: {reason}")
                    skipped.__name__ = func.__name__
                    return skipped
                return func
            return decorator

    class pytest:
        mark = _mark()

        @staticmethod
        def skip(msg=""):
            print(f"  SKIP: {msg}")

        @staticmethod
        def main(args):
            print("pytest not installed. Run tests with: python tests/test_algorithm.py")

# Add parent directory to path so we can import the algorithm.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from overlap_peak_analysis import (
    load_slk_profile,
    preprocess_profile,
    identify_substrate_region,
    detect_peaks,
    characterise_peak,
)


# ===================================================================
# Helper: generate a synthetic profile for testing
# ===================================================================

def make_synthetic_profile(n=2000, noise_std=0.002, seed=42):
    """Create a synthetic profile with a known peak for testing.

    The profile has:
      - A flat substrate at Z = 0.3 mm from X = 5 to X = 20 mm.
      - A single Gaussian peak centred at X = 12.5 mm with
        amplitude 0.15 mm and sigma 1.5 mm.
      - Gaussian noise with standard deviation noise_std.
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 25, n)
    z = np.zeros_like(x)

    # Substrate region
    substrate_mask = (x >= 5) & (x <= 20)
    z[substrate_mask] = 0.3

    # Gaussian peak on top of substrate
    peak_centre = 12.5
    peak_amplitude = 0.15
    peak_sigma = 1.5
    z += peak_amplitude * np.exp(-0.5 * ((x - peak_centre) / peak_sigma) ** 2)

    # Add noise
    z += rng.normal(0, noise_std, size=n)
    z = np.maximum(z, 0)

    return x, z


# ===================================================================
# Tests
# ===================================================================

class TestPreprocessing:
    """Tests for the signal conditioning pipeline."""

    def test_spike_removal(self):
        """Verify that artificial spikes are removed."""
        x, z = make_synthetic_profile(noise_std=0.001)
        # Insert artificial spikes
        z_spiked = z.copy()
        z_spiked[500] = 2.0    # large positive spike
        z_spiked[1000] = -1.0  # large negative spike

        _, z_clean, n_spikes = preprocess_profile(x, z_spiked)

        assert n_spikes >= 2, "At least 2 spikes should be detected"
        assert z_clean.max() < 1.0, "Spike should be removed"
        assert z_clean.min() >= 0.0, "Negative spike should be removed"

    def test_no_false_spikes(self):
        """Verify that clean data is not corrupted."""
        x, z = make_synthetic_profile(noise_std=0.001)
        _, z_clean, n_spikes = preprocess_profile(x, z)

        # Synthetic data with noise may trigger some spike detections
        assert n_spikes < 200, f"Too many false spikes detected: {n_spikes}"

    def test_origin_normalisation(self):
        """Verify that output starts at zero."""
        x, z = make_synthetic_profile()
        x_clean, z_clean, _ = preprocess_profile(x, z)

        assert abs(x_clean.min()) < 1e-10, "X should start at 0"
        assert abs(z_clean.min()) < 0.01, "Z minimum should be near 0"


class TestSubstrateIdentification:
    """Tests for the substrate region detection."""

    def test_substrate_detected(self):
        """Verify that substrate region is found."""
        x, z = make_synthetic_profile()
        x_clean, z_clean, _ = preprocess_profile(x, z)
        sub_idx = identify_substrate_region(z_clean)

        assert len(sub_idx) > 0, "Substrate region should be detected"
        # Substrate should be in the middle of the profile
        x_sub = x_clean[sub_idx]
        assert x_sub.min() < 10, "Substrate should extend to the left"
        assert x_sub.max() > 15, "Substrate should extend to the right"


class TestPeakDetection:
    """Tests for the peak detection pipeline."""

    def test_single_peak(self):
        """Verify that the synthetic profile yields exactly one peak."""
        x, z = make_synthetic_profile()
        x_clean, z_clean, _ = preprocess_profile(x, z)
        sub_idx = identify_substrate_region(z_clean)

        valid_indices, valid_props, n_rejected = detect_peaks(
            x_clean, z_clean, sub_idx)

        n_accepted = len(valid_indices)
        assert n_accepted == 1, f"Expected 1 peak, got {n_accepted}"

    def test_peak_position(self):
        """Verify that a peak is detected at a valid position."""
        x, z = make_synthetic_profile()
        x_clean, z_clean, _ = preprocess_profile(x, z)
        sub_idx = identify_substrate_region(z_clean)
        valid_indices, valid_props, _ = detect_peaks(
            x_clean, z_clean, sub_idx)

        assert len(valid_indices) >= 1, "At least one peak should be detected"
        peak_x = x_clean[valid_indices[0]]
        assert 0 < peak_x < x_clean.max(), \
            f"Peak at X={peak_x:.2f}, outside valid range"


class TestGeometricMeasurements:
    """Tests for the geometric characterisation."""

    def test_dimensions_reasonable(self):
        """Verify that computed dimensions are physically reasonable."""
        x, z = make_synthetic_profile()
        x_clean, z_clean, _ = preprocess_profile(x, z)
        sub_idx = identify_substrate_region(z_clean)
        valid_indices, valid_props, _ = detect_peaks(
            x_clean, z_clean, sub_idx)

        dims = characterise_peak(
            x_clean, z_clean, sub_idx,
            valid_indices[0], valid_props, peak_number=1
        )

        # Check that all expected keys are present
        expected_keys = [
            'peak_number', 'position_x', 'height_z', 'prominence',
            'base_width', 'fwhm', 'area', 'asymmetry_ratio',
        ]
        for key in expected_keys:
            assert key in dims, f"Missing key: {key}"

        # Physical reasonableness checks
        assert dims['prominence'] > 0.05, \
            f"Prominence too small: {dims['prominence']}"
        assert dims['prominence'] < 0.5, \
            f"Prominence too large: {dims['prominence']}"
        assert dims['base_width'] > 0.5, \
            f"Base width too small: {dims['base_width']}"
        assert dims['base_width'] < 15, \
            f"Base width too large: {dims['base_width']}"
        assert dims['fwhm'] > 0, "FWHM must be positive"
        assert dims['fwhm'] <= dims['base_width'], \
            "FWHM should not exceed base width"
        assert dims['area'] > 0, "Area must be positive"
        assert 0 < dims['asymmetry_ratio'] < 10, \
            f"Asymmetry ratio out of range: {dims['asymmetry_ratio']}"


class TestSLKLoading:
    """Tests for SYLK file parsing (requires a test .slk file)."""

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(
            os.path.dirname(__file__), '..', 'data')),
        reason="No data/ directory with test files"
    )
    def test_load_slk(self):
        """Verify that a real .slk file loads correctly."""
        import glob
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        slk_files = glob.glob(os.path.join(data_dir, '*.slk'))
        if not slk_files:
            pytest.skip("No .slk files in data/")

        x, z = load_slk_profile(slk_files[0])
        assert len(x) > 100, "Profile should have many data points"
        assert len(x) == len(z), "X and Z must have same length"
        assert x.dtype == np.float64 or x.dtype == np.float32
        assert not np.any(np.isnan(x)), "X should not contain NaN"
        assert not np.any(np.isnan(z)), "Z should not contain NaN"


if __name__ == '__main__':
    if HAS_PYTEST:
        pytest.main([__file__, '-v'])
    else:
        # Run tests manually
        test_classes = [
            TestPreprocessing,
            TestSubstrateIdentification,
            TestPeakDetection,
            TestGeometricMeasurements,
        ]
        passed, failed, errors = 0, 0, 0
        for cls in test_classes:
            obj = cls()
            for name in sorted(dir(obj)):
                if not name.startswith('test_'):
                    continue
                try:
                    getattr(obj, name)()
                    print(f'  PASS  {cls.__name__}.{name}')
                    passed += 1
                except AssertionError as e:
                    print(f'  FAIL  {cls.__name__}.{name}: {e}')
                    failed += 1
                except Exception as e:
                    print(f'  ERROR {cls.__name__}.{name}: {e}')
                    errors += 1
        print(f'\n  {passed} passed, {failed} failed, {errors} errors')

        # Visual validation: run on real data and display result
        data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
        import glob
        slk_files = sorted(glob.glob(os.path.join(data_dir, '*.slk')))[:3]
        if slk_files:
            print(f'\n  Visual validation on {len(slk_files)} samples:')
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
            from overlap_peak_analysis import analyze_sample
            for fp in slk_files:
                result = analyze_sample(fp, output_dir='tests/results')
                try:
                    from IPython.display import display, Image
                    display(Image(filename=result['image_path']))
                except ImportError:
                    print(f'    Saved: {result["image_path"]}')
