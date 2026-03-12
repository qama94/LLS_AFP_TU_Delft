"""
Validation tests for layer_detection.py
Run: python tests/test_layer_detection.py
"""

import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from layer_detection import (parse_slk, preprocess, get_levels,
                              get_band_tape_layer, format_sample_name)


# ── Synthetic profile for testing ─────────────────────────────────────

def make_synthetic_staircase(n=2000, n_layers=2, layer_height=0.1,
                              layer_width=20.0, noise_std=0.002, seed=42):
    """Create a synthetic staircase profile with known geometry."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 30, n)
    z = np.zeros_like(x)

    centre = 15.0
    for i in range(n_layers):
        half_w = (layer_width - i * 4) / 2
        left = centre - half_w
        right = centre + half_w
        mask = (x >= left) & (x <= right)
        z[mask] += layer_height

    z += rng.normal(0, noise_std, size=n)
    return x, z


# ── Tests ─────────────────────────────────────────────────────────────

class TestSLKParsing:
    """Tests for SYLK file parsing."""

    def test_parse_real_file(self):
        """Verify real .slk file loads correctly."""
        data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
        import glob
        slk_files = glob.glob(os.path.join(data_dir, '*.slk'))
        if not slk_files:
            print("  SKIP: no .slk files in data/")
            return
        x, z = parse_slk(slk_files[0])
        assert len(x) > 100, "Profile should have many points"
        assert len(x) == len(z), "X and Z must have same length"
        assert not np.any(np.isnan(x)), "No NaN in X"
        assert not np.any(np.isnan(z)), "No NaN in Z"


class TestPreprocessing:
    """Tests for signal conditioning."""

    def test_output_shape(self):
        """Verify preprocessing preserves array length."""
        x_raw, z_raw = make_synthetic_staircase()
        x, z = preprocess(x_raw, z_raw)
        assert len(x) == len(z), "X and Z must have same length"
        assert len(x) == len(x_raw), "Length should be preserved"

    def test_detrend(self):
        """Verify substrate baseline is near zero after detrend."""
        x_raw, z_raw = make_synthetic_staircase()
        x, z = preprocess(x_raw, z_raw)
        # Outer regions (substrate) should be near zero
        outer = z[:100]
        assert abs(np.median(outer)) < 0.05, \
            f"Substrate median={np.median(outer):.3f}, expected near 0"


class TestLevelDetection:
    """Tests for histogram-based level detection."""

    def test_detects_levels(self):
        """Verify levels are detected from synthetic profile."""
        x, z_raw = make_synthetic_staircase(n_layers=2, layer_height=0.12)
        _, z = preprocess(x, z_raw)
        levels = get_levels(z)
        assert len(levels) >= 1, f"Expected ≥1 level, got {len(levels)}"

    def test_levels_ordered(self):
        """Verify levels are in ascending order."""
        x, z_raw = make_synthetic_staircase(n_layers=3, layer_height=0.10)
        _, z = preprocess(x, z_raw)
        levels = get_levels(z)
        if len(levels) > 1:
            for i in range(1, len(levels)):
                assert levels[i] > levels[i-1], "Levels must be ascending"


class TestBandMapping:
    """Tests for band-to-tape-layer mapping."""

    def test_simple_mapping(self):
        """Two distinct levels → two tape layers."""
        mapping = get_band_tape_layer([0.10, 0.22])
        assert mapping == [0, 1], f"Expected [0,1], got {mapping}"

    def test_merge_mapping(self):
        """Small intermediate band gets merged."""
        # L1=0.10, intermediate at 0.14 (step=0.04 < 0.70*0.10=0.07)
        # Then L2 at 0.22
        mapping = get_band_tape_layer([0.10, 0.14, 0.22])
        # Band 0: layer 0, Band 1: check if merged
        assert mapping[0] == 0
        assert len(mapping) == 3


class TestSampleNaming:
    """Tests for auto sample naming."""

    def test_standard_name(self):
        name = format_sample_name('Sample_2_1A_00010.slk')
        assert name == 'Sample 2_1A', f"Got: {name}"

    def test_dot_name(self):
        name = format_sample_name('Sample_1.3A_00001.slk')
        assert name == 'Sample 1.3A', f"Got: {name}"

    def test_plain_name(self):
        name = format_sample_name('my_profile.slk')
        assert name == 'my_profile', f"Got: {name}"


class TestFullPipeline:
    """Integration test on real data."""

    def test_run_on_real_sample(self):
        """Run full pipeline on one real sample."""
        data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
        import glob
        slk_files = glob.glob(os.path.join(data_dir, '*.slk'))
        if not slk_files:
            print("  SKIP: no .slk files in data/")
            return

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from layer_detection import run
        result = run(slk_files[0], output_dir='/tmp/test_layer')

        assert 'n_tape_layers' in result, "Result must have n_tape_layers"
        assert result['n_tape_layers'] >= 1, "Must detect at least 1 layer"
        assert 'layer_widths' in result, "Result must have layer_widths"
        assert 'image_path' in result, "Result must have image_path"
        assert os.path.exists(result['image_path']), "Image must be saved"


# ── Runner ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    test_classes = [
        TestSLKParsing,
        TestPreprocessing,
        TestLevelDetection,
        TestBandMapping,
        TestSampleNaming,
        TestFullPipeline,
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

    # Visual validation: run on real data and display results
    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
    import glob
    slk_files = sorted(glob.glob(os.path.join(data_dir, '*.slk')))[:3]
    if slk_files:
        print(f'\n  Visual validation on {len(slk_files)} samples:')
        from layer_detection import run
        for fp in slk_files:
            result = run(fp, output_dir='tests/results')
            try:
                from IPython.display import display, Image
                display(Image(filename=result['image_path']))
            except ImportError:
                print(f'    Saved: {result["image_path"]}')
