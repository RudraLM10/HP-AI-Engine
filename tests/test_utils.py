"""
Tests for utility modules.
"""

import numpy as np
import pytest


class TestMetrics:
    """Tests for hp_ai_engine.utils.metrics"""

    def test_mape_basic(self):
        from hp_ai_engine.utils.metrics import mape

        actual = np.array([100, 200, 300])
        predicted = np.array([110, 190, 310])
        result = mape(actual, predicted)
        assert isinstance(result, float)
        assert result > 0

    def test_rmse_basic(self):
        from hp_ai_engine.utils.metrics import rmse

        actual = np.array([100, 200, 300])
        predicted = np.array([100, 200, 300])
        assert rmse(actual, predicted) == pytest.approx(0.0, abs=1e-6)

    def test_mae_basic(self):
        from hp_ai_engine.utils.metrics import mae

        actual = np.array([100, 200, 300])
        predicted = np.array([110, 210, 310])
        assert mae(actual, predicted) == pytest.approx(10.0, abs=1e-6)


class TestGeo:
    """Tests for hp_ai_engine.utils.geo"""

    def test_haversine_same_point(self):
        from hp_ai_engine.utils.geo import haversine

        dist = haversine(19.0, 72.8, 19.0, 72.8)
        assert dist == pytest.approx(0.0, abs=1e-6)

    def test_haversine_known_distance(self):
        from hp_ai_engine.utils.geo import haversine

        # Mumbai to Pune (~150 km)
        dist = haversine(19.076, 72.877, 18.520, 73.856)
        assert 100 < dist < 200  # rough check

    def test_haversine_matrix_shape(self):
        from hp_ai_engine.utils.geo import haversine_matrix

        coords = [(19.0, 72.8), (18.5, 73.8), (19.1, 72.9)]
        matrix = haversine_matrix(coords)
        assert matrix.shape == (3, 3)
        assert np.allclose(np.diag(matrix), 0)


class TestTimeUtils:
    """Tests for hp_ai_engine.utils.time_utils"""

    def test_time_bin(self):
        from hp_ai_engine.utils.time_utils import time_to_bin

        assert time_to_bin(8) == "morning_peak"
        assert time_to_bin(18) == "evening_peak"
        assert time_to_bin(3) == "night"

    def test_cyclical_encoding_range(self):
        from hp_ai_engine.utils.time_utils import cyclical_encode

        sin_val, cos_val = cyclical_encode(12, 24)
        assert -1 <= sin_val <= 1
        assert -1 <= cos_val <= 1

    def test_cyclical_encode_boundaries(self):
        from hp_ai_engine.utils.time_utils import cyclical_encode

        sin_0, cos_0 = cyclical_encode(0, 24)
        sin_24, cos_24 = cyclical_encode(24, 24)
        # 0 and 24 should produce same values (cyclic)
        assert sin_0 == pytest.approx(sin_24, abs=1e-6)
        assert cos_0 == pytest.approx(cos_24, abs=1e-6)


class TestConfig:
    """Tests for hp_ai_engine.utils.config"""

    def test_load_config_returns_namespace(self, tmp_path):
        import yaml
        from hp_ai_engine.utils.config import load_config

        config_file = tmp_path / "test.yaml"
        config_file.write_text(yaml.dump({"model": {"hidden_dim": 128}}))

        cfg = load_config(str(config_file))
        assert cfg.model.hidden_dim == 128
