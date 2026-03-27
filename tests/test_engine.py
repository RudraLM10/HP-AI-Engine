"""
Tests for model, training, optimiser, location, and scalability modules.
"""

import numpy as np
import torch
import pytest


class TestGCNEncoder:
    """Tests for hp_ai_engine.models.gcn_encoder"""

    def test_forward_shape(self):
        from hp_ai_engine.models.gcn_encoder import SpatialGCNEncoder

        encoder = SpatialGCNEncoder(in_channels=8, hidden_dim=32, num_layers=2)
        x = torch.randn(5, 8)  # 5 nodes, 8 features
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)

        output = encoder(x, edge_index)
        assert output.shape == (5, 32)

    def test_embedding_dim(self):
        from hp_ai_engine.models.gcn_encoder import SpatialGCNEncoder

        encoder = SpatialGCNEncoder(in_channels=4, hidden_dim=64)
        assert encoder.get_embedding_dim() == 64


class TestTFTModel:
    """Tests for hp_ai_engine.models.tft_model"""

    def test_grn_output_dim(self):
        from hp_ai_engine.models.tft_model import GatedResidualNetwork

        grn = GatedResidualNetwork(input_dim=32, hidden_dim=64, output_dim=32)
        x = torch.randn(4, 32)
        output = grn(x)
        assert output.shape == (4, 32)

    def test_tft_core_output_shapes(self):
        from hp_ai_engine.models.tft_model import TFTCore

        tft = TFTCore(
            num_static_features=2,
            num_dynamic_features=8,
            num_future_features=8,
            spatial_embedding_dim=32,
            hidden_size=64,
            short_horizon=6,
            mid_horizon=24,
            long_horizon=48,
            lookback_hours=24,
        )

        batch_size = 4
        static = torch.randn(batch_size, 2)
        past = torch.randn(batch_size, 24, 8)
        future = torch.randn(batch_size, 48, 8)
        spatial = torch.randn(batch_size, 32)

        short, mid, long, info = tft(static, past, future, spatial)

        assert short.shape == (batch_size, 6)
        assert mid.shape == (batch_size, 24)
        assert long.shape == (batch_size, 48)
        assert "attention_weights" in info


class TestContextAttention:
    """Tests for hp_ai_engine.models.context_attention"""

    def test_no_override_below_threshold(self):
        from hp_ai_engine.models.context_attention import ContextAttentionOverride

        ctx = ContextAttentionOverride(num_signals=3, deviation_threshold=2.0)
        pred = torch.ones(2, 6) * 100

        # Signals within 1 std → no override
        current = torch.tensor([[10.0, 20.0, 30.0], [10.0, 20.0, 30.0]])
        means = torch.tensor([[10.0, 20.0, 30.0], [10.0, 20.0, 30.0]])
        stds = torch.tensor([[5.0, 5.0, 5.0], [5.0, 5.0, 5.0]])

        adjusted, info = ctx(pred, current, means, stds)
        assert not info["override_active"].any()
        assert torch.allclose(adjusted, pred)


class TestLoss:
    """Tests for hp_ai_engine.training.loss"""

    def test_huber_mape_loss(self):
        from hp_ai_engine.training.loss import HuberMAPELoss

        loss_fn = HuberMAPELoss()
        pred = torch.tensor([[100.0, 200.0, 300.0]])
        target = torch.tensor([[110.0, 190.0, 310.0]])

        loss = loss_fn(pred, target)
        assert loss.item() > 0

    def test_multi_horizon_loss(self):
        from hp_ai_engine.training.loss import MultiHorizonLoss

        loss_fn = MultiHorizonLoss()
        total, breakdown = loss_fn(
            pred_short=torch.randn(4, 6),
            pred_mid=torch.randn(4, 24),
            pred_long=torch.randn(4, 48),
            target_short=torch.randn(4, 6),
            target_mid=torch.randn(4, 24),
            target_long=torch.randn(4, 48),
        )
        assert total.item() > 0
        assert "loss_short" in breakdown


class TestDispenserRules:
    """Tests for hp_ai_engine.optimiser.dispenser_rules"""

    def test_high_utilisation_opens_dispensers(self):
        from hp_ai_engine.optimiser.dispenser_rules import (
            DispenserRuleEngine, StationState,
        )

        engine = DispenserRuleEngine()
        state = StationState(
            station_id="ST_0001",
            active_dispensers=2,
            total_dispensers=4,
            current_staff=2,
            current_inventory_kg=3000,
            storage_capacity_kg=5000,
            current_utilisation=0.9,
        )
        forecast = [200, 220, 250, 240, 230, 210]  # high demand

        actions = engine.evaluate(forecast, state)
        action_types = [a.action_type for a in actions]
        assert "open_dispenser" in action_types

    def test_low_inventory_requests_tanker(self):
        from hp_ai_engine.optimiser.dispenser_rules import (
            DispenserRuleEngine, StationState,
        )

        engine = DispenserRuleEngine()
        state = StationState(
            station_id="ST_0001",
            active_dispensers=2,
            total_dispensers=4,
            current_staff=2,
            current_inventory_kg=500,  # low
            storage_capacity_kg=5000,
            current_utilisation=0.3,
        )
        forecast = [50, 60, 55, 50, 45, 40]

        actions = engine.evaluate(forecast, state)
        action_types = [a.action_type for a in actions]
        assert "request_tanker" in action_types


class TestDriftDetection:
    """Tests for hp_ai_engine.scalability.drift_detection"""

    def test_psi_identical_distributions(self):
        from hp_ai_engine.scalability.drift_detection import DriftDetector

        detector = DriftDetector()
        data = np.random.randn(1000)
        psi = detector.compute_psi(data, data)
        assert psi < 0.01  # should be ~0 for identical distributions

    def test_psi_shifted_distribution(self):
        from hp_ai_engine.scalability.drift_detection import DriftDetector

        detector = DriftDetector()
        ref = np.random.randn(1000)
        shifted = ref + 3.0  # significant shift
        psi = detector.compute_psi(ref, shifted)
        assert psi > 0.25  # should trigger critical

    def test_concept_drift_cusum(self):
        from hp_ai_engine.scalability.drift_detection import DriftDetector

        detector = DriftDetector(cusum_threshold=5.0)
        # Biased residuals → concept drift
        residuals = np.ones(100) * 0.5  # model consistently under-predicts
        result = detector.check_concept_drift(residuals)
        assert result.is_drifted


class TestMDUDecision:
    """Tests for hp_ai_engine.location.mdu_decision"""

    def test_consistent_demand_recommends_permanent(self):
        from hp_ai_engine.location.mdu_decision import MDUDecisionEngine

        engine = MDUDecisionEngine(cv_threshold=0.2)
        # Stable demand: 100 kg/day ± 5%
        forecast = np.ones(4320) * 100 + np.random.randn(4320) * 5

        decision = engine.decide("SITE_001", forecast)
        assert decision.recommendation == "permanent"
        assert decision.cv_value < 0.2

    def test_variable_demand_recommends_mdu(self):
        from hp_ai_engine.location.mdu_decision import MDUDecisionEngine

        engine = MDUDecisionEngine(cv_threshold=0.2)
        # Highly variable demand: alternating months
        forecast = np.concatenate([
            np.ones(720) * 100,
            np.ones(720) * 20,
            np.ones(720) * 150,
            np.ones(720) * 30,
            np.ones(720) * 120,
            np.ones(720) * 25,
        ])

        decision = engine.decide("SITE_002", forecast)
        assert decision.recommendation == "mdu_first"
        assert decision.cv_value > 0.2


class TestSiteScoring:
    """Tests for hp_ai_engine.location.site_scoring"""

    def test_score_ranking(self):
        from hp_ai_engine.location.site_scoring import (
            CandidateSite, SiteScoringModel,
        )

        model = SiteScoringModel()
        candidates = [
            CandidateSite(
                site_id="A", location=(19.0, 72.8),
                vehicle_density=100, traffic_volume=5000,
                predicted_demand=500, competitor_distance_km=10,
                population_density=10000, income_bracket=3,
            ),
            CandidateSite(
                site_id="B", location=(19.1, 72.9),
                vehicle_density=50, traffic_volume=2000,
                predicted_demand=200, competitor_distance_km=2,
                population_density=5000, income_bracket=1,
            ),
        ]

        results = model.score_candidates(candidates)
        assert results[0].rank == 1
        assert results[0].total_score > results[1].total_score
