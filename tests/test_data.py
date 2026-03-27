"""
Tests for data layer modules.
"""

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError


class TestSchemas:
    """Tests for hp_ai_engine.data.schemas"""

    def test_dispensing_record_valid(self):
        from hp_ai_engine.data.schemas import DispensingRecord

        record = DispensingRecord(
            station_id="ST_0001",
            timestamp="2024-01-01T10:00:00",
            volume_kg=25.5,
            dispenser_id="D1",
            transaction_type="retail",
        )
        assert record.volume_kg == 25.5

    def test_dispensing_record_rejects_excessive_volume(self):
        from hp_ai_engine.data.schemas import DispensingRecord

        with pytest.raises(ValidationError):
            DispensingRecord(
                station_id="ST_0001",
                timestamp="2024-01-01T10:00:00",
                volume_kg=600,  # exceeds 500 kg limit
                dispenser_id="D1",
                transaction_type="retail",
            )

    def test_vehicle_mix_validation(self):
        from hp_ai_engine.data.schemas import VehiclePopulation

        # Valid mix summing to ~1.0
        vp = VehiclePopulation(
            catchment_id="CA_001",
            date="2024-01-01",
            total_cng_vehicles=10000,
            new_registrations_month=100,
            vehicle_mix={"auto": 0.4, "bus": 0.2, "car": 0.3, "taxi": 0.1},
        )
        assert sum(vp.vehicle_mix.values()) == pytest.approx(1.0, abs=0.05)

    def test_vehicle_mix_rejects_bad_sum(self):
        from hp_ai_engine.data.schemas import VehiclePopulation

        with pytest.raises(ValidationError):
            VehiclePopulation(
                catchment_id="CA_001",
                date="2024-01-01",
                total_cng_vehicles=10000,
                new_registrations_month=100,
                vehicle_mix={"auto": 0.1, "bus": 0.1},  # sums to 0.2
            )

    def test_station_meta_valid(self):
        from hp_ai_engine.data.schemas import StationMeta

        sm = StationMeta(
            station_id="ST_0001",
            latitude=19.076,
            longitude=72.877,
            station_type="retail",
            dispenser_count=4,
            storage_capacity_kg=5000,
            catchment_id="CA_001",
            city_cluster="Mumbai_West",
        )
        assert sm.station_id == "ST_0001"


class TestSyntheticGeneration:
    """Tests for hp_ai_engine.data.synthetic"""

    def test_generate_stations(self):
        from hp_ai_engine.data.synthetic import generate_stations

        df = generate_stations(num_stations=10, seed=42)
        assert len(df) == 10
        assert "station_id" in df.columns
        assert "latitude" in df.columns

    def test_generate_dispensing(self):
        from hp_ai_engine.data.synthetic import generate_dispensing, generate_stations

        stations = generate_stations(num_stations=3, seed=42)
        dispensing = generate_dispensing(stations, num_days=7, seed=42)
        assert len(dispensing) == 3 * 7 * 24  # 3 stations × 7 days × 24 hours
        assert (dispensing["volume_kg"] >= 0).all()

    def test_synthetic_data_generator(self):
        from hp_ai_engine.data.synthetic import SyntheticDataGenerator

        gen = SyntheticDataGenerator(num_stations=5, num_days=7, seed=42)
        data = gen.generate_all()
        assert "stations" in data
        assert "dispensing" in data
        assert len(data["stations"]) == 5


class TestFeatureEngineering:
    """Tests for hp_ai_engine.data.feature_engineering"""

    def test_lag_features(self):
        from hp_ai_engine.data.feature_engineering import add_lag_features

        df = pd.DataFrame({
            "station_id": ["ST_0001"] * 200,
            "timestamp": pd.date_range("2024-01-01", periods=200, freq="h"),
            "volume_kg": np.random.rand(200) * 100,
        })
        result = add_lag_features(df, lags=[1, 24])
        assert "volume_kg_lag_1h" in result.columns
        assert "volume_kg_lag_24h" in result.columns

    def test_rolling_features(self):
        from hp_ai_engine.data.feature_engineering import add_rolling_features

        df = pd.DataFrame({
            "station_id": ["ST_0001"] * 100,
            "volume_kg": np.random.rand(100) * 100,
        })
        result = add_rolling_features(df, windows=[6, 24])
        assert "volume_kg_rolling_mean_6h" in result.columns
        assert "volume_kg_rolling_std_24h" in result.columns


class TestGraphBuilder:
    """Tests for hp_ai_engine.data.graph_builder"""

    def test_build_static_graph(self):
        from hp_ai_engine.data.graph_builder import StationGraphBuilder

        builder = StationGraphBuilder(max_distance_km=50, use_road_distance=False)
        station_ids = ["ST_0001", "ST_0002", "ST_0003"]
        coords = [(19.0, 72.8), (19.01, 72.81), (19.5, 73.5)]

        graph = builder.build_static_graph(station_ids, coords)
        assert graph.num_nodes == 3
        assert graph.edge_index.shape[0] == 2  # 2 rows (source, target)

    def test_add_and_remove_node(self):
        from hp_ai_engine.data.graph_builder import StationGraphBuilder

        builder = StationGraphBuilder(max_distance_km=100, use_road_distance=False)
        station_ids = ["ST_0001", "ST_0002"]
        coords = [(19.0, 72.8), (19.01, 72.81)]

        builder.build_static_graph(station_ids, coords)
        assert builder.graph.num_nodes == 2

        # Add node
        builder.add_node("ST_0003", (19.02, 72.82))
        assert builder.graph.num_nodes == 3

        # Remove node
        builder.remove_node("ST_0002")
        assert builder.graph.num_nodes == 2
        assert "ST_0002" not in builder.station_ids
