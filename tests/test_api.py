"""Tests for FastAPI deployment server."""

import numpy as np
import pytest
import tempfile
import os

from fastapi.testclient import TestClient

from src.api import app, set_model_path
from src.agent import DQNAgent
from src.environment import STATE_DIM


@pytest.fixture
def trained_model_path():
    """Create a temporary trained model for testing."""
    agent = DQNAgent(seed=42, device="cpu")
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    agent.save(path)
    yield path
    os.unlink(path)


@pytest.fixture
def client(trained_model_path):
    """Create test client with a trained model."""
    set_model_path(trained_model_path)
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True


class TestPredictEndpoint:
    def test_predict_valid_input(self, client):
        commit_data = {
            "diff_size": 0.3,
            "developer_id": 0.5,
            "file_types_modified": 0.2,
            "historical_defect_rate": 0.1,
            "prior_test_pass_rate": 0.9,
            "time_since_last_commit": 0.4,
            "num_files_changed": 0.2,
            "is_merge_commit": 0.0,
            "branch_depth": 0.1,
            "code_complexity": 0.3,
        }
        response = client.post("/predict", json=commit_data)
        assert response.status_code == 200
        data = response.json()
        assert data["action"] in ["full_test", "partial_test", "skip_test"]
        assert data["action_id"] in [0, 1, 2]
        assert len(data["q_values"]) == 3
        assert 0 <= data["confidence"] <= 1

    def test_predict_low_risk_commit(self, client):
        """Low-risk commit: small diff, high pass rate."""
        commit_data = {
            "diff_size": 0.05,
            "developer_id": 0.5,
            "file_types_modified": 0.1,
            "historical_defect_rate": 0.05,
            "prior_test_pass_rate": 0.95,
            "time_since_last_commit": 0.2,
            "num_files_changed": 0.05,
            "is_merge_commit": 0.0,
            "branch_depth": 0.1,
            "code_complexity": 0.05,
        }
        response = client.post("/predict", json=commit_data)
        assert response.status_code == 200

    def test_predict_high_risk_commit(self, client):
        """High-risk commit: large diff, low pass rate."""
        commit_data = {
            "diff_size": 0.9,
            "developer_id": 0.5,
            "file_types_modified": 0.8,
            "historical_defect_rate": 0.5,
            "prior_test_pass_rate": 0.3,
            "time_since_last_commit": 0.9,
            "num_files_changed": 0.8,
            "is_merge_commit": 1.0,
            "branch_depth": 0.8,
            "code_complexity": 0.9,
        }
        response = client.post("/predict", json=commit_data)
        assert response.status_code == 200

    def test_predict_invalid_input(self, client):
        """Values outside [0, 1] should fail validation."""
        commit_data = {
            "diff_size": 1.5,  # Invalid: > 1
            "developer_id": 0.5,
            "file_types_modified": 0.2,
            "historical_defect_rate": 0.1,
            "prior_test_pass_rate": 0.9,
            "time_since_last_commit": 0.4,
            "num_files_changed": 0.2,
            "is_merge_commit": 0.0,
            "branch_depth": 0.1,
            "code_complexity": 0.3,
        }
        response = client.post("/predict", json=commit_data)
        assert response.status_code == 422

    def test_predict_missing_field(self, client):
        commit_data = {
            "diff_size": 0.3,
            # Missing other fields
        }
        response = client.post("/predict", json=commit_data)
        assert response.status_code == 422


class TestBatchPredictEndpoint:
    def test_batch_predict(self, client):
        commits = [
            {
                "diff_size": 0.1 * i,
                "developer_id": 0.5,
                "file_types_modified": 0.2,
                "historical_defect_rate": 0.1,
                "prior_test_pass_rate": 0.9,
                "time_since_last_commit": 0.4,
                "num_files_changed": 0.2,
                "is_merge_commit": 0.0,
                "branch_depth": 0.1,
                "code_complexity": 0.1 * i,
            }
            for i in range(1, 6)
        ]
        response = client.post("/batch_predict", json=commits)
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 5

    def test_batch_predict_empty(self, client):
        response = client.post("/batch_predict", json=[])
        assert response.status_code == 200
        assert response.json() == []
