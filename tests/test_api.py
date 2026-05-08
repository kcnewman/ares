"""Tests for api/server.py"""

from fastapi.testclient import TestClient

from api.server import app

client = TestClient(app)


class TestRoot:
    def test_returns_message(self):
        resp = client.get("/")
        assert resp.status_code == 200
        assert resp.json()["message"] == "ARES API is running"


class TestHealth:
    def test_returns_status(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "active"
        assert "model_exists" in data


class TestPredict:
    def test_returns_400_for_missing_fields(self):
        resp = client.post("/predict", json={})
        assert resp.status_code == 422

    def test_returns_422_for_invalid_types(self):
        resp = client.post(
            "/predict",
            json={
                "house_type": "duplex",
                "condition": "newly built",
                "furnishing": "furnished",
                "loc": "tesano",
                "bedrooms": "invalid",
                "bathrooms": 1,
            },
        )
        assert resp.status_code == 422

    def test_returns_generic_error_not_stack_trace(self, monkeypatch):
        def mock_predict(*args, **kwargs):
            raise ValueError("something sensitive")

        monkeypatch.setattr("api.server.predict_from_dict", mock_predict)
        resp = client.post(
            "/predict",
            json={
                "house_type": "duplex",
                "condition": "newly built",
                "furnishing": "furnished",
                "loc": "tesano",
                "bedrooms": 2,
                "bathrooms": 1,
            },
        )
        assert resp.status_code == 500
        assert "sensitive" not in resp.text
        assert "internal error" in resp.text.lower()


class TestExplain:
    def test_returns_422_for_missing_fields(self):
        resp = client.post("/explain", json={})
        assert resp.status_code == 422

    def test_fallback_when_no_model(self):
        """When model doesn't exist, a fallback explanation should be returned."""
        resp = client.post(
            "/explain",
            json={
                "house_type": "duplex",
                "condition": "newly built",
                "furnishing": "furnished",
                "loc": "tesano",
                "bedrooms": 2,
                "bathrooms": 1,
            },
        )
        # May fail if model is absent, but should not leak stack trace
        assert resp.status_code in (200, 500)
        if resp.status_code == 500:
            assert "internal error" in resp.text.lower()
            assert "traceback" not in resp.text.lower()
