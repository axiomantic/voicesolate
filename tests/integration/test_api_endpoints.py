import pytest
from starlette.testclient import TestClient
from voicesolate.server.api import app

@pytest.fixture
def client():
    return TestClient(app)

@pytest.mark.integration
class TestApiEndpoints:
    def test_system_status(self, client: TestClient):
        response = client.get("/api/v1/system/status")
        assert response.status_code == 200
        data = response.json()
        assert "os" in data
        assert "device" in data
        assert "python_version" in data

    def test_system_engines(self, client: TestClient):
        response = client.get("/api/v1/system/engines")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        engine_ids = [e["id"] for e in data]
        assert "f5-tts" in engine_ids
        assert "piper" in engine_ids
        assert "xtts-v2" in engine_ids

    def test_detect_script_endpoint(self, client: TestClient):
        response = client.get("/api/v1/scripts/detect?filename=Star_Trek_The_Next_Generation_S06E01_Times_Arrow.mkv")
        assert response.status_code == 200
        data = response.json()
        assert data["detected_episode"] == "s06e01"
        assert "characters" in data
        assert isinstance(data["characters"], list)

    def test_episodes_list_endpoint(self, client: TestClient):
        response = client.get("/api/v1/episodes")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_character_details_endpoint(self, client: TestClient):
        response = client.get("/api/v1/characters/CLEMENS/details")
        assert response.status_code == 200
        data = response.json()
        assert data["character_name"] == "CLEMENS"
        assert "engines" in data
        assert "dataset_stats" in data
        assert "cached_syntheses" in data
