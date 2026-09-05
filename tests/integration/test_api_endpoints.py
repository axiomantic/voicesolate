import re
import pytest
from pathlib import Path
from starlette.testclient import TestClient
from voicesolate.server.api import app

@pytest.fixture
def client():
    return TestClient(app)

@pytest.mark.integration
class TestApiEndpoints:
    def test_system_status_contract(self, client: TestClient):
        response = client.get("/api/v1/system/status")
        assert response.status_code == 200
        data = response.json()

        # Contract assertion: check types and valid ranges, not just presence
        assert data["os"] in ("darwin", "linux", "win32")
        assert data["device"] in ("mps", "cuda", "cpu")
        assert re.match(r"^3\.\d+\.\d+", data["python_version"])
        assert re.match(r"^\d+\.\d+", data["torch_version"])

    def test_system_engines_complete_schema(self, client: TestClient):
        response = client.get("/api/v1/system/engines")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 4

        required_keys = {"id", "name", "architecture", "installed", "trained", "ready"}
        for eng in data:
            assert required_keys.issubset(eng.keys()), f"Missing schema keys in engine: {eng}"
            assert isinstance(eng["installed"], bool)
            assert isinstance(eng["trained"], bool)
            assert isinstance(eng["ready"], bool)
            assert len(eng["name"]) > 0

        engine_ids = [e["id"] for e in data]
        assert "f5-tts" in engine_ids
        assert "piper" in engine_ids
        assert "xtts-v2" in engine_ids
        assert "kokoro" in engine_ids

    def test_detect_script_endpoint_and_negative_control(self, client: TestClient):
        # 1. Positive case
        response = client.get("/api/v1/scripts/detect?filename=Star_Trek_The_Next_Generation_S06E01_Times_Arrow.mkv")
        assert response.status_code == 200
        data = response.json()
        assert data["detected_episode"] == "s06e01"
        assert isinstance(data["characters"], list)
        assert len(data["characters"]) > 0

        first_char = data["characters"][0]
        assert "name" in first_char
        assert "lines" in first_char
        assert isinstance(first_char["lines"], int)
        assert first_char["lines"] > 0

        # 2. Negative control: Missing required parameter returns 422
        bad_response = client.get("/api/v1/scripts/detect")
        assert bad_response.status_code == 422

    def test_episodes_list_endpoint_contract(self, client: TestClient):
        response = client.get("/api/v1/episodes")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        for ep in data:
            assert "id" in ep
            assert "name" in ep
            assert "clips_count" in ep
            assert isinstance(ep["clips_count"], int)

    def test_character_details_endpoint_and_negative_control(self, client: TestClient):
        # 1. Known character
        response = client.get("/api/v1/characters/CLEMENS/details")
        assert response.status_code == 200
        data = response.json()
        assert data["character_name"] == "CLEMENS"
        assert isinstance(data["engines"], list)
        assert isinstance(data["dataset_stats"], dict)
        assert "clip_count" in data["dataset_stats"]
        assert isinstance(data["dataset_stats"]["clip_count"], int)
        assert isinstance(data["cached_syntheses"], list)
        if len(data["cached_syntheses"]) > 0:
            for synth in data["cached_syntheses"]:
                assert "engine" in synth and synth["engine"]
                assert "engine_display" in synth and synth["engine_display"]
                assert "model_name" in synth and synth["model_name"]
                assert "model_architecture" in synth and synth["model_architecture"]
                assert "duration" in synth and isinstance(synth["duration"], (int, float))
                assert "samplerate" in synth and isinstance(synth["samplerate"], int)
                assert "url" in synth and synth["url"].startswith("/api/v1/audio/stream")
                assert "speed" in synth and isinstance(synth["speed"], (int, float))
                assert "cfg_strength" in synth and isinstance(synth["cfg_strength"], (int, float))
                assert "nfe_step" in synth and isinstance(synth["nfe_step"], int)
                assert "seed" in synth and isinstance(synth["seed"], int)
                assert "text" in synth
                assert "Previous Session" not in synth["model_name"]

        # 2. Negative control: Nonexistent character must return 200 with empty stats, not crash with 500
        ghost_res = client.get("/api/v1/characters/NONEXISTENT_GHOST_CHARACTER_999/details")
        assert ghost_res.status_code == 200
        ghost_data = ghost_res.json()
        assert ghost_data["character_name"] == "NONEXISTENT_GHOST_CHARACTER_999"
        assert ghost_data["dataset_stats"]["clip_count"] == 0
        assert ghost_data["quotes"] == []

    def test_delete_synthesis_contract_and_negative_control(self, client: TestClient, tmp_path):
        from pathlib import Path
        cache_dir = Path("cache/synthesized").resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        test_wav = cache_dir / "test_synth_delete_sample.wav"
        test_json = cache_dir / "test_synth_delete_sample.json"

        # Create dummy sample
        test_wav.write_bytes(b"RIFFdummywavdata")
        test_json.write_text('{"synth_id": "test_synth_delete_sample"}', encoding="utf-8")
        assert test_wav.exists()
        assert test_json.exists()

        # 1. Positive deletion case
        del_res = client.post("/api/v1/synthesis/delete", json={"synth_id": "test_synth_delete_sample"})
        assert del_res.status_code == 200
        del_data = del_res.json()
        assert del_data["status"] == "deleted"
        assert not test_wav.exists()
        assert not test_json.exists()

        # 2. Idempotent / not_found case
        del_res2 = client.post("/api/v1/synthesis/delete", json={"synth_id": "test_synth_delete_sample"})
        assert del_res2.status_code == 200
        assert del_res2.json()["status"] == "not_found"

        # 3. Negative control: Empty body returns 400
        bad_del = client.post("/api/v1/synthesis/delete", json={})
        assert bad_del.status_code == 400

    def test_piper_character_model_and_baseline_resolution(self, client: TestClient):
        response = client.get("/api/v1/characters/CLEMENS/details")
        assert response.status_code == 200
        data = response.json()
        engines = {e["id"]: e for e in data.get("engines", [])}
        assert "piper" in engines
        piper = engines["piper"]
        assert piper["installed"] is True
        assert piper["is_baseline"] is False
        if piper["trained"]:
            assert piper["model_path"] is not None
            assert piper["model_path"].endswith(".onnx")
        else:
            assert piper["model_path"] is None

        # Verify real character models are ready
        assert "f5-tts" in engines
        assert engines["f5-tts"]["trained"] is True
        assert engines["f5-tts"]["ready"] is True

        assert "xtts-v2" in engines
        assert engines["xtts-v2"]["installed"] is True

    def test_untrained_piper_rejection_no_bryce_fallback(self, client: TestClient):
        # Negative control: Synthesis for character without Piper model must reject and never fallback to Bryce
        res = client.post(
            "/api/v1/synthesize",
            json={
                "character_name": "NONEXISTENT_UNTRAINED_VOICE",
                "engine": "piper",
                "text": "Testing fallback elimination."
            }
        )
        assert res.status_code == 404 or res.status_code == 400
        detail = res.json().get("detail", "")
        assert "not found" in detail.lower() or "not been trained" in detail.lower()

    def test_kokoro_model_training_and_synthesis(self, client: TestClient):
        # 1. Train / generate Kokoro style profile for CLEMENS
        train_res = client.post(
            "/api/v1/training/train",
            json={
                "character_name": "CLEMENS",
                "engine": "kokoro",
                "epochs": 1
            }
        )
        assert train_res.status_code == 200
        job_id = train_res.json()["job_id"]
        assert job_id is not None

        # Verify engine status reports Kokoro as ready
        eng_res = client.get("/api/v1/system/engines?character=CLEMENS")
        assert eng_res.status_code == 200
        eng_data = {e["id"]: e for e in eng_res.json()}
        assert "kokoro" in eng_data
        assert eng_data["kokoro"]["ready"] is True
        assert "Kokoro-82M" in eng_data["kokoro"]["architecture"]

        # 2. Synthesize using Kokoro engine
        synth_res = client.post(
            "/api/v1/synthesize",
            json={
                "character_name": "CLEMENS",
                "engine": "kokoro",
                "text": "The secret of getting ahead is getting started.",
                "speed": 0.95,
                "voice_preset": "character_custom"
            }
        )
        assert synth_res.status_code == 200
        synth_data = synth_res.json()
        assert synth_data["engine"] == "kokoro"
        assert synth_data["samplerate"] == 24000
        assert synth_data["duration"] > 0.5
        assert Path(synth_data["file_path"]).exists()



