import pytest
from pathlib import Path
from voicesolate.audio_extractor import AudioExtractor

@pytest.mark.unit
class TestAudioExtractor:
    def test_parse_local_file(self, temp_dir: Path, make_wav_file):
        test_wav = make_wav_file(temp_dir / "test_local.wav", duration_sec=1.0)
        extractor = AudioExtractor(str(test_wav))

        assert not extractor.is_remote
        assert extractor.local_path == test_wav.resolve()
        assert extractor.remote_host is None

    def test_parse_sftp_url(self):
        sftp_url = "sftp://elijah@flanopticon.lan/mnt/nas/media/downloads/Star%20Trek%20S06E01.mkv"
        extractor = AudioExtractor(sftp_url)

        assert extractor.is_remote is True
        assert extractor.remote_user == "elijah"
        assert extractor.remote_host == "flanopticon.lan"
        assert extractor.remote_file_path == "/mnt/nas/media/downloads/Star Trek S06E01.mkv"

    def test_parse_ssh_scp_format(self):
        ssh_path = "media_user@storage.server.local:/data/movies/film.mp4"
        extractor = AudioExtractor(ssh_path)

        assert extractor.is_remote is True
        assert extractor.remote_user == "media_user"
        assert extractor.remote_host == "storage.server.local"
        assert extractor.remote_file_path == "/data/movies/film.mp4"

    def test_missing_local_file_raises(self):
        with pytest.raises(FileNotFoundError):
            AudioExtractor("/nonexistent/path/to/random_media_file_12345.mkv")

    def test_remote_inferred_for_nas_path(self):
        # Paths starting with /mnt/ or containing flanopticon that don't exist locally
        # are inferred as remote
        nas_path = "/mnt/nas/media/episode.mkv"
        extractor = AudioExtractor(nas_path, ssh_user="admin")
        assert extractor.is_remote is True
        assert extractor.remote_host == "flanopticon.lan"
        assert extractor.remote_user == "admin"
