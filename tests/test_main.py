import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.main import app, extract_songfinder_track, find_movie_for_track


class FakeSongFinderResponse:
    status_code = 200
    text = '{"track":{"title":"Saved Song"}}'

    def json(self):
        return {"track": {"title": "Saved Song"}}


class FakeSongFinderClient:
    last_content = None

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def post(self, *args, **kwargs):
        FakeSongFinderClient.last_content = kwargs["content"]
        return FakeSongFinderResponse()


class MainApiHelpersTests(unittest.TestCase):
    def test_record_page_serves_microphone_wav_uploader(self):
        client = TestClient(app)

        response = client.get("/record")

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn("Start recording", response.text)
        self.assertIn("/api/recognize/songfinder?startTime=0", response.text)
        self.assertIn("audio/wav", response.text)
        self.assertIn("recording.wav", response.text)

    def test_recognize_song_saves_uploaded_audio_to_disk(self):
        audio_bytes = b"RIFF....WAVEfmt "

        with tempfile.TemporaryDirectory() as tmp:
            recordings_dir = Path(tmp)
            with (
                patch("src.main.RAPIDAPI_KEY", "test-key"),
                patch("src.main.RECORDINGS_DIR", recordings_dir),
                patch("src.main.httpx.AsyncClient", FakeSongFinderClient),
                patch("src.main.find_movie_for_track", return_value=None),
            ):
                client = TestClient(app)

                response = client.post(
                    "/api/recognize/songfinder?startTime=0",
                    files={"audio": ("recording.wav", audio_bytes, "audio/wav")},
                )

            self.assertEqual(response.status_code, 200)
            data = response.json()
            saved_audio_path = Path(data["saved_audio_path"])
            self.assertEqual(saved_audio_path.suffix, ".wav")
            self.assertEqual(saved_audio_path.read_bytes(), audio_bytes)

        self.assertEqual(FakeSongFinderClient.last_content, audio_bytes)

    def test_extract_songfinder_track_returns_expected_metadata(self):
        songfinder_data = {
            "success": True,
            "track": {
                "title": "Fast Car",
                "artist": "Tracy Chapman",
                "album": "Tracy Chapman",
                "releaseDate": "1988",
                "genre": "Singer/Songwriter",
                "label": "Elektra Records",
                "coverArt": "https://example.com/cover.jpg",
                "isrc": "USEE10180719",
            },
        }

        track = extract_songfinder_track(songfinder_data)

        self.assertEqual(
            track,
            {
                "title": "Fast Car",
                "artist": "Tracy Chapman",
                "album": "Tracy Chapman",
                "release_date": "1988",
                "genre": "Singer/Songwriter",
                "label": "Elektra Records",
                "cover_art": "https://example.com/cover.jpg",
                "isrc": "USEE10180719",
            },
        )

    def test_find_movie_for_track_returns_none_without_api_key(self):
        movie = find_movie_for_track(
            {
                "title": "Fast Car",
                "artist": "Tracy Chapman",
                "album": "Tracy Chapman",
                "release_date": "1988",
            },
            openai_api_key=None,
        )

        self.assertIsNone(movie)


if __name__ == "__main__":
    unittest.main()
