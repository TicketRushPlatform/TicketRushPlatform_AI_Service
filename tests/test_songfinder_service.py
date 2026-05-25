import unittest

import httpx

from src.song_recognition.songfinder_service import (
    SongFinderSettings,
    extract_songfinder_track,
    recognize_song_bytes,
)


class FakeSongFinderResponse:
    status_code = 200
    text = '{"track":{"title":"Saved Song"}}'

    def json(self):
        return {"track": {"title": "Saved Song"}}

    def raise_for_status(self):
        return None


class FakeSongFinderClient:
    last_content = None
    last_headers = None
    last_params = None

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def post(self, *args, **kwargs):
        FakeSongFinderClient.last_content = kwargs["content"]
        FakeSongFinderClient.last_headers = kwargs["headers"]
        FakeSongFinderClient.last_params = kwargs["params"]
        return FakeSongFinderResponse()


class SongFinderServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_recognize_song_bytes_calls_songfinder_and_llm_lookup(self):
        result = await recognize_song_bytes(
            b"RIFF....WAVEfmt ",
            start_time=7,
            settings=SongFinderSettings(
                rapidapi_key="test-key",
                rapidapi_host="songfinder.test",
                songfinder_url="https://songfinder.test/recognize",
            ),
            http_client_factory=FakeSongFinderClient,
            movie_lookup=lambda track: "Saved Movie",
        )

        self.assertEqual(
            result,
            {
                "success": True,
                "song": "Saved Song",
                "movie": "Saved Movie",
                "track": {"title": "Saved Song"},
            },
        )
        self.assertEqual(FakeSongFinderClient.last_content, b"RIFF....WAVEfmt ")
        self.assertEqual(FakeSongFinderClient.last_headers["x-rapidapi-key"], "test-key")
        self.assertEqual(FakeSongFinderClient.last_headers["x-rapidapi-host"], "songfinder.test")
        self.assertEqual(FakeSongFinderClient.last_params, {"startTime": 7})

    async def test_recognize_song_bytes_requires_rapidapi_key(self):
        with self.assertRaises(ValueError) as error:
            await recognize_song_bytes(
                b"audio",
                settings=SongFinderSettings(rapidapi_key=None),
                http_client_factory=FakeSongFinderClient,
            )

        self.assertIn("Missing RAPIDAPI_KEY", str(error.exception))

    async def test_recognize_song_bytes_maps_songfinder_http_error(self):
        class ErrorClient(FakeSongFinderClient):
            async def post(self, *args, **kwargs):
                request = httpx.Request("POST", "https://songfinder.test/recognize")
                return httpx.Response(429, text="rate limited", request=request)

        with self.assertRaises(httpx.HTTPStatusError) as error:
            await recognize_song_bytes(
                b"audio",
                settings=SongFinderSettings(rapidapi_key="test-key"),
                http_client_factory=ErrorClient,
            )

        self.assertEqual(error.exception.response.status_code, 429)

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


if __name__ == "__main__":
    unittest.main()
