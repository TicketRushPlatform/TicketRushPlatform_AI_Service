import unittest

from src.main import find_movie_for_track, extract_songfinder_track


class MainApiHelpersTests(unittest.TestCase):
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
