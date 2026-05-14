import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.midi_vector_store import (
    create_note_histogram,
    enrich_metadata_item,
    extract_lakh_ids,
    interval_sequence,
    load_store,
    normalize_melody_pitches,
    parse_title_artist_hint,
    rerank_by_sequence,
    save_store,
    search_embeddings,
)


class MidiVectorStoreTests(unittest.TestCase):
    def test_create_note_histogram_returns_normalized_128_dim_vector(self):
        embedding = create_note_histogram([60, 60, 64, 67])

        self.assertEqual(embedding.shape, (128,))
        self.assertAlmostEqual(float(np.linalg.norm(embedding)), 1.0, places=6)
        self.assertGreater(embedding[60], embedding[64])

    def test_search_embeddings_ranks_closest_cosine_match(self):
        embeddings = np.array(
            [
                create_note_histogram([60, 62, 64]),
                create_note_histogram([72, 74, 76]),
            ],
            dtype=np.float32,
        )

        results = search_embeddings(create_note_histogram([60, 62, 64]), embeddings, top_k=2)

        self.assertEqual(results[0].index, 0)
        self.assertGreater(results[0].score, results[1].score)

    def test_save_and_load_store_round_trips_embeddings_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            store_dir = Path(tmp)
            embeddings = np.array([create_note_histogram([60, 64, 67])], dtype=np.float32)
            metadata = [{"path": "song.mid", "note_count": 3}]

            save_store(store_dir, embeddings, metadata)
            loaded_embeddings, loaded_metadata = load_store(store_dir)

            np.testing.assert_allclose(loaded_embeddings, embeddings)
            self.assertEqual(loaded_metadata, metadata)

    def test_extract_lakh_ids_from_matched_path(self):
        path = r"data\lakh\lmd_matched\A\A\A\TRAAAGR128F425B14B\1d9d16a9da90c090809c153754823c2b.mid"

        ids = extract_lakh_ids(path)

        self.assertEqual(ids["track_id"], "TRAAAGR128F425B14B")
        self.assertEqual(ids["midi_md5"], "1d9d16a9da90c090809c153754823c2b")

    def test_enrich_metadata_item_keeps_existing_fields(self):
        item = {
            "path": r"data\lakh\lmd_matched\A\A\A\TRAAAGR128F425B14B\1d9d16a9da90c090809c153754823c2b.mid",
            "status": "indexed",
            "note_count": 3448,
        }

        enriched = enrich_metadata_item(item)

        self.assertEqual(enriched["status"], "indexed")
        self.assertEqual(enriched["note_count"], 3448)
        self.assertEqual(enriched["track_id"], "TRAAAGR128F425B14B")
        self.assertEqual(enriched["midi_md5"], "1d9d16a9da90c090809c153754823c2b")

    def test_interval_sequence_is_key_invariant(self):
        self.assertEqual(interval_sequence([60, 62, 64, 65]), interval_sequence([65, 67, 69, 70]))

    def test_normalize_melody_pitches_reduces_octave_harmonic_jumps(self):
        pitches = normalize_melody_pitches([54, 52, 93, 54, 54, 57])

        self.assertEqual(pitches, [54, 52, 57, 54, 57])
        self.assertNotIn(93, pitches)

    def test_parse_title_artist_hint(self):
        parsed = parse_title_artist_hint(["Fast Car - Tracy Chapman", "Melody"])

        self.assertEqual(parsed["title_hint"], "Fast Car")
        self.assertEqual(parsed["artist_hint"], "Tracy Chapman")

    def test_rerank_by_sequence_finds_short_query_inside_longer_song(self):
        query = interval_sequence([60, 62, 64, 65])
        candidates = [
            {
                "path": "wrong.mid",
                "interval_sequence": interval_sequence([60, 59, 57, 55, 54]),
            },
            {
                "path": "right.mid",
                "interval_sequence": interval_sequence([50, 51, 53, 55, 57, 58, 57]),
            },
        ]

        ranked = rerank_by_sequence(query, candidates, top_k=2)

        self.assertEqual(ranked[0].metadata["path"], "right.mid")
        self.assertGreater(ranked[0].score, ranked[1].score)


if __name__ == "__main__":
    unittest.main()
