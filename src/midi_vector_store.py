import argparse
import json
import os
import sys
import tarfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np


LMD_FULL_URL = "http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz"
LMD_MATCHED_URL = "http://hog.ee.columbia.edu/craffel/lmd/lmd_matched.tar.gz"
EMBEDDINGS_FILE = "embeddings.npy"
METADATA_FILE = "metadata.json"
CONFIG_FILE = "config.json"
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
MELODY_NAME_HINTS = ("melody", "vocal", "lead", "theme", "solo")
GENERIC_MIDI_NAMES = {
    "bass",
    "drums",
    "guitar",
    "melody",
    "piano",
    "strings",
    "vocal line",
    "vocals",
}


@dataclass(frozen=True)
class SearchResult:
    index: int
    score: float
    metadata: dict | None = None


def normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def create_note_histogram(pitches: Sequence[int]) -> np.ndarray:
    histogram, _ = np.histogram(pitches, bins=np.arange(0, 129))
    return normalize(histogram.astype(np.float32))


def interval_sequence(pitches: Sequence[int], *, max_interval: int = 24) -> List[int]:
    intervals: List[int] = []
    for previous, current in zip(pitches, pitches[1:]):
        interval = int(current) - int(previous)
        if interval == 0:
            continue
        intervals.append(max(-max_interval, min(max_interval, interval)))
    return intervals


def normalize_melody_pitches(
    pitches: Sequence[int],
    *,
    min_midi: int = 36,
    max_midi: int = 84,
    max_jump: int = 12,
) -> List[int]:
    normalized: List[int] = []
    for raw_pitch in pitches:
        pitch = int(raw_pitch)
        octave_candidates = [
            pitch + 12 * shift
            for shift in range(-5, 6)
            if min_midi <= pitch + 12 * shift <= max_midi
        ]
        if not octave_candidates:
            continue

        if normalized:
            previous = normalized[-1]
            pitch = min(octave_candidates, key=lambda candidate: abs(candidate - previous))
            if abs(pitch - previous) > max_jump:
                continue
        else:
            pitch = min(octave_candidates, key=lambda candidate: abs(candidate - 60))

        if normalized and normalized[-1] == pitch:
            continue
        normalized.append(pitch)

    return normalized


def inspect_midi(midi_path: str | Path) -> dict:
    try:
        import pretty_midi
    except ImportError as exc:
        raise RuntimeError("pretty_midi is required to parse MIDI files") from exc

    midi = pretty_midi.PrettyMIDI(str(midi_path))
    names: List[str] = []
    instruments = []
    for instrument in midi.instruments:
        clean_name = (instrument.name or "").replace("\x00", "").strip()
        if clean_name and clean_name not in names:
            names.append(clean_name)
        instruments.append(
            {
                "name": clean_name,
                "program": int(instrument.program),
                "is_drum": bool(instrument.is_drum),
                "note_count": len(instrument.notes),
            }
        )

    return {
        "path": str(midi_path),
        **extract_lakh_ids(midi_path),
        **parse_title_artist_hint(names),
        "names": names,
        "instruments": instruments,
    }


def parse_title_artist_hint(names: Sequence[str]) -> dict:
    for name in names:
        clean = " ".join(str(name).replace("\x00", "").split())
        if not clean or "@" in clean:
            continue
        if clean.lower() in GENERIC_MIDI_NAMES:
            continue

        separators = [" - ", " by ", " / "]
        for separator in separators:
            if separator in clean:
                left, right = clean.split(separator, 1)
                left = left.strip(" ;")
                right = right.strip(" ;")
                if left and right:
                    return {"title_hint": left, "artist_hint": right}
        return {"title_hint": clean}

    return {}


def _instrument_name_looks_melodic(name: str) -> bool:
    lowered = name.lower()
    return any(hint in lowered for hint in MELODY_NAME_HINTS)


def _select_melody_notes(midi) -> List[dict]:
    instrument_candidates: List[Tuple[float, List[dict]]] = []

    for instrument in midi.instruments:
        if instrument.is_drum:
            continue

        notes_for_instrument: List[dict] = []
        for note in instrument.notes:
            notes_for_instrument.append(
                {
                    "pitch": int(note.pitch),
                    "start": float(note.start),
                    "end": float(note.end),
                    "duration": float(max(0.0, note.end - note.start)),
                    "velocity": int(note.velocity),
                }
            )

        if not notes_for_instrument:
            continue

        pitches = [note["pitch"] for note in notes_for_instrument]
        avg_pitch = sum(pitches) / len(pitches)
        melody_range_notes = sum(48 <= pitch <= 84 for pitch in pitches)
        melody_ratio = melody_range_notes / len(pitches)
        unique_pitches = len(set(pitches))
        name_bonus = 1000.0 if _instrument_name_looks_melodic(instrument.name or "") else 0.0
        pitch_bonus = 120.0 if 52 <= avg_pitch <= 78 else 0.0
        count_score = min(len(notes_for_instrument), 800) / 8.0
        variety_score = min(unique_pitches, 24) * 3.0
        score = name_bonus + pitch_bonus + count_score + variety_score + melody_ratio * 100.0
        instrument_candidates.append((score, notes_for_instrument))

    if not instrument_candidates:
        return []

    _, notes = max(instrument_candidates, key=lambda item: item[0])
    notes.sort(key=lambda item: (item["start"], -item["pitch"], -item["velocity"]))
    return notes


def melody_notes_from_midi(midi_path: str | Path, *, onset_bucket_s: float = 0.08) -> List[dict]:
    try:
        import pretty_midi
    except ImportError as exc:
        raise RuntimeError("pretty_midi is required to parse MIDI files") from exc

    midi = pretty_midi.PrettyMIDI(str(midi_path))
    notes = _select_melody_notes(midi)
    if not notes:
        return []

    compressed: List[dict] = []
    current_bucket = None
    best_note = None
    for note in notes:
        bucket = int(round(note["start"] / onset_bucket_s))
        if current_bucket is None:
            current_bucket = bucket
            best_note = note
            continue

        if bucket != current_bucket:
            if best_note is not None:
                compressed.append(best_note)
            current_bucket = bucket
            best_note = note
            continue

        if best_note is None or (note["pitch"], note["velocity"]) > (best_note["pitch"], best_note["velocity"]):
            best_note = note

    if best_note is not None:
        compressed.append(best_note)

    deduped: List[dict] = []
    for note in compressed:
        if deduped and deduped[-1]["pitch"] == note["pitch"]:
            continue
        deduped.append(note)
    return deduped


def melody_pitches_from_midi(midi_path: str | Path) -> List[int]:
    return [note["pitch"] for note in melody_notes_from_midi(midi_path)]


def extract_pitches_from_midi(midi_path: str | Path) -> List[int]:
    try:
        import pretty_midi
    except ImportError as exc:
        raise RuntimeError("pretty_midi is required to parse MIDI files") from exc

    midi = pretty_midi.PrettyMIDI(str(midi_path))
    pitches: List[int] = []
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            pitches.append(int(note.pitch))
    return pitches


def _local_alignment_score(query: Sequence[int], candidate: Sequence[int]) -> float:
    if not query or not candidate:
        return 0.0

    previous = [0.0] * (len(candidate) + 1)
    best = 0.0
    gap_penalty = -0.7

    for query_value in query:
        current = [0.0] * (len(candidate) + 1)
        for j, candidate_value in enumerate(candidate, start=1):
            distance = abs(int(query_value) - int(candidate_value))
            if distance == 0:
                match_score = 2.0
            elif distance == 1:
                match_score = 1.0
            else:
                match_score = -1.0

            current[j] = max(
                0.0,
                previous[j - 1] + match_score,
                previous[j] + gap_penalty,
                current[j - 1] + gap_penalty,
            )
            best = max(best, current[j])
        previous = current

    return best / (2.0 * len(query))


def rerank_by_sequence(query_intervals: Sequence[int], candidates: Sequence[dict], top_k: int = 5) -> List[SearchResult]:
    ranked: List[SearchResult] = []
    for index, candidate in enumerate(candidates):
        score = _local_alignment_score(query_intervals, candidate.get("interval_sequence", []))
        ranked.append(SearchResult(index=index, score=score, metadata=dict(candidate)))

    ranked.sort(key=lambda result: result.score, reverse=True)
    return ranked[:top_k]


def iter_midi_files(root: str | Path) -> Iterable[Path]:
    root_path = Path(root)
    yield from root_path.rglob("*.mid")
    yield from root_path.rglob("*.midi")


def extract_lakh_ids(midi_path: str | Path) -> dict:
    path = Path(midi_path)
    midi_md5 = path.stem
    track_id = None

    for part in reversed(path.parts):
        if part.startswith("TR") and len(part) == 18:
            track_id = part
            break

    ids = {"midi_md5": midi_md5}
    if track_id:
        ids["track_id"] = track_id
    return ids


def _sequence_metadata(midi_path: str | Path) -> dict:
    melody_pitches = normalize_melody_pitches(melody_pitches_from_midi(midi_path))
    return {
        "melody_pitch_sequence": melody_pitches,
        "interval_sequence": interval_sequence(melody_pitches),
    }


def enrich_metadata_item(item: dict, *, include_sequences: bool = False) -> dict:
    if "path" not in item:
        return item
    enriched = {**item, **extract_lakh_ids(item["path"])}
    if include_sequences and Path(item["path"]).exists():
        try:
            inspected = inspect_midi(item["path"])
            enriched = {
                **enriched,
                **_sequence_metadata(item["path"]),
                **{k: v for k, v in inspected.items() if k in {"title_hint", "artist_hint", "names"}},
            }
        except Exception as exc:
            enriched["sequence_error"] = f"{type(exc).__name__}: {exc}"
    return enriched


def enrich_metadata_file(store_dir: str | Path, *, include_sequences: bool = False) -> int:
    metadata_path = Path(store_dir) / METADATA_FILE
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    enriched_metadata = []
    for index, item in enumerate(metadata, start=1):
        enriched_metadata.append(enrich_metadata_item(item, include_sequences=include_sequences))
        if include_sequences and (index == len(metadata) or index % 100 == 0):
            _print_progress("Enrich", index, len(metadata))
    if include_sequences:
        print("", file=sys.stderr)

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(enriched_metadata, f, ensure_ascii=False, indent=2)

    return len(enriched_metadata)


def build_embeddings(
    midi_dir: str | Path,
    *,
    limit: int | None = None,
    min_notes: int = 8,
) -> Tuple[np.ndarray, List[dict]]:
    vectors: List[np.ndarray] = []
    metadata: List[dict] = []

    for midi_path in iter_midi_files(midi_dir):
        if limit is not None and len(vectors) >= limit:
            break

        try:
            pitches = extract_pitches_from_midi(midi_path)
        except Exception as exc:
            metadata.append(
                {
                    "path": str(midi_path),
                    "status": "skipped",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        if len(pitches) < min_notes:
            metadata.append(
                {
                    "path": str(midi_path),
                    "status": "skipped",
                    "reason": f"too_few_notes:{len(pitches)}",
                }
            )
            continue

        sequence_metadata = _sequence_metadata(midi_path)
        vectors.append(create_note_histogram(pitches))
        metadata.append(
            {
                "path": str(midi_path),
                "status": "indexed",
                "note_count": len(pitches),
                **extract_lakh_ids(midi_path),
                **sequence_metadata,
            }
        )

    if not vectors:
        return np.empty((0, 128), dtype=np.float32), metadata

    return np.vstack(vectors).astype(np.float32), metadata


def save_store(store_dir: str | Path, embeddings: np.ndarray, metadata: List[dict]) -> None:
    store_path = Path(store_dir)
    store_path.mkdir(parents=True, exist_ok=True)
    np.save(store_path / EMBEDDINGS_FILE, np.asarray(embeddings, dtype=np.float32))
    with (store_path / METADATA_FILE).open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    with (store_path / CONFIG_FILE).open("w", encoding="utf-8") as f:
        json.dump(
            {
                "embedding": "normalized_128_bin_midi_pitch_histogram",
                "metric": "cosine",
                "metadata_file": METADATA_FILE,
                "embeddings_file": EMBEDDINGS_FILE,
            },
            f,
            indent=2,
        )


def load_store(store_dir: str | Path) -> Tuple[np.ndarray, List[dict]]:
    store_path = Path(store_dir)
    embeddings = np.load(store_path / EMBEDDINGS_FILE)
    with (store_path / METADATA_FILE).open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    indexed_metadata = [item for item in metadata if item.get("status", "indexed") == "indexed"]
    return embeddings.astype(np.float32), indexed_metadata


def search_embeddings(query_embedding: np.ndarray, embeddings: np.ndarray, top_k: int = 5) -> List[SearchResult]:
    if embeddings.size == 0:
        return []

    query = normalize(query_embedding)
    matrix = np.asarray(embeddings, dtype=np.float32)
    scores = matrix @ query
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [SearchResult(index=int(i), score=float(scores[i])) for i in top_indices]


def search_store(query_embedding: np.ndarray, store_dir: str | Path, top_k: int = 5) -> List[SearchResult]:
    embeddings, metadata = load_store(store_dir)
    results = search_embeddings(query_embedding, embeddings, top_k)
    return [
        SearchResult(index=result.index, score=result.score, metadata=metadata[result.index])
        for result in results
    ]


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"

    amount = float(value)
    for unit in ("B", "KB", "MB", "GB"):
        if amount < 1024 or unit == "GB":
            return f"{amount:.1f} {unit}"
        amount /= 1024

    return f"{amount:.1f} GB"


def _print_progress(prefix: str, current: int, total: int | None) -> None:
    if total:
        percent = current / total * 100
        message = f"\r{prefix}: {_format_bytes(current)} / {_format_bytes(total)} ({percent:5.1f}%)"
    else:
        message = f"\r{prefix}: {_format_bytes(current)}"
    print(message, end="", file=sys.stderr, flush=True)


def _remote_file_size(url: str) -> int | None:
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request) as response:
            content_length = response.headers.get("Content-Length")
            return int(content_length) if content_length else None
    except Exception:
        return None


def download_file(url: str, destination: str | Path) -> Path:
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    existing_size = destination_path.stat().st_size if destination_path.exists() else 0
    remote_size = _remote_file_size(url)
    if existing_size and remote_size is not None and existing_size == remote_size:
        print(f"Archive already downloaded: {destination_path} ({_format_bytes(existing_size)})", file=sys.stderr)
        return destination_path

    request = urllib.request.Request(url)
    if existing_size:
        request.add_header("Range", f"bytes={existing_size}-")

    print(f"Downloading {url}", file=sys.stderr)
    print(f"Destination: {destination_path}", file=sys.stderr)

    try:
        response_context = urllib.request.urlopen(request)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and existing_size:
            print(f"Archive already downloaded: {destination_path} ({_format_bytes(existing_size)})", file=sys.stderr)
            return destination_path
        raise

    with response_context as response:
        status = response.getcode()
        content_length = response.headers.get("Content-Length")
        response_size = int(content_length) if content_length else None

        if existing_size and status == 206:
            mode = "ab"
            total_size = existing_size + response_size if response_size is not None else None
            downloaded = existing_size
        elif existing_size and response_size == existing_size:
            print(f"Archive already downloaded: {_format_bytes(existing_size)}", file=sys.stderr)
            return destination_path
        else:
            mode = "wb"
            total_size = response_size
            downloaded = 0

        with destination_path.open(mode) as out_file:
            while True:
                chunk = response.read(DOWNLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                _print_progress("Download", downloaded, total_size)

    print("", file=sys.stderr)
    return destination_path


def _safe_members(tar: tarfile.TarFile, destination: Path):
    destination = destination.resolve()
    for member in tar.getmembers():
        target = (destination / member.name).resolve()
        if os.path.commonpath([destination, target]) != str(destination):
            raise ValueError(f"Unsafe path in archive: {member.name}")
        yield member


def extract_tar_gz(archive_path: str | Path, destination: str | Path) -> None:
    destination_path = Path(destination)
    destination_path.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tar:
        members = list(_safe_members(tar, destination_path))
        total = len(members)
        print(f"Extracting {archive_path} to {destination_path}", file=sys.stderr)
        for index, member in enumerate(members, start=1):
            tar.extract(member, destination_path)
            if index == total or index % 100 == 0:
                _print_progress("Extract", index, total)
        print("", file=sys.stderr)


def download_lakh(dataset: str, output_dir: str | Path, *, extract: bool = True, url: str | None = None) -> Path:
    if dataset not in {"matched", "full"}:
        raise ValueError("dataset must be 'matched' or 'full'")

    dataset_url = url or (LMD_MATCHED_URL if dataset == "matched" else LMD_FULL_URL)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    archive_path = output_path / Path(dataset_url).name
    download_file(dataset_url, archive_path)
    if extract:
        extract_tar_gz(archive_path, output_path)
    return archive_path


def _features_from_audio(audio_path: str | Path) -> dict:
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import predict
    from basic_pitch.note_creation import model_output_to_notes

    model_output, _, _ = predict(audio_path, ICASSP_2022_MODEL_PATH)
    midi, _ = model_output_to_notes(
        output=model_output,
        onset_thresh=0.5,
        frame_thresh=0.3,
        infer_onsets=True,
        min_note_len=11,
        min_freq=1,
        max_freq=3500,
        include_pitch_bends=True,
        multiple_pitch_bends=False,
        melodia_trick=True,
        midi_tempo=120,
    )
    notes = sorted(
        [
            {
                "pitch": int(note.pitch),
                "start": float(note.start),
                "end": float(note.end),
                "duration": float(max(0.0, note.end - note.start)),
                "velocity": int(note.velocity),
            }
            for instrument in midi.instruments
            for note in instrument.notes
        ],
        key=lambda item: item["start"],
    )
    raw_pitches = [note["pitch"] for note in notes]
    pitches = normalize_melody_pitches(raw_pitches)
    return {
        "embedding": create_note_histogram(pitches),
        "melody_pitch_sequence": pitches,
        "interval_sequence": interval_sequence(pitches),
    }


def _embedding_from_audio(audio_path: str | Path) -> np.ndarray:
    return _features_from_audio(audio_path)["embedding"]


def _features_from_midi(midi_path: str | Path) -> dict:
    pitches = extract_pitches_from_midi(midi_path)
    melody_pitches = normalize_melody_pitches(melody_pitches_from_midi(midi_path))
    return {
        "embedding": create_note_histogram(pitches),
        "melody_pitch_sequence": melody_pitches,
        "interval_sequence": interval_sequence(melody_pitches),
    }


def _hybrid_search(
    query_features: dict,
    store_dir: str | Path,
    *,
    top_k: int,
    candidate_k: int,
    rerank: bool,
) -> List[SearchResult]:
    embeddings, metadata = load_store(store_dir)
    vector_results = search_embeddings(query_features["embedding"], embeddings, top_k=max(top_k, candidate_k))
    if not rerank or not query_features.get("interval_sequence"):
        return [
            SearchResult(index=result.index, score=result.score, metadata=metadata[result.index])
            for result in vector_results[:top_k]
        ]

    candidate_metadata = []
    for result in vector_results[:candidate_k]:
        item = dict(metadata[result.index])
        item["vector_score"] = result.score
        item["store_index"] = result.index
        candidate_metadata.append(item)

    candidates_with_sequences = [item for item in candidate_metadata if item.get("interval_sequence")]
    if not candidates_with_sequences:
        print(
            "No interval sequences found in metadata; run enrich-metadata --include-sequences or rebuild the store.",
            file=sys.stderr,
        )
        return [
            SearchResult(index=result.index, score=result.score, metadata=metadata[result.index])
            for result in vector_results[:top_k]
        ]

    return rerank_by_sequence(query_features["interval_sequence"], candidates_with_sequences, top_k=top_k)


def _print_search_results(results: Sequence[SearchResult]) -> None:
    for result in results:
        metadata = result.metadata or {}
        vector_score = metadata.get("vector_score")
        prefix = f"{result.score:.5f}"
        if vector_score is not None:
            prefix += f"\tvector={vector_score:.5f}"
        print(f"{prefix}\t{metadata.get('track_id', '')}\t{metadata.get('path', '')}")


def _cmd_download(args: argparse.Namespace) -> None:
    archive_path = download_lakh(args.dataset, args.output_dir, extract=not args.no_extract, url=args.url)
    print(f"Downloaded {archive_path}")


def _cmd_build(args: argparse.Namespace) -> None:
    embeddings, metadata = build_embeddings(args.midi_dir, limit=args.limit, min_notes=args.min_notes)
    save_store(args.store_dir, embeddings, metadata)
    skipped = len([item for item in metadata if item.get("status") == "skipped"])
    print(f"Indexed {len(embeddings)} MIDI files into {args.store_dir}; skipped {skipped}.")


def _cmd_enrich_metadata(args: argparse.Namespace) -> None:
    count = enrich_metadata_file(args.store_dir, include_sequences=args.include_sequences)
    print(f"Enriched {count} metadata rows in {Path(args.store_dir) / METADATA_FILE}.")


def _cmd_inspect_midi(args: argparse.Namespace) -> None:
    inspected = inspect_midi(args.midi_path)
    print(json.dumps(inspected, ensure_ascii=False, indent=2))


def _cmd_rank_track(args: argparse.Namespace) -> None:
    features = _features_from_audio(args.audio_path) if args.audio_path else _features_from_midi(args.midi_path)
    embeddings, metadata = load_store(args.store_dir)
    vector_results = search_embeddings(features["embedding"], embeddings, top_k=len(metadata))
    candidates = []
    for result in vector_results:
        item = dict(metadata[result.index])
        item["vector_score"] = result.score
        item["store_index"] = result.index
        candidates.append(item)
    ranked = rerank_by_sequence(features["interval_sequence"], candidates, top_k=len(candidates))

    matches = [result for result in ranked if result.metadata.get("track_id") == args.track_id]
    if not matches:
        print(f"{args.track_id} not found in store.")
        return

    first = matches[0]
    rank = ranked.index(first) + 1
    print(f"rank={rank}\tscore={first.score:.5f}\tvector={first.metadata.get('vector_score', 0):.5f}")
    print(json.dumps(first.metadata, ensure_ascii=False, indent=2))


def _cmd_query_audio(args: argparse.Namespace) -> None:
    features = _features_from_audio(args.audio_path)
    results = _hybrid_search(
        features,
        args.store_dir,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        rerank=not args.no_rerank,
    )
    _print_search_results(results)


def _cmd_query_midi(args: argparse.Namespace) -> None:
    features = _features_from_midi(args.midi_path)
    results = _hybrid_search(
        features,
        args.store_dir,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        rerank=not args.no_rerank,
    )
    _print_search_results(results)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and query a simple MIDI pitch-vector store.")
    subparsers = parser.add_subparsers(required=True)

    download_parser = subparsers.add_parser("download-lakh")
    download_parser.add_argument("--dataset", choices=["matched", "full"], default="matched")
    download_parser.add_argument("--output-dir", default="data/lakh")
    download_parser.add_argument("--url", default=None)
    download_parser.add_argument("--no-extract", action="store_true")
    download_parser.set_defaults(func=_cmd_download)

    build = subparsers.add_parser("build")
    build.add_argument("--midi-dir", required=True)
    build.add_argument("--store-dir", default="data/vectorstore")
    build.add_argument("--limit", type=int, default=None)
    build.add_argument("--min-notes", type=int, default=8)
    build.set_defaults(func=_cmd_build)

    enrich = subparsers.add_parser("enrich-metadata")
    enrich.add_argument("--store-dir", default="data/vectorstore")
    enrich.add_argument("--include-sequences", action="store_true")
    enrich.set_defaults(func=_cmd_enrich_metadata)

    inspect_parser = subparsers.add_parser("inspect-midi")
    inspect_parser.add_argument("--midi-path", required=True)
    inspect_parser.set_defaults(func=_cmd_inspect_midi)

    rank = subparsers.add_parser("rank-track")
    query_group = rank.add_mutually_exclusive_group(required=True)
    query_group.add_argument("--audio-path")
    query_group.add_argument("--midi-path")
    rank.add_argument("--track-id", required=True)
    rank.add_argument("--store-dir", default="data/vectorstore")
    rank.set_defaults(func=_cmd_rank_track)

    query_audio = subparsers.add_parser("query-audio")
    query_audio.add_argument("--audio-path", required=True)
    query_audio.add_argument("--store-dir", default="data/vectorstore")
    query_audio.add_argument("--top-k", type=int, default=5)
    query_audio.add_argument("--candidate-k", type=int, default=200)
    query_audio.add_argument("--no-rerank", action="store_true")
    query_audio.set_defaults(func=_cmd_query_audio)

    query_midi = subparsers.add_parser("query-midi")
    query_midi.add_argument("--midi-path", required=True)
    query_midi.add_argument("--store-dir", default="data/vectorstore")
    query_midi.add_argument("--top-k", type=int, default=5)
    query_midi.add_argument("--candidate-k", type=int, default=200)
    query_midi.add_argument("--no-rerank", action="store_true")
    query_midi.set_defaults(func=_cmd_query_midi)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
