from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Optional

import httpx
from pydantic import BaseModel, Field


class MovieLookupOutput(BaseModel):
    movie: Optional[str] = Field(
        default=None,
        description="Movie title associated with the song, or null if unknown.",
    )


@dataclass(frozen=True)
class SongFinderSettings:
    rapidapi_key: Optional[str] = None
    rapidapi_host: str = "songfinder-file-recognition.p.rapidapi.com"
    songfinder_url: str = "https://songfinder-file-recognition.p.rapidapi.com/api/rapidapi/recognize/file"
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"

    @classmethod
    def from_env(cls) -> "SongFinderSettings":
        return cls(
            rapidapi_key=os.getenv("RAPIDAPI_KEY"),
            rapidapi_host=os.getenv("RAPIDAPI_HOST", cls.rapidapi_host),
            songfinder_url=os.getenv("SONGFINDER_URL", cls.songfinder_url),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_model=os.getenv("OPENAI_MODEL", cls.openai_model),
        )


def extract_songfinder_track(songfinder_data: Any) -> Optional[dict[str, Any]]:
    if not isinstance(songfinder_data, dict):
        return None

    track = songfinder_data.get("track")
    if not isinstance(track, dict):
        return None

    extracted = {
        "title": track.get("title"),
        "artist": track.get("artist"),
        "album": track.get("album"),
        "release_date": track.get("releaseDate"),
        "genre": track.get("genre"),
        "label": track.get("label"),
        "cover_art": track.get("coverArt"),
        "isrc": track.get("isrc"),
    }
    return {key: value for key, value in extracted.items() if value}


def find_movie_for_track(
    track: Optional[dict[str, Any]],
    *,
    openai_api_key: Optional[str] = None,
    openai_model: str = "gpt-4o-mini",
) -> Optional[str]:
    if not track or not openai_api_key:
        return None

    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        return None

    prompt = f"""
You are a music supervisor research assistant.
Find the movie most strongly associated with this song.

Song metadata:
- Title: {track.get("title", "Unknown")}
- Artist: {track.get("artist", "Unknown")}
- Album: {track.get("album", "Unknown")}
- Release date: {track.get("release_date", "Unknown")}
- Genre: {track.get("genre", "Unknown")}
- Label: {track.get("label", "Unknown")}
- ISRC: {track.get("isrc", "Unknown")}

Return only the movie title in the structured output.
If there is no reliable movie match, set movie to null.
""".strip()

    try:
        llm = ChatOpenAI(
            model=openai_model,
            api_key=openai_api_key,
            temperature=0,
        )
        result = llm.with_structured_output(MovieLookupOutput).invoke(prompt)

        if isinstance(result, MovieLookupOutput):
            return result.movie
        if isinstance(result, dict):
            return result.get("movie")
        return None
    except Exception:
        return None


async def recognize_song_bytes(
    audio_bytes: bytes,
    *,
    start_time: int = 0,
    settings: Optional[SongFinderSettings] = None,
    http_client_factory: Callable[..., Any] = httpx.AsyncClient,
    movie_lookup: Optional[Callable[[Optional[dict[str, Any]]], Optional[str]]] = None,
) -> dict[str, Any]:
    resolved_settings = settings or SongFinderSettings.from_env()
    if not resolved_settings.rapidapi_key:
        raise ValueError("Missing RAPIDAPI_KEY in .env")

    headers = {
        "content-type": "application/octet-stream",
        "x-rapidapi-host": resolved_settings.rapidapi_host,
        "x-rapidapi-key": resolved_settings.rapidapi_key,
    }
    params = {"startTime": start_time}

    async with http_client_factory(timeout=60.0) as client:
        response = await client.post(
            resolved_settings.songfinder_url,
            params=params,
            headers=headers,
            content=audio_bytes,
        )

    response.raise_for_status()

    try:
        data = response.json()
    except ValueError:
        data = response.text

    track = extract_songfinder_track(data)
    lookup = movie_lookup or (
        lambda found_track: find_movie_for_track(
            found_track,
            openai_api_key=resolved_settings.openai_api_key,
            openai_model=resolved_settings.openai_model,
        )
    )
    movie = lookup(track)

    return {
        "success": True,
        "song": track.get("title") if track else None,
        "movie": movie,
        "track": track,
    }
