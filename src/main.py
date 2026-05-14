import os
from typing import Any, Dict, Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()

app = FastAPI(title="SongFinder Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
RAPIDAPI_HOST = os.getenv(
    "RAPIDAPI_HOST",
    "songfinder-file-recognition.p.rapidapi.com",
)
SONGFINDER_URL = os.getenv(
    "SONGFINDER_URL",
    "https://songfinder-file-recognition.p.rapidapi.com/api/rapidapi/recognize/file",
)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


class MovieLookupOutput(BaseModel):
    movie: Optional[str] = Field(
        default=None,
        description="Movie title associated with the song, or null if unknown.",
    )


def extract_songfinder_track(songfinder_data: Any) -> Optional[Dict[str, Any]]:
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
    track: Optional[Dict[str, Any]],
    openai_api_key: Optional[str] = OPENAI_API_KEY,
) -> Optional[str]:
    if not track:
        return None

    if not openai_api_key:
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
            model=OPENAI_MODEL,
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


@app.get("/")
def health_check():
    return {"status": "ok"}


@app.post("/api/recognize/songfinder")
async def recognize_song(
    audio: UploadFile = File(...),
    start_time: int = Query(0, alias="startTime"),
) -> Dict[str, Any]:
    if not RAPIDAPI_KEY:
        raise HTTPException(
            status_code=500,
            detail="Missing RAPIDAPI_KEY in .env",
        )

    audio_bytes = await audio.read()

    if not audio_bytes:
        raise HTTPException(
            status_code=400,
            detail="Empty audio file",
        )

    headers = {
        "content-type": "application/octet-stream",
        "x-rapidapi-host": RAPIDAPI_HOST,
        "x-rapidapi-key": RAPIDAPI_KEY,
    }

    params = {
        "startTime": start_time,
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                SONGFINDER_URL,
                params=params,
                headers=headers,
                content=audio_bytes,
            )

        if response.status_code >= 400:
            raise HTTPException(
                status_code=response.status_code,
                detail={
                    "message": "SongFinder API error",
                    "status_code": response.status_code,
                    "response": response.text,
                },
            )

        try:
            data = response.json()
        except Exception:
            data = response.text
        track = extract_songfinder_track(data)
        # track = {
        #     "title": "My Heart Will Go On",
        #     "artist": "Celine Dion",
        #     "album": "Let's Talk About Love",
        #     "release_date": "1997",
        #     "genre": "Pop",
        #     "label": "Epic Records",
        #     "cover_art": "https://example.com/cover.jpg",
        #     "isrc": "USUM79901234",
        # }
        movie = find_movie_for_track(track)
        # data = {
        #     "mock": True,
        #     "message": "This is a mocked response. Replace with actual SongFinder API call.",
        # }
        return {
            "success": True,
            "song": track.get("title") if track else None,
            "movie": movie,
        }

    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="SongFinder request timeout",
        )
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Cannot connect to SongFinder API: {str(e)}",
        )
