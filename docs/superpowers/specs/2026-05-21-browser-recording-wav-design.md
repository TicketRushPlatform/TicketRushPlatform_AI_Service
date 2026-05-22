# Browser Recording WAV Design

## Goal

Add a small browser page served by the existing FastAPI service so a user can record audio from the device microphone, convert the recording to a WAV file, and submit it to the current SongFinder recognition endpoint.

## Scope

- Add a `GET /record` page to the existing FastAPI app.
- Keep the existing `POST /api/recognize/songfinder` upload endpoint unchanged for curl and external clients.
- Convert recorded browser audio to `audio/wav` before upload.
- Avoid new frontend build tooling or new dependencies.

## User Flow

1. User opens `/record`.
2. User clicks Start and grants microphone permission in the browser.
3. User clicks Stop.
4. The page converts the captured audio to a WAV blob and shows an audio preview.
5. The page uploads the WAV as multipart form field `audio` to `/api/recognize/songfinder?startTime=0`.
6. The page displays the returned song and movie fields, or an error message.

## Architecture

The page is returned directly from FastAPI as HTML with embedded CSS and JavaScript. JavaScript uses `navigator.mediaDevices.getUserMedia()` and `MediaRecorder` for capture. After recording, the page combines the chunks into a source blob, decodes it through `AudioContext.decodeAudioData`, encodes PCM data into a 16-bit WAV buffer, and posts the result as `recording.wav`.

The backend recognition path remains the same: `UploadFile` is read as bytes and sent to SongFinder with the configured RapidAPI credentials. This keeps the new UI as a client of the existing API rather than creating a second recognition path.

## WAV Conversion

The browser page will:

- Prefer a MediaRecorder MIME type supported by the browser.
- Decode the recorded blob into an `AudioBuffer`.
- Mix down to mono when multiple channels are present.
- Write a standard RIFF/WAVE header.
- Encode samples as signed 16-bit PCM with the original decoded sample rate.
- Upload the resulting blob with MIME type `audio/wav`.

This avoids adding server-side audio conversion dependencies. If SongFinder later requires a fixed sample rate, server-side or client-side resampling can be added as a separate change.

## Error Handling

- If microphone access is unavailable or denied, show a clear browser-page error.
- If recording produces no audio chunks, do not upload and show an error.
- If WAV conversion fails, show an error and leave the user on the page.
- If the API response is not successful, display the returned error detail.

## Testing

- Add a FastAPI test that `GET /record` returns HTML containing the recording controls and upload target.
- Keep existing helper tests unchanged.
- Run the full test suite after implementation.

## Out Of Scope

- A separate frontend application.
- Server-side microphone capture.
- Server-side audio transcoding.
- Styling beyond a compact usable page.
