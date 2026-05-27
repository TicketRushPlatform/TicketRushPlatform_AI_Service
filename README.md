# TicketRush AI Service

## Run From Scratch

1. Create local environment config:

```bash
cp .env.example .env
```

2. Fill real values in `.env`, especially:

```text
RAPIDAPI_KEY
OPENAI_API_KEY
JWT_SECRET
EVENT_SERVICE_BASE_URL
BOOKING_SERVICE_BASE_URL
```

3. Install dependencies:

```bash
uv sync
```

4. Start the API locally:

```bash
uv run python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

After changing backend code, restart this command if the server was not started with `--reload`.

5. Test the health check:

```bash
curl http://127.0.0.1:8000/
```

Open API docs:

```text
http://127.0.0.1:8000/docs
```

Detailed API testing docs are in [docs/API_TESTING.md](docs/API_TESTING.md).

## Container deployment

Build the AI service image as a reusable artifact:

```bash
docker build -t ticketrush-ai-service .
```

Rebuild this image after backend code changes.

Run it with runtime environment variables instead of baking `.env` into the image:

```bash
docker run --rm --env-file .env -p 8000:8000 ticketrush-ai-service
```

For a microservices deployment, keep services separate:

- Frontend is deployed as its own service and should call the API through the platform gateway or reverse proxy.
- AI service exposes only API endpoints such as `/`, `/api/chat`, and `/api/recognize/songfinder`.
- Event and booking services stay behind internal service DNS names such as `event-service` and `booking-service`.
- Secrets stay in runtime configuration (`.env`, Compose secrets, Kubernetes secrets, or CI/CD variables), not in the Docker image.

## Quick Chat Test

```powershell
$body = @{
  message = "Hay dung tool list_events de goi Event Service, page=1, page_size=5."
  thread_id = "test"
} | ConvertTo-Json -Compress

$body | curl.exe -X POST "http://127.0.0.1:8000/api/chat" `
  -H "Content-Type: application/json" `
  --data-binary '@-'
```

If PowerShell only prints `Internal Server Error`, run the request with this wrapper to see the response body:

```powershell
try {
  $body = @{
    message = "Hay dung tool list_events de goi Event Service, page=1, page_size=5."
    thread_id = "test"
  } | ConvertTo-Json -Compress

  $response = Invoke-RestMethod `
    -Uri "http://127.0.0.1:8000/api/chat" `
    -Method Post `
    -ContentType "application/json; charset=utf-8" `
    -Body $body

  $response | ConvertTo-Json -Depth 10
} catch {
  $statusCode = [int]$_.Exception.Response.StatusCode
  $reader = [System.IO.StreamReader]::new($_.Exception.Response.GetResponseStream())
  $errorBody = $reader.ReadToEnd()
  "HTTP $statusCode"
  $errorBody
}
```

Common causes:

- `OPENAI_API_KEY` is missing or invalid.
- `OPENAI_MODEL` is not available to your key.
- `EVENT_SERVICE_BASE_URL` points to a service that is not running.
- `BOOKING_SERVICE_BASE_URL` points to a service that is not running.
- If the response is plain `Internal Server Error` instead of JSON, restart the API process or rebuild and rerun the Docker container.

## Quick Song Recognition Test

Use an existing WAV file:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/recognize/songfinder?startTime=0" `
  -F "audio=@recording.wav;type=audio/wav"
```

Replace `recording.wav` with the path to your test audio file.

If the file is outside the current folder:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/recognize/songfinder?startTime=0" `
  -F "audio=@F:\path\to\recording.wav;type=audio/wav"
```
