```bash
uv run python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

```bash
http://127.0.0.1:8000/record
```

```powershell
$body = @{
  message = "Hay dung tool list_events de goi Event Service, page=1, page_size=5."
  thread_id = "test"
} | ConvertTo-Json -Compress

$response = Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/chat" `
  -Method Post `
  -ContentType "application/json; charset=utf-8" `
  -Body $body

$response.message

# Or inspect the full response object:
$response | ConvertTo-Json -Depth 10
```
