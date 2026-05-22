from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()

RECORD_PAGE_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SongFinder Recorder</title>
  <style>
    :root {
      color-scheme: light dark;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f7f3ea;
      color: #1d2430;
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      padding: 24px;
      background:
        linear-gradient(135deg, rgba(18, 92, 92, 0.16), transparent 34%),
        linear-gradient(315deg, rgba(162, 73, 54, 0.16), transparent 38%),
        #f7f3ea;
    }

    main {
      width: min(720px, 100%);
      display: grid;
      gap: 20px;
    }

    h1 {
      margin: 0;
      font-size: 32px;
      font-weight: 760;
      line-height: 1.1;
    }

    p {
      margin: 0;
      color: #4b5563;
      line-height: 1.6;
    }

    .panel {
      display: grid;
      gap: 18px;
      padding: 24px;
      border: 1px solid rgba(29, 36, 48, 0.14);
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.74);
      box-shadow: 0 20px 60px rgba(29, 36, 48, 0.12);
    }

    .controls {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
    }

    button {
      min-height: 44px;
      border: 0;
      border-radius: 6px;
      padding: 0 18px;
      font: inherit;
      font-weight: 720;
      cursor: pointer;
      background: #116466;
      color: #ffffff;
    }

    button.secondary {
      background: #9a4f3f;
    }

    button:disabled {
      cursor: not-allowed;
      opacity: 0.48;
    }

    audio {
      width: 100%;
      min-height: 44px;
    }

    pre {
      min-height: 120px;
      margin: 0;
      overflow: auto;
      padding: 16px;
      border-radius: 6px;
      background: #18212f;
      color: #e8eef6;
      white-space: pre-wrap;
      word-break: break-word;
    }

    .status {
      min-height: 24px;
      font-weight: 680;
      color: #116466;
    }
  </style>
</head>
<body>
  <main>
    <section class="panel">
      <h1>Record and recognize</h1>
      <p>Record from this device, convert the capture to WAV, and send it to SongFinder.</p>

      <div class="controls">
        <button id="startButton" type="button">Start recording</button>
        <button id="stopButton" class="secondary" type="button" disabled>Stop recording</button>
      </div>

      <div id="status" class="status" role="status">Ready.</div>
      <audio id="preview" controls></audio>
      <pre id="result">{}</pre>
    </section>
  </main>

  <script>
    const endpoint = "/api/recognize/songfinder?startTime=0";
    const startButton = document.getElementById("startButton");
    const stopButton = document.getElementById("stopButton");
    const statusBox = document.getElementById("status");
    const preview = document.getElementById("preview");
    const resultBox = document.getElementById("result");

    let mediaRecorder = null;
    let mediaStream = null;
    let chunks = [];

    function setStatus(message) {
      statusBox.textContent = message;
    }

    function supportedRecorderOptions() {
      const candidates = [
        "audio/webm;codecs=opus",
        "audio/webm",
        "audio/mp4"
      ];
      const mimeType = candidates.find((candidate) => MediaRecorder.isTypeSupported(candidate));
      return mimeType ? { mimeType } : undefined;
    }

    function writeString(view, offset, value) {
      for (let index = 0; index < value.length; index += 1) {
        view.setUint8(offset + index, value.charCodeAt(index));
      }
    }

    function mixToMono(audioBuffer) {
      const output = new Float32Array(audioBuffer.length);
      for (let channel = 0; channel < audioBuffer.numberOfChannels; channel += 1) {
        const input = audioBuffer.getChannelData(channel);
        for (let index = 0; index < input.length; index += 1) {
          output[index] += input[index] / audioBuffer.numberOfChannels;
        }
      }
      return output;
    }

    function encodeWav(audioBuffer) {
      const samples = mixToMono(audioBuffer);
      const bytesPerSample = 2;
      const dataSize = samples.length * bytesPerSample;
      const buffer = new ArrayBuffer(44 + dataSize);
      const view = new DataView(buffer);
      const sampleRate = audioBuffer.sampleRate;

      writeString(view, 0, "RIFF");
      view.setUint32(4, 36 + dataSize, true);
      writeString(view, 8, "WAVE");
      writeString(view, 12, "fmt ");
      view.setUint32(16, 16, true);
      view.setUint16(20, 1, true);
      view.setUint16(22, 1, true);
      view.setUint32(24, sampleRate, true);
      view.setUint32(28, sampleRate * bytesPerSample, true);
      view.setUint16(32, bytesPerSample, true);
      view.setUint16(34, 16, true);
      writeString(view, 36, "data");
      view.setUint32(40, dataSize, true);

      let offset = 44;
      for (let index = 0; index < samples.length; index += 1) {
        const sample = Math.max(-1, Math.min(1, samples[index]));
        view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
        offset += bytesPerSample;
      }

      return new Blob([view], { type: "audio/wav" });
    }

    async function recordingBlobToWav(blob) {
      const arrayBuffer = await blob.arrayBuffer();
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      const audioContext = new AudioContextClass();
      try {
        const decoded = await audioContext.decodeAudioData(arrayBuffer);
        return encodeWav(decoded);
      } finally {
        await audioContext.close();
      }
    }

    async function uploadWav(wavBlob) {
      const formData = new FormData();
      formData.append("audio", wavBlob, "recording.wav");

      const response = await fetch(endpoint, {
        method: "POST",
        body: formData
      });
      const responseText = await response.text();
      let payload = responseText;
      try {
        payload = JSON.parse(responseText);
      } catch (error) {
        payload = { raw: responseText };
      }

      if (!response.ok) {
        throw new Error(JSON.stringify(payload, null, 2));
      }

      return payload;
    }

    async function finishRecording() {
      startButton.disabled = false;
      stopButton.disabled = true;
      if (mediaStream) {
        mediaStream.getTracks().forEach((track) => track.stop());
      }

      if (!chunks.length) {
        throw new Error("No audio was captured.");
      }

      setStatus("Converting recording to WAV...");
      const sourceBlob = new Blob(chunks, { type: mediaRecorder.mimeType || "audio/webm" });
      const wavBlob = await recordingBlobToWav(sourceBlob);
      preview.src = URL.createObjectURL(wavBlob);

      setStatus("Uploading WAV for recognition...");
      const result = await uploadWav(wavBlob);
      resultBox.textContent = JSON.stringify(result, null, 2);
      setStatus("Done.");
    }

    async function startRecording() {
      try {
        resultBox.textContent = "{}";
        preview.removeAttribute("src");
        chunks = [];
        mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(mediaStream, supportedRecorderOptions());
        mediaRecorder.addEventListener("dataavailable", (event) => {
          if (event.data && event.data.size > 0) {
            chunks.push(event.data);
          }
        });
        mediaRecorder.addEventListener("stop", () => {
          finishRecording().catch((error) => {
            setStatus("Error.");
            resultBox.textContent = error.message;
          });
        });
        mediaRecorder.start();
        startButton.disabled = true;
        stopButton.disabled = false;
        setStatus("Recording...");
      } catch (error) {
        setStatus("Error.");
        resultBox.textContent = error.message;
        startButton.disabled = false;
        stopButton.disabled = true;
      }
    }

    function stopRecording() {
      if (mediaRecorder && mediaRecorder.state !== "inactive") {
        mediaRecorder.stop();
        setStatus("Stopping...");
      }
    }

    startButton.addEventListener("click", startRecording);
    stopButton.addEventListener("click", stopRecording);
  </script>
</body>
</html>
""".strip()


@router.get("/record", response_class=HTMLResponse)
def record_page() -> str:
    return RECORD_PAGE_HTML
