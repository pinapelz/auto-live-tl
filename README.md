# auto-live-tl
A basic LOCAL translation backend that listens to an audio sink via PCM and runs translation via faster-whisper. Also supports the option to use `qwen2.5-7b-instruct` (can be changed but has to be edited in the source code) to format/clean-up subtitles based on sliding window context.

Translations and trascriptions are transformers based, inaccuracies and hallucinations will occur.

# Setup
> It's highly recommended that you run this with a GPU, running with CPU is possible but inference will be very slow outside of using tiny models (which compromise accuracy)
>
> For this, you will need to install a Nvidia CUDA 12 toolkit. I am running with [CUDA Toolkit 12.9](https://developer.nvidia.com/cuda-12-9-0-download-archive)

```
uv sync
uv run server.py
```
A GUI is available for configuration

`server.py` serves a backend for translating incoming audio data. It expects some other client to hit the `/events` endpoint to fetch the translated data.


# Clients:

`youtube-subtitle.user.js` is one such example client that can fetch data from this endpoint and render it beneath a YouTube video. You can install it as a userscript.

<img width="1210" height="109" alt="image" src="https://github.com/user-attachments/assets/2bffde45-bc61-4d63-b779-b7a8cd183bc0" />

# SSE Subtitle Server API
Generated subtitles are broadcast as server-sent-events (event stream). See the API below

---

### `GET /health`
Simple liveness check.

**Response:** `200 OK`, body `ok` (plain text)

---

### `GET /events`
The main subtitle stream. Uses **Server-Sent Events (SSE)** — keep the connection open and read events as they arrive.

**Response headers:**
```/dev/null/example.http#L1-3
Content-Type: text/event-stream
Cache-Control: no-cache
Access-Control-Allow-Origin: *
```

**Event types you'll receive:**

| Type | When |
|---|---|
| `subtitle` | A new subtitle is ready |
| *(keep-alive comment)* | Every 15 s of silence, to prevent connection drops |

**`subtitle` event payload** — JSON in the `data` field:
```/dev/null/subtitle.json#L1-3
{
  "text": "The cleaned subtitle string."
}
```

**Keep-alive** lines look like:
```
: keep-alive
```
These carry no data and should be ignored.

---

### How to connect (examples)

**JavaScript:**
```js
const source = new EventSource("http://127.0.0.1:5000/events");

source.addEventListener("subtitle", (event) => {
  const { text } = JSON.parse(event.data);
  console.log(text);
});
```

**Python:**
```python
import sseclient, requests

resp = requests.get("http://127.0.0.1:5000/events", stream=True)
client = sseclient.SSEClient(resp)
for event in client.events():
    if event.event == "subtitle":
        import json
        print(json.loads(event.data)["text"])
```


## Demo
Ran using faster-whisper medium and qwen2.5-7B-instruct on RTX 3060 Mobile (CUDA)

Example 1:

https://github.com/user-attachments/assets/db602a11-2d13-4e58-a5e8-1d4a71c1be0e


Example 2:

https://github.com/user-attachments/assets/a480809e-77f7-4b66-9686-aa2ffea8333d




