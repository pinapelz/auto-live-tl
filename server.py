import time
import threading
import json
import queue
import os
from flask import Flask, Response, stream_with_context
from flask_cors import CORS
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from gui import select_settings, prompt_input_sample_rate

TARGET_SAMPLE_RATE = 16000
CAPTURE_SAMPLE_RATE = 0
BUFFER_SECONDS = 10
MAX_SAMPLES = 0
PROCESS_INTERVAL_SECONDS = 2
SSE_EVENT_SUBTITLE = "subtitle"
SSE_KEEPALIVE_SECONDS = 15

SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "settings.json")

DEFAULT_SETTINGS = {
    "audio_device_name": "",
    "model_name": "medium",
    "device": "cpu",
    "compute_type": "int8",
    "task": "translate",
    "beam_size": 3,
    "language": "",
    "context_seconds": 10,
    "update_interval_seconds": 2,
}

MODEL_CHOICES = ["tiny", "base", "small", "medium", "large-v2", "large-v3", "distil-large-v3"]
DEVICE_CHOICES = ["cpu", "cuda", "auto"]
COMPUTE_CHOICES = ["int8", "int8_float16", "float16", "float32"]
TASK_CHOICES = ["translate", "transcribe"]

audio_buffer = np.zeros(0, dtype=np.float32)
lock = threading.Lock()
model = None
WHISPER_TASK = DEFAULT_SETTINGS["task"]
WHISPER_BEAM_SIZE = DEFAULT_SETTINGS["beam_size"]
WHISPER_LANGUAGE = DEFAULT_SETTINGS["language"]

last_payload = None
clients = set()
clients_lock = threading.Lock()
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 5000
app = Flask(__name__)
CORS(app)

def resample_audio(audio_np: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return audio_np
    if len(audio_np) == 0:
        return audio_np
    dst_len = int(len(audio_np) * dst_rate / src_rate)
    if dst_len <= 0:
        return audio_np[:0]
    x_old = np.arange(len(audio_np))
    x_new = np.linspace(0, len(audio_np) - 1, dst_len)
    return np.interp(x_new, x_old, audio_np).astype(np.float32)


def load_settings() -> dict:
    if not os.path.exists(SETTINGS_PATH):
        return DEFAULT_SETTINGS.copy()
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return DEFAULT_SETTINGS.copy()
    merged = DEFAULT_SETTINGS.copy()
    for key, value in data.items():
        if key in merged:
            merged[key] = value
    return merged


def save_settings(settings: dict) -> None:
    try:
        with open(SETTINGS_PATH, "w", encoding="utf-8") as handle:
            json.dump(settings, handle, indent=2)
    except OSError as exc:
        print(f"Failed to save settings: {exc}")


def run_whisper(audio_np: np.ndarray) -> str:
    transcribe_kwargs = {"task": WHISPER_TASK, "beam_size": WHISPER_BEAM_SIZE}
    if WHISPER_LANGUAGE:
        transcribe_kwargs["language"] = WHISPER_LANGUAGE
    segments, _info = model.transcribe(audio_np, **transcribe_kwargs)
    text = " ".join(seg.text for seg in segments).strip()
    if text:
        print("🟢", text)
        broadcast_subtitle(text)
    return text


def broadcast_subtitle(text: str) -> None:
    global last_payload
    payload = {"text": text}
    last_payload = payload
    with clients_lock:
        targets = list(clients)
    for client_queue in targets:
        try:
            client_queue.put_nowait(payload)
        except queue.Full:
            pass

def format_sse_event(event: str, payload: dict) -> str:
    data = json.dumps(payload)
    return f"event: {event}\ndata: {data}\n\n"

def event_stream():
    client_queue = queue.Queue(maxsize=10)
    with clients_lock:
        clients.add(client_queue)

    if last_payload:
        yield format_sse_event(SSE_EVENT_SUBTITLE, last_payload)

    try:
        while True:
            try:
                payload_data = client_queue.get(timeout=SSE_KEEPALIVE_SECONDS)
            except queue.Empty:
                yield ": keep-alive\n\n"
                continue
            yield format_sse_event(SSE_EVENT_SUBTITLE, payload_data)
    finally:
        with clients_lock:
            clients.discard(client_queue)

@app.get("/events")
def events():
    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Access-Control-Allow-Origin": "*",
    }
    return Response(stream_with_context(event_stream()), mimetype="text/event-stream", headers=headers)


@app.get("/health")
def health():
    response = Response("ok", mimetype="text/plain")
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response


def start_subtitle_server():
    thread = threading.Thread(
        target=lambda: app.run(
            host=SERVER_HOST,
            port=SERVER_PORT,
            threaded=True,
            use_reloader=False,
        ),
        daemon=True,
    )
    thread.start()
    print(f"SSE subtitle server listening on http://{SERVER_HOST}:{SERVER_PORT}/events")
    return thread


def list_audio_devices() -> None:
    devices = sd.query_devices()
    print("Available audio devices:")
    for idx, dev in enumerate(devices):
        io = []
        if dev["max_input_channels"] > 0:
            io.append("input")
        if dev["max_output_channels"] > 0:
            io.append("output")
        io_str = "/".join(io) if io else "none"
        print(f"[{idx}] {dev['name']} ({io_str})")

def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"Audio status: {status}")
    chunk = indata[:, 0].copy()

    global audio_buffer
    with lock:
        audio_buffer = np.concatenate([audio_buffer, chunk])
        if len(audio_buffer) > MAX_SAMPLES:
            audio_buffer = audio_buffer[-MAX_SAMPLES:]


def processing_loop():
    while True:
        time.sleep(PROCESS_INTERVAL_SECONDS)
        with lock:
            if len(audio_buffer) == 0 or CAPTURE_SAMPLE_RATE <= 0:
                continue
            audio_copy = audio_buffer.copy()
            capture_rate = CAPTURE_SAMPLE_RATE
        audio_16k = resample_audio(audio_copy, capture_rate, TARGET_SAMPLE_RATE)
        run_whisper(audio_16k)


def select_input_sample_rate(device_index: int, preferred_rate: int) -> int:
    common_rates = [48000, 44100, 32000, 24000, 22050, 16000, 12000, 8000]
    tried = set()
    for rate in [preferred_rate] + common_rates:
        if rate in tried or rate <= 0:
            continue
        tried.add(rate)
        try:
            sd.check_input_settings(device=device_index, channels=1, samplerate=rate, dtype="float32")
            return rate
        except sd.PortAudioError:
            continue
    return prompt_input_sample_rate(device_index, common_rates)


def main():
    global CAPTURE_SAMPLE_RATE, MAX_SAMPLES, model, WHISPER_TASK, WHISPER_BEAM_SIZE, WHISPER_LANGUAGE
    global BUFFER_SECONDS, PROCESS_INTERVAL_SECONDS
    start_subtitle_server()

    settings = load_settings()
    devices = sd.query_devices()
    input_devices = [(idx, dev) for idx, dev in enumerate(devices) if dev["max_input_channels"] > 0]
    settings = select_settings(
        settings,
        input_devices,
        DEFAULT_SETTINGS,
        MODEL_CHOICES,
        DEVICE_CHOICES,
        COMPUTE_CHOICES,
        TASK_CHOICES,
    )
    save_settings(settings)

    device_name = settings.get("audio_device_name", "")
    matched_index = None
    for idx, dev in enumerate(devices):
        if dev.get("name") == device_name and dev.get("max_input_channels", 0) > 0:
            matched_index = idx
            break
    if matched_index is None:
        raise RuntimeError("Saved audio device not found. Please reselect in the settings window.")
    device_index = matched_index

    model_name = settings["model_name"]
    whisper_device = settings["device"]
    compute_type = settings["compute_type"]
    WHISPER_TASK = settings["task"]
    WHISPER_BEAM_SIZE = int(settings["beam_size"])
    WHISPER_LANGUAGE = settings["language"].strip() if settings["language"] else ""
    BUFFER_SECONDS = float(settings.get("context_seconds", BUFFER_SECONDS))
    PROCESS_INTERVAL_SECONDS = float(settings.get("update_interval_seconds", PROCESS_INTERVAL_SECONDS))
    if BUFFER_SECONDS <= 0:
        BUFFER_SECONDS = DEFAULT_SETTINGS["context_seconds"]
    if PROCESS_INTERVAL_SECONDS <= 0:
        PROCESS_INTERVAL_SECONDS = DEFAULT_SETTINGS["update_interval_seconds"]

    model = WhisperModel(model_name, device=whisper_device, compute_type=compute_type)

    device_info = sd.query_devices(device_index)
    preferred_rate = int(device_info["default_samplerate"])
    if preferred_rate <= 0:
        preferred_rate = 48000
    CAPTURE_SAMPLE_RATE = select_input_sample_rate(device_index, preferred_rate)
    MAX_SAMPLES = int(CAPTURE_SAMPLE_RATE * BUFFER_SECONDS)
    print(f"Using device {device_index}: {device_info['name']}")
    print(f"Model: {model_name} | task={WHISPER_TASK} | beam_size={WHISPER_BEAM_SIZE}")
    print(f"Compute: device={whisper_device} | compute_type={compute_type}")
    print(f"Capture sample rate: {CAPTURE_SAMPLE_RATE} Hz (resampling to {TARGET_SAMPLE_RATE} Hz)")
    processing_thread = threading.Thread(target=processing_loop, daemon=True)
    processing_thread.start()
    with sd.InputStream(
        device=device_index,
        channels=1,
        samplerate=CAPTURE_SAMPLE_RATE,
        dtype="float32",
        callback=audio_callback,
        blocksize=int(CAPTURE_SAMPLE_RATE * 0.5),
    ):
        print("Listening... Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Stopping.")


if __name__ == "__main__":
    main()
