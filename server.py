import time
import threading
import json
import queue
import os
from typing import Any, Dict, Optional, Set, List, Iterator
from flask import Flask
from flask_cors import CORS
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from gui import select_settings, prompt_input_sample_rate
from routes import register_routes

TARGET_SAMPLE_RATE: int = 16000
CAPTURE_SAMPLE_RATE: int = 0
BUFFER_SECONDS: float = 10
MAX_SAMPLES: int = 0
PROCESS_INTERVAL_SECONDS: float = 2
SSE_EVENT_SUBTITLE: str = "subtitle"
SSE_KEEPALIVE_SECONDS: int = 15

SETTINGS_PATH: str = os.path.join(os.path.dirname(__file__), "settings.json")

DEFAULT_SETTINGS: Dict[str, Any] = {
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

MODEL_CHOICES: List[str] = ["tiny", "base", "small", "medium", "large-v2", "large-v3", "distil-large-v3"]
DEVICE_CHOICES: List[str] = ["cpu", "cuda", "auto"]
COMPUTE_CHOICES: List[str] = ["int8", "int8_float16", "float16", "float32"]
TASK_CHOICES: List[str] = ["translate", "transcribe"]

audio_buffer: np.ndarray = np.zeros(0, dtype=np.float32)
lock: threading.Lock = threading.Lock()
model: Optional[WhisperModel] = None
WHISPER_TASK: str = DEFAULT_SETTINGS["task"]
WHISPER_BEAM_SIZE: int = DEFAULT_SETTINGS["beam_size"]
WHISPER_LANGUAGE: str = DEFAULT_SETTINGS["language"]

last_payload: Optional[Dict[str, Any]] = None
clients: Set[queue.Queue] = set()
clients_lock: threading.Lock = threading.Lock()
SERVER_HOST: str = "127.0.0.1"
SERVER_PORT: int = 5000
app: Flask = Flask(__name__)
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


def load_settings() -> Dict[str, Any]:
    if not os.path.exists(SETTINGS_PATH):
        return DEFAULT_SETTINGS.copy()
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return DEFAULT_SETTINGS.copy()
    merged: Dict[str, Any] = DEFAULT_SETTINGS.copy()
    for key, value in data.items():
        if key in merged:
            merged[key] = value
    return merged


def save_settings(settings: Dict[str, Any]) -> None:
    try:
        with open(SETTINGS_PATH, "w", encoding="utf-8") as handle:
            json.dump(settings, handle, indent=2)
    except OSError as exc:
        print(f"Failed to save settings: {exc}")


def run_whisper(audio_np: np.ndarray) -> str:
    transcribe_kwargs: Dict[str, Any] = {"task": WHISPER_TASK, "beam_size": WHISPER_BEAM_SIZE}
    if WHISPER_LANGUAGE:
        transcribe_kwargs["language"] = WHISPER_LANGUAGE
    # model is expected to be initialized in main()
    assert model is not None, "Whisper model is not initialized"
    segments, _info = model.transcribe(audio_np, **transcribe_kwargs)
    text = " ".join(seg.text for seg in segments).strip()
    if text:
        print("🟢", text)
        broadcast_subtitle(text)
    return text


def broadcast_subtitle(text: str) -> None:
    global last_payload
    payload: Dict[str, Any] = {"text": text}
    last_payload = payload
    with clients_lock:
        targets = list(clients)
    for client_queue in targets:
        try:
            client_queue.put_nowait(payload)
        except queue.Full:
            pass


def format_sse_event(event: str, payload: Dict[str, Any]) -> str:
    data = json.dumps(payload)
    return f"event: {event}\ndata: {data}\n\n"


def event_stream() -> Iterator[str]:
    client_queue: queue.Queue = queue.Queue(maxsize=10)
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


def start_subtitle_server() -> threading.Thread:
    register_routes(app, event_stream)

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


def audio_callback(indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
    if status:
        print(f"Audio status: {status}")
    # Take first channel
    chunk: np.ndarray = indata[:, 0].copy()

    global audio_buffer
    with lock:
        audio_buffer = np.concatenate([audio_buffer, chunk])
        if len(audio_buffer) > MAX_SAMPLES:
            audio_buffer = audio_buffer[-MAX_SAMPLES:]


def is_silent(audio_16k: Optional[np.ndarray]) -> bool:
    if audio_16k is None or len(audio_16k) == 0:
        return False
    rms: float = float(np.sqrt(np.mean(np.square(audio_16k))))  # root mean square
    return rms < 0.003


def processing_loop() -> None:
    while True:
        time.sleep(PROCESS_INTERVAL_SECONDS)
        with lock:
            if len(audio_buffer) == 0 or CAPTURE_SAMPLE_RATE <= 0:
                continue
            audio_copy: np.ndarray = audio_buffer.copy()
            capture_rate: int = CAPTURE_SAMPLE_RATE
        audio_16k: np.ndarray = resample_audio(audio_copy, capture_rate, TARGET_SAMPLE_RATE)
        if is_silent(audio_16k):
            continue
        run_whisper(audio_16k)


def select_input_sample_rate(device_index: int, preferred_rate: int) -> int:
    common_rates: List[int] = [48000, 44100, 32000, 24000, 22050, 16000, 12000, 8000]
    tried: Set[int] = set()
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


def main() -> None:
    global CAPTURE_SAMPLE_RATE, MAX_SAMPLES, model, WHISPER_TASK, WHISPER_BEAM_SIZE, WHISPER_LANGUAGE
    global BUFFER_SECONDS, PROCESS_INTERVAL_SECONDS
    start_subtitle_server()

    settings: Dict[str, Any] = load_settings()
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

    device_name: str = settings.get("audio_device_name", "")
    matched_index: Optional[int] = None
    for idx, dev in enumerate(devices):
        if dev.get("name") == device_name and dev.get("max_input_channels", 0) > 0:
            matched_index = idx
            break
    if matched_index is None:
        raise RuntimeError("Saved audio device not found. Please reselect in the settings window.")
    device_index: int = matched_index

    model_name: str = settings["model_name"]
    whisper_device: str = settings["device"]
    compute_type: str = settings["compute_type"]
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
    preferred_rate: int = int(device_info["default_samplerate"])
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
