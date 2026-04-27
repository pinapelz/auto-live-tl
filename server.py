import time
import threading
import json
import queue
import os
from collections import Counter, deque
import re
from typing import Any, Deque, Dict, Optional, Set, List, Iterator
from flask import Flask
from flask_cors import CORS
import ollama as _ollama
from ollama import chat
from ollama import ChatResponse
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from gui import select_settings, prompt_input_sample_rate
from routes import register_routes
from config import _SYSTEM_PROMPT

TARGET_SAMPLE_RATE: int = 16000
CAPTURE_SAMPLE_RATE: int = 0
BUFFER_SECONDS: float = 10
MAX_SAMPLES: int = 0
PROCESS_INTERVAL_SECONDS: float = 2
SSE_EVENT_SUBTITLE: str = "subtitle"
SSE_KEEPALIVE_SECONDS: int = 15

USE_OLLAMA_CLEANUP: bool = True
OLLAMA_MODEL: str = "qwen2.5:7b-instruct"
OLLAMA_CONTEXT_WINDOW: int = 6  # number of recent cleaned segments kept as context
OLLAMA_OPTIONS: Dict[str, Any] = {"num_gpu": 1}
RAW_BATCH_SIZE: int = 2  # accumulate this many raw Whisper lines before calling the LLM

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
    "use_ollama_cleanup": True,
    "ollama_device": "GPU",
    "ollama_context_window": 5,
    "ollama_raw_batch_size": 2,
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

# OLLAMA stuff
llm_input_queue: queue.Queue = queue.Queue(maxsize=1)
subtitle_context: Deque[str] = deque(maxlen=OLLAMA_CONTEXT_WINDOW) # sliding window context
subtitle_context_lock: threading.Lock = threading.Lock()
_raw_batch: List[str] = []
_raw_batch_lock: threading.Lock = threading.Lock()

def resample_audio(audio_np: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """
    Resamples audio to TARGET_SAMPLE_RATE (default is 16000hz), speeds up inference time, fetched as a nd array
    """
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

def cleanup_subtitle_with_ollama(raw_text: str, context: List[str]) -> Optional[str]:
    if context:
        context_block = "\n".join(f"- {seg}" for seg in context)
    else:
        context_block = "(none yet)"

    user_message = (
        f"ALREADY SHOWN:\n{context_block}\n\n"
        "RAW INPUT (multiple consecutive transcriptions of the same rolling window — "
        f"deduplicate and extract only the genuinely new spoken content as one subtitle):\n{raw_text}"
    )

    try:
        response: ChatResponse = chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            options=OLLAMA_OPTIONS,
        )
        return response.message.content.strip()
    except Exception as exc:
        print(f"⚠️  OLLAMA cleanup error: {exc}")
        return None


def ensure_ollama_ready() -> None:
    """
    Pulls Ollama model is necessary, checks model is downloaded
    """
    try:
        local = _ollama.list()
    except Exception as exc:
        raise RuntimeError(
            f"Cannot reach Ollama — is the server running?  ({exc})"
        ) from exc
    model_names: List[str] = [m.model for m in local.models]
    if not any(name.startswith(OLLAMA_MODEL) for name in model_names):
        print(f"   '{OLLAMA_MODEL}' not found locally — pulling (this may take a while) ...")
        try:
            _ollama.pull(OLLAMA_MODEL)
            print("   Pull complete.")
        except Exception as exc:
            raise RuntimeError(f"Failed to pull model '{OLLAMA_MODEL}': {exc}") from exc
    else:
        print(f"   Model found locally.")
    print("   Warming up model, almost done ...")
    try:
        chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": "Ready?"}],
            options=OLLAMA_OPTIONS,
        )
        print("   ✅ Ollama is ready.")
    except Exception as exc:
        raise RuntimeError(f"Ollama warm-up failed: {exc}") from exc

_LLM_EMPTY_SENTINELS: frozenset = frozenset({
    "empty string", "empty", "(empty)", "[empty]",
    "(empty string)", "[empty string]", "(none)", "none", "n/a",
})


def normalize_llm_output(text: str) -> str:
    if text.strip().lower().rstrip(".") in _LLM_EMPTY_SENTINELS:
        return ""
    return text


def is_hallucination(text: str) -> bool:
    """
    Algorithmic hallucination detection by checking if the output from whisper is unusually long
    given sliding window length, or if there are too many repeating words/phrases

    False-alarms generally do not impact quality since the same information is likely captured in the
    previous subtitle
    """
    words = text.split()
    if not words:
        return False
    max_expected = int(BUFFER_SECONDS * 4.5)
    if len(words) > max_expected:
        print(f"🔴 Hallucination (too long: {len(words)} words > {max_expected}): {text[:60]!r}")
        return True
    clean = [re.sub(r"[^\w']+", "", w).lower() for w in words]
    clean = [w for w in clean if w]
    for n in [2, 3]:
        if len(clean) < n * 3:
            continue
        ngrams = [" ".join(clean[i : i + n]) for i in range(len(clean) - n + 1)]
        top, count = Counter(ngrams).most_common(1)[0]
        if count >= 3:
            print(f"🔴 Hallucination (\'{top}\' x{count}): {text[:60]!r}")
            return True
    top, count = Counter(clean).most_common(1)[0]
    if count >= 4 and count / len(clean) > 0.40:
        print(f"🔴 Hallucination (\'{top}\' x{count}, {count/len(clean):.0%}): {text[:60]!r}")
        return True
    return False


def llm_processing_loop() -> None:
    print(f"LLM cleanup thread started (model={OLLAMA_MODEL})")
    while True:
        try:
            raw_text: str = llm_input_queue.get(timeout=1)
        except queue.Empty:
            continue

        with subtitle_context_lock:
            context = list(subtitle_context)

        cleaned: Optional[str] = cleanup_subtitle_with_ollama(raw_text, context)

        if cleaned is None:
            cleaned = raw_text
        else:
            cleaned = normalize_llm_output(cleaned)

        if cleaned:
            with subtitle_context_lock:
                subtitle_context.append(cleaned)
            print(f"🔵 (cleaned) {cleaned}")
            broadcast_subtitle(cleaned)
        else:
            print("🟡 (LLM: no new content)")


def run_whisper(audio_np: np.ndarray) -> str:
    transcribe_kwargs: Dict[str, Any] = {"task": WHISPER_TASK, "beam_size": WHISPER_BEAM_SIZE}
    if WHISPER_LANGUAGE:
        transcribe_kwargs["language"] = WHISPER_LANGUAGE
    assert model is not None, "Whisper model is not initialized"
    segments, _info = model.transcribe(audio_np, **transcribe_kwargs)
    text = " ".join(seg.text for seg in segments).strip()
    if not text:
        return text

    print(f"🟢 (raw) {text}")

    if is_hallucination(text):
        return text

    if USE_OLLAMA_CLEANUP:
        with _raw_batch_lock:
            _raw_batch.append(text)
            if len(_raw_batch) >= RAW_BATCH_SIZE:
                batch_text = "\n".join(_raw_batch)
                _raw_batch.clear()
            else:
                batch_text = None
        if batch_text is not None:
            try:
                llm_input_queue.put_nowait(batch_text)
            except queue.Full:
                try:
                    llm_input_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    llm_input_queue.put_nowait(batch_text)
                except queue.Full:
                    pass
    else:
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
    """
    Creates an SSE event raw payload
    """
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
    """
    Run flask app to expose processed data as a server-sent-events (subtitle)
    """
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
    """
    Get all audio devices
    """
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
    """
    Callback definition for audio sink. Unload all data into global audio_buffer
    """
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
    """
    Basic rudimentary silence detection, do not run whisper if rms value isn't reached
    """
    if audio_16k is None or len(audio_16k) == 0:
        return False
    rms: float = float(np.sqrt(np.mean(np.square(audio_16k))))  # root mean square
    return rms < 0.003


def processing_loop() -> None:
    """
    Core logic for processing incoming data
    Capture Audio -> If not silent -> Run Whisper on audio buffer
    """
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
    """
    Attempts to automatically identify the sample rate of audio sink. Otherwise prompt for input
    """
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
    global BUFFER_SECONDS, PROCESS_INTERVAL_SECONDS, USE_OLLAMA_CLEANUP
    global OLLAMA_CONTEXT_WINDOW, RAW_BATCH_SIZE, subtitle_context
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

    USE_OLLAMA_CLEANUP = bool(settings.get("use_ollama_cleanup", True))
    OLLAMA_OPTIONS["num_gpu"] = 0 if settings.get("ollama_device", "CPU").upper() == "CPU" else 1
    OLLAMA_CONTEXT_WINDOW = int(settings.get("ollama_context_window", 6))
    subtitle_context = deque(maxlen=OLLAMA_CONTEXT_WINDOW)
    RAW_BATCH_SIZE = int(settings.get("ollama_raw_batch_size", 3))
    if USE_OLLAMA_CLEANUP:
        ensure_ollama_ready()
        llm_thread = threading.Thread(target=llm_processing_loop, daemon=True)
        llm_thread.start()

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
    print(f"Ollama cleanup: {'enabled' if USE_OLLAMA_CLEANUP else 'disabled'} (model={OLLAMA_MODEL})")

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
