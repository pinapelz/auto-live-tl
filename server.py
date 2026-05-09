import time
import threading
import json
import queue
import os
from collections import Counter, deque
import re
from typing import Any, Deque, Dict, Optional, Set, List, Iterator, Callable

from flask import Flask
from flask_cors import CORS
import ollama as _ollama
from ollama import chat
from ollama import ChatResponse
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

from gui.gui import select_settings, prompt_input_sample_rate, run_runtime_dashboard, run_with_loading_popup
from openai_realtime import OpenAIRealtimeTranslator
from routes import register_routes
from config import _SYSTEM_PROMPT, _LLM_EMPTY_SENTINELS, _HALLUCINATION_PHRASES

TARGET_SAMPLE_RATE: int = 16000
CAPTURE_SAMPLE_RATE: int = 0
BUFFER_SECONDS: float = 10
MAX_SAMPLES: int = 0
PROCESS_INTERVAL_SECONDS: float = 2
SSE_EVENT_SUBTITLE: str = "subtitle"
SSE_EVENT_AUDIO_ACTIVITY: str = "audio_activity"
SSE_KEEPALIVE_SECONDS: int = 15
RUNTIME_SUBTITLE_LINES_MAX: int = 120
RUNTIME_LOG_LINES_MAX: int = 300

USE_OLLAMA_CLEANUP: bool = True
USE_OPENAI_REALTIME_TRANSLATE: bool = False
OLLAMA_MODEL: str = "qwen2.5:7b-instruct"
OLLAMA_CONTEXT_WINDOW: int = 6  # number of recent cleaned segments kept as context
OLLAMA_OPTIONS: Dict[str, Any] = {"num_gpu": 1}
RAW_BATCH_SIZE: int = 2  # accumulate this many raw Whisper lines before calling the LLM

OPENAI_REALTIME_MODEL: str = "gpt-realtime-translate"
OPENAI_API_KEY: str = ""
OPENAI_OUTPUT_LANGUAGE: str = "es"
OPENAI_SAFETY_IDENTIFIER: str = ""

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
    "audio_activity_threshold": 0.003,
    "use_ollama_cleanup": True,
    "ollama_device": "GPU",
    "ollama_model": "qwen2.5:7b-instruct",
    "ollama_context_window": 5,
    "ollama_raw_batch_size": 1,
    "use_openai_realtime_translate": False,
    "openai_api_key": "",
    "openai_output_language": "en",
    "openai_model": "gpt-realtime-translate",
    "openai_safety_identifier": "",
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
AUDIO_ACTIVITY_THRESHOLD: float = float(DEFAULT_SETTINGS["audio_activity_threshold"])
AUDIO_ACTIVITY_HOLD_SECONDS: float = 0.75
AUDIO_ACTIVITY_REPORT_INTERVAL_SECONDS: float = 0.5

last_subtitle_payload: Optional[Dict[str, Any]] = None
last_audio_activity_payload: Dict[str, Any] = {
    "active": False,
    "rms": 0.0,
    "threshold": AUDIO_ACTIVITY_THRESHOLD,
}
_audio_active_until: float = 0.0
_audio_last_emit: float = 0.0
_audio_state_lock: threading.Lock = threading.Lock()
recent_subtitle_lines: Deque[str] = deque(maxlen=RUNTIME_SUBTITLE_LINES_MAX)
recent_subtitle_lines_lock: threading.Lock = threading.Lock()
runtime_logs: Deque[str] = deque(maxlen=RUNTIME_LOG_LINES_MAX)
runtime_logs_lock: threading.Lock = threading.Lock()
clients: Set[queue.Queue] = set()
clients_lock: threading.Lock = threading.Lock()
SERVER_HOST: str = "127.0.0.1"
SERVER_PORT: int = 5000
app: Flask = Flask(__name__)
CORS(app)

# OLLAMA stuff
llm_input_queue: queue.Queue = queue.Queue(maxsize=1)
subtitle_context: Deque[str] = deque(maxlen=OLLAMA_CONTEXT_WINDOW)  # sliding window context
subtitle_context_lock: threading.Lock = threading.Lock()
_raw_batch: List[str] = []
_raw_batch_lock: threading.Lock = threading.Lock()

openai_realtime_client: Optional[OpenAIRealtimeTranslator] = None

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


def ensure_ollama_ready(status_callback: Optional[Callable[[str], None]] = None) -> None:
    """
    Pulls Ollama model is necessary, checks model is downloaded
    """
    def report(message: str) -> None:
        print(message)
        if status_callback is not None:
            status_callback(message.strip())

    report("Checking Ollama server availability...")
    try:
        local = _ollama.list()
    except Exception as exc:
        raise RuntimeError(
            f"Cannot reach Ollama — is the server running?  ({exc})"
        ) from exc

    model_names: List[str] = [m.model for m in local.models]
    if not any(name.startswith(OLLAMA_MODEL) for name in model_names):
        report(f"Model '{OLLAMA_MODEL}' not found locally. Pulling now (this can take a while)...")
        try:
            _ollama.pull(OLLAMA_MODEL)
            report("Model pull complete.")
        except Exception as exc:
            raise RuntimeError(f"Failed to pull model '{OLLAMA_MODEL}': {exc}") from exc
    else:
        report("Model found locally.")

    report("Warming up Ollama model...")
    try:
        chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": "Ready?"}],
            options=OLLAMA_OPTIONS,
        )
        report("✅ Ollama is ready.")
    except Exception as exc:
        raise RuntimeError(f"Ollama warm-up failed: {exc}") from exc


def normalize_llm_output(text: str) -> str:
    if text.strip().lower().rstrip(".") in _LLM_EMPTY_SENTINELS:
        return ""
    return text


def add_runtime_log(kind: str, message: str) -> None:
    timestamp = time.strftime("%H:%M:%S")
    line = f"[{timestamp}] [{kind.upper()}] {message}"
    with runtime_logs_lock:
        runtime_logs.append(line)


def is_hallucination(text: str) -> Optional[str]:
    """
    Algorithmic hallucination detection by checking if the output from whisper is unusually long
    given sliding window length, or if there are too many repeating words/phrases

    False-alarms generally do not impact quality since the same information is likely captured in the
    previous subtitle
    """
    words = text.split()
    if not words:
        return None
    max_expected = int(BUFFER_SECONDS * 4.5)
    if len(words) > max_expected:
        return f"too long: {len(words)} words > {max_expected}"
    clean = [re.sub(r"[^\w']+", "", w).lower() for w in words]
    clean = [w for w in clean if w]
    for n in [2, 3]:
        if len(clean) < n * 3:
            continue
        ngrams = [" ".join(clean[i : i + n]) for i in range(len(clean) - n + 1)]
        top, count = Counter(ngrams).most_common(1)[0]
        if count >= 3:
            tokens_covered = count * n
            if tokens_covered / max(1, len(clean)) > 0.35:
                return f"repeating phrase '{top}' x{count} (covers {tokens_covered}/{len(clean)} tokens)"
    top, count = Counter(clean).most_common(1)[0]
    if count >= 4 and count / len(clean) > 0.40:
        return f"repeating token '{top}' x{count} ({count/len(clean):.0%})"

    normalized = re.sub(r"[^\w\s]", "", text.lower()).strip()
    if normalized in _HALLUCINATION_PHRASES:
        return "blocked phrase pattern"

    return None


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
            add_runtime_log("LLM", "cleanup failed, falling back to raw text")
            cleaned = raw_text
        else:
            cleaned = normalize_llm_output(cleaned)

        if cleaned:
            with subtitle_context_lock:
                subtitle_context.append(cleaned)
            add_runtime_log("FINAL", cleaned)
            broadcast_subtitle(cleaned)
        else:
            add_runtime_log("LLM", "no new content from cleanup")


def run_whisper(audio_np: np.ndarray) -> str:
    transcribe_kwargs: Dict[str, Any] = {"task": WHISPER_TASK, "beam_size": WHISPER_BEAM_SIZE}
    if WHISPER_LANGUAGE:
        transcribe_kwargs["language"] = WHISPER_LANGUAGE
    assert model is not None, "Whisper model is not initialized"
    segments, _info = model.transcribe(audio_np, **transcribe_kwargs)
    text = " ".join(seg.text for seg in segments).strip()
    if not text:
        return text

    add_runtime_log("RAW", text)

    hallucination_reason = is_hallucination(text)
    if hallucination_reason:
        add_runtime_log("HALLUCINATION", f"{hallucination_reason} | text={text}")
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
            add_runtime_log("RAW->LLM", batch_text.replace("\n", " || "))
            try:
                llm_input_queue.put_nowait(batch_text)
            except queue.Full:
                add_runtime_log("LLM", "queue full, dropping previous batch")
                try:
                    llm_input_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    llm_input_queue.put_nowait(batch_text)
                except queue.Full:
                    add_runtime_log("LLM", "queue still full, skipped batch")
    else:
        add_runtime_log("FINAL", text)
        broadcast_subtitle(text)

    return text


def broadcast_event(event: str, payload: Dict[str, Any]) -> None:
    message: Dict[str, Any] = {"event": event, "payload": payload}
    with clients_lock:
        targets = list(clients)
    for client_queue in targets:
        try:
            client_queue.put_nowait(message)
        except queue.Full:
            pass


def broadcast_subtitle(text: str) -> None:
    global last_subtitle_payload
    payload: Dict[str, Any] = {"text": text}
    last_subtitle_payload = payload
    with recent_subtitle_lines_lock:
        if not recent_subtitle_lines or recent_subtitle_lines[-1] != text:
            recent_subtitle_lines.append(text)
    broadcast_event(SSE_EVENT_SUBTITLE, payload)


def get_audio_activity_snapshot() -> Dict[str, Any]:
    with _audio_state_lock:
        return dict(last_audio_activity_payload)


def get_recent_subtitle_lines_snapshot() -> List[str]:
    with recent_subtitle_lines_lock:
        return list(recent_subtitle_lines)


def get_runtime_logs_snapshot() -> List[str]:
    with runtime_logs_lock:
        return list(runtime_logs)


def format_sse_event(event: str, payload: Dict[str, Any]) -> str:
    """
    Creates an SSE event raw payload
    """
    data = json.dumps(payload)
    return f"event: {event}\ndata: {data}\n\n"


def event_stream() -> Iterator[str]:
    client_queue: queue.Queue = queue.Queue(maxsize=20)
    with clients_lock:
        clients.add(client_queue)

    if last_subtitle_payload:
        yield format_sse_event(SSE_EVENT_SUBTITLE, last_subtitle_payload)
    yield format_sse_event(SSE_EVENT_AUDIO_ACTIVITY, last_audio_activity_payload)

    try:
        while True:
            try:
                event_data = client_queue.get(timeout=SSE_KEEPALIVE_SECONDS)
            except queue.Empty:
                yield ": keep-alive\n\n"
                continue
            if not isinstance(event_data, dict):
                continue
            event_name = str(event_data.get("event", SSE_EVENT_SUBTITLE))
            payload = event_data.get("payload", {})
            if not isinstance(payload, dict):
                payload = {}
            yield format_sse_event(event_name, payload)
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


def publish_audio_activity(chunk_rms: float) -> None:
    global _audio_active_until, _audio_last_emit, last_audio_activity_payload

    now_mono = time.monotonic()
    if chunk_rms >= AUDIO_ACTIVITY_THRESHOLD:
        _audio_active_until = now_mono + AUDIO_ACTIVITY_HOLD_SECONDS

    with _audio_state_lock:
        active = now_mono <= _audio_active_until
        previous_active = bool(last_audio_activity_payload.get("active", False))
        state_changed = active != previous_active
        report_due = (now_mono - _audio_last_emit) >= AUDIO_ACTIVITY_REPORT_INTERVAL_SECONDS

        if not state_changed and not report_due:
            return

        payload: Dict[str, Any] = {
            "active": active,
            "rms": round(chunk_rms, 6),
            "threshold": AUDIO_ACTIVITY_THRESHOLD,
        }
        last_audio_activity_payload = payload
        _audio_last_emit = now_mono

    broadcast_event(SSE_EVENT_AUDIO_ACTIVITY, payload)


def audio_callback(indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
    """
    Callback definition for audio sink. Sends audio to local Whisper buffer or OpenAI realtime queue.
    """
    if status:
        print(f"Audio status: {status}")
    # Take first channel
    chunk: np.ndarray = indata[:, 0].copy()
    chunk_rms: float = float(np.sqrt(np.mean(np.square(chunk)))) if len(chunk) > 0 else 0.0
    publish_audio_activity(chunk_rms)

    if USE_OPENAI_REALTIME_TRANSLATE:
        if openai_realtime_client is not None:
            openai_realtime_client.enqueue_audio_chunk(chunk, CAPTURE_SAMPLE_RATE)
        return

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
    return rms < AUDIO_ACTIVITY_THRESHOLD


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
    global BUFFER_SECONDS, PROCESS_INTERVAL_SECONDS, USE_OLLAMA_CLEANUP, USE_OPENAI_REALTIME_TRANSLATE
    global OLLAMA_MODEL, OLLAMA_CONTEXT_WINDOW, RAW_BATCH_SIZE, subtitle_context
    global AUDIO_ACTIVITY_THRESHOLD, last_audio_activity_payload, _audio_active_until, _audio_last_emit
    global OPENAI_API_KEY, OPENAI_OUTPUT_LANGUAGE, OPENAI_REALTIME_MODEL, OPENAI_SAFETY_IDENTIFIER
    global openai_realtime_client

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

    USE_OPENAI_REALTIME_TRANSLATE = bool(settings.get("use_openai_realtime_translate", False))
    OPENAI_REALTIME_MODEL = str(settings.get("openai_model", OPENAI_REALTIME_MODEL)).strip() or OPENAI_REALTIME_MODEL
    OPENAI_OUTPUT_LANGUAGE = str(settings.get("openai_output_language", "es")).strip() or "es"
    OPENAI_SAFETY_IDENTIFIER = str(settings.get("openai_safety_identifier", "")).strip()
    OPENAI_API_KEY = str(settings.get("openai_api_key", "")).strip() or str(os.environ.get("OPENAI_API_KEY", "")).strip()

    USE_OLLAMA_CLEANUP = bool(settings.get("use_ollama_cleanup", True)) and not USE_OPENAI_REALTIME_TRANSLATE
    OLLAMA_OPTIONS["num_gpu"] = 0 if settings.get("ollama_device", "CPU").upper() == "CPU" else 1
    OLLAMA_MODEL = str(settings.get("ollama_model", OLLAMA_MODEL)).strip() or OLLAMA_MODEL
    OLLAMA_CONTEXT_WINDOW = int(settings.get("ollama_context_window", 6))
    subtitle_context = deque(maxlen=OLLAMA_CONTEXT_WINDOW)
    RAW_BATCH_SIZE = int(settings.get("ollama_raw_batch_size", 3))

    if USE_OPENAI_REALTIME_TRANSLATE and not OPENAI_API_KEY:
        raise RuntimeError("OpenAI realtime translation is enabled, but no API key was provided.")

    if USE_OLLAMA_CLEANUP:
        run_with_loading_popup(
            title="Preparing Ollama model",
            initial_message="Checking model availability...",
            task=ensure_ollama_ready,
        )
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
    AUDIO_ACTIVITY_THRESHOLD = float(settings.get("audio_activity_threshold", AUDIO_ACTIVITY_THRESHOLD))
    if BUFFER_SECONDS <= 0:
        BUFFER_SECONDS = DEFAULT_SETTINGS["context_seconds"]
    if PROCESS_INTERVAL_SECONDS <= 0:
        PROCESS_INTERVAL_SECONDS = DEFAULT_SETTINGS["update_interval_seconds"]
    if AUDIO_ACTIVITY_THRESHOLD <= 0:
        AUDIO_ACTIVITY_THRESHOLD = float(DEFAULT_SETTINGS["audio_activity_threshold"])

    last_audio_activity_payload = {
        "active": False,
        "rms": 0.0,
        "threshold": AUDIO_ACTIVITY_THRESHOLD,
    }
    _audio_active_until = 0.0
    _audio_last_emit = 0.0
    broadcast_event(SSE_EVENT_AUDIO_ACTIVITY, last_audio_activity_payload)

    device_info = sd.query_devices(device_index)
    preferred_rate: int = int(device_info["default_samplerate"])
    if preferred_rate <= 0:
        preferred_rate = 48000
    CAPTURE_SAMPLE_RATE = select_input_sample_rate(device_index, preferred_rate)
    MAX_SAMPLES = int(CAPTURE_SAMPLE_RATE * BUFFER_SECONDS)

    with recent_subtitle_lines_lock:
        recent_subtitle_lines.clear()
    with runtime_logs_lock:
        runtime_logs.clear()

    add_runtime_log("SYSTEM", "Runtime dashboard started")
    add_runtime_log("SYSTEM", f"Device: {device_info['name']} @ {CAPTURE_SAMPLE_RATE} Hz")

    processing_thread: Optional[threading.Thread] = None

    if USE_OPENAI_REALTIME_TRANSLATE:
        openai_realtime_client = OpenAIRealtimeTranslator(
            api_key=OPENAI_API_KEY,
            model=OPENAI_REALTIME_MODEL,
            output_language=OPENAI_OUTPUT_LANGUAGE,
            safety_identifier=OPENAI_SAFETY_IDENTIFIER,
            add_runtime_log=add_runtime_log,
            broadcast_subtitle=broadcast_subtitle,
            resample_audio=resample_audio,
        )
        openai_realtime_client.start()

        print(f"Using device {device_index}: {device_info['name']}")
        print(f"Realtime translation backend: OpenAI ({OPENAI_REALTIME_MODEL})")
        print(
            f"Capture sample rate: {CAPTURE_SAMPLE_RATE} Hz "
            f"(resampling to {openai_realtime_client.target_sample_rate} Hz for OpenAI)"
        )
        print(f"Audio activity threshold (RMS): {AUDIO_ACTIVITY_THRESHOLD}")
        print("Ollama cleanup: disabled (OpenAI realtime translation selected)")

        add_runtime_log("SYSTEM", f"Backend=OpenAI realtime | model={OPENAI_REALTIME_MODEL} | lang={OPENAI_OUTPUT_LANGUAGE}")
    else:
        openai_realtime_client = None
        model = WhisperModel(model_name, device=whisper_device, compute_type=compute_type)

        print(f"Using device {device_index}: {device_info['name']}")
        print(f"Model: {model_name} | task={WHISPER_TASK} | beam_size={WHISPER_BEAM_SIZE}")
        print(f"Compute: device={whisper_device} | compute_type={compute_type}")
        print(f"Capture sample rate: {CAPTURE_SAMPLE_RATE} Hz (resampling to {TARGET_SAMPLE_RATE} Hz)")
        print(f"Audio activity threshold (RMS): {AUDIO_ACTIVITY_THRESHOLD}")
        print(f"Ollama cleanup: {'enabled' if USE_OLLAMA_CLEANUP else 'disabled'} (model={OLLAMA_MODEL})")

        add_runtime_log("SYSTEM", f"Task={WHISPER_TASK} | Beam={WHISPER_BEAM_SIZE} | Cleanup={'on' if USE_OLLAMA_CLEANUP else 'off'}")

        processing_thread = threading.Thread(target=processing_loop, daemon=True)
        processing_thread.start()

    stream = sd.InputStream(
        device=device_index,
        channels=1,
        samplerate=CAPTURE_SAMPLE_RATE,
        dtype="float32",
        callback=audio_callback,
        blocksize=int(CAPTURE_SAMPLE_RATE * 0.5),
    )

    def _on_dashboard_close() -> None:
        print("Stopping.")

    try:
        stream.start()
        print("Listening... Close the runtime window to stop.")
        run_runtime_dashboard(
            get_audio_activity=get_audio_activity_snapshot,
            get_runtime_logs=get_runtime_logs_snapshot,
            get_subtitle_lines=get_recent_subtitle_lines_snapshot,
            on_close=_on_dashboard_close,
        )
    finally:
        if USE_OPENAI_REALTIME_TRANSLATE and openai_realtime_client is not None:
            openai_realtime_client.stop()
            openai_realtime_client = None

        try:
            stream.stop()
        except Exception:
            pass
        try:
            stream.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
