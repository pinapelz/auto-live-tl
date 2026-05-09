import base64
import json
import queue
import re
import threading
import time
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import websocket


AddRuntimeLogFn = Callable[[str, str], None]
BroadcastSubtitleFn = Callable[[str], None]
ResampleAudioFn = Callable[[np.ndarray, int, int], np.ndarray]


class OpenAIRealtimeTranslator:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        output_language: str,
        safety_identifier: str,
        add_runtime_log: AddRuntimeLogFn,
        broadcast_subtitle: BroadcastSubtitleFn,
        resample_audio: ResampleAudioFn,
        target_sample_rate: int = 24000,
        reconnect_seconds: float = 2.0,
        buffer_stale_seconds: float = 1.1,
        ws_url_template: str = "wss://api.openai.com/v1/realtime/translations?model={model}",
        queue_maxsize: int = 200,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.output_language = output_language
        self.safety_identifier = safety_identifier

        self.target_sample_rate = target_sample_rate
        self.reconnect_seconds = reconnect_seconds
        self.buffer_stale_seconds = buffer_stale_seconds
        self.ws_url_template = ws_url_template

        self._add_runtime_log = add_runtime_log
        self._broadcast_subtitle = broadcast_subtitle
        self._resample_audio = resample_audio

        self._audio_queue: queue.Queue = queue.Queue(maxsize=queue_maxsize)
        self._stop_sentinel: object = object()
        self._stop_event: threading.Event = threading.Event()
        self._transcript_buffer: str = ""
        self._last_delta_monotonic: float = 0.0
        self._transcript_lock: threading.Lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    @staticmethod
    def _float_audio_to_pcm16_base64(audio_np: np.ndarray) -> str:
        if len(audio_np) == 0:
            return ""
        clipped = np.clip(audio_np, -1.0, 1.0)
        pcm16 = (clipped * 32767.0).astype(np.int16)
        return base64.b64encode(pcm16.tobytes()).decode("ascii")

    @staticmethod
    def _normalize_subtitle_chunk(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def _extract_completed_sentences(self, buffer: str) -> Tuple[List[str], str]:
        completed: List[str] = []
        remaining = buffer

        while True:
            match = re.search(r"(.+?[.!?])(?=\s|$)", remaining, flags=re.DOTALL)
            if not match:
                break
            sentence = self._normalize_subtitle_chunk(match.group(1))
            if sentence:
                completed.append(sentence)
            remaining = remaining[match.end() :].lstrip()

        if "\n" in remaining:
            parts = [part.strip() for part in remaining.split("\n")]
            for part in parts[:-1]:
                normalized_part = self._normalize_subtitle_chunk(part)
                if normalized_part:
                    completed.append(normalized_part)
            remaining = parts[-1] if parts else ""

        return completed, remaining

    @staticmethod
    def _clear_queue(target_queue: queue.Queue) -> None:
        while True:
            try:
                target_queue.get_nowait()
            except queue.Empty:
                break

    def _flush_transcript_buffer(self, force: bool = False) -> None:
        with self._transcript_lock:
            text = self._normalize_subtitle_chunk(self._transcript_buffer)
            if not text:
                self._transcript_buffer = ""
                return
            if not force and len(text) < 2:
                return
            self._transcript_buffer = ""

        self._add_runtime_log("FINAL", text)
        self._broadcast_subtitle(text)

    def _flush_transcript_buffer_if_stale(self) -> None:
        with self._transcript_lock:
            if not self._transcript_buffer:
                return
            elapsed = time.monotonic() - self._last_delta_monotonic
            if elapsed < self.buffer_stale_seconds:
                return
        self._flush_transcript_buffer(force=True)

    def _handle_transcript_delta(self, delta: str) -> None:
        if not delta:
            return

        delta = delta.replace("\r", "")

        with self._transcript_lock:
            self._last_delta_monotonic = time.monotonic()
            self._transcript_buffer += delta
            completed, remaining = self._extract_completed_sentences(self._transcript_buffer)

            if len(remaining) > 180:
                split_idx = remaining.rfind(" ")
                if split_idx > 80:
                    overflow = self._normalize_subtitle_chunk(remaining[:split_idx])
                    if overflow:
                        completed.append(overflow)
                    remaining = remaining[split_idx:].lstrip()

            self._transcript_buffer = remaining

        for sentence in completed:
            self._add_runtime_log("FINAL", sentence)
            self._broadcast_subtitle(sentence)

    def _audio_sender_loop(self, ws: websocket.WebSocket) -> None:
        while not self._stop_event.is_set():
            try:
                item = self._audio_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if item is self._stop_sentinel:
                break
            if not isinstance(item, str) or not item:
                continue

            payload = {
                "type": "session.input_audio_buffer.append",
                "audio": item,
            }
            try:
                ws.send(json.dumps(payload))
            except Exception:
                break

    def _run_loop(self) -> None:
        ws_url = self.ws_url_template.format(model=self.model)

        while not self._stop_event.is_set():
            ws: Any = None
            sender_thread: Optional[threading.Thread] = None

            try:
                headers: List[str] = [f"Authorization: Bearer {self.api_key}"]
                if self.safety_identifier:
                    headers.append(f"OpenAI-Safety-Identifier: {self.safety_identifier}")

                ws = websocket.WebSocket()
                ws.connect(ws_url, header=headers)
                ws.settimeout(0.6)

                session_update = {
                    "type": "session.update",
                    "session": {
                        "audio": {
                            "output": {
                                "language": self.output_language,
                            },
                        },
                    },
                }
                ws.send(json.dumps(session_update))
                self._add_runtime_log(
                    "OPENAI",
                    f"Connected to realtime translation (lang={self.output_language}, model={self.model})",
                )

                sender_thread = threading.Thread(target=self._audio_sender_loop, args=(ws,), daemon=True)
                sender_thread.start()

                while not self._stop_event.is_set():
                    try:
                        incoming = ws.recv()
                    except websocket.WebSocketTimeoutException:
                        self._flush_transcript_buffer_if_stale()
                        continue

                    if incoming is None:
                        break
                    incoming = incoming.strip()
                    if not incoming:
                        continue

                    try:
                        event = json.loads(incoming)
                    except json.JSONDecodeError:
                        self._add_runtime_log("OPENAI", "Received non-JSON event from realtime translation socket")
                        continue

                    event_type = str(event.get("type", ""))
                    if event_type == "session.output_transcript.delta":
                        delta = str(event.get("delta", ""))
                        self._handle_transcript_delta(delta)
                    elif event_type in {"session.output_transcript.done", "session.output_transcript.completed"}:
                        self._flush_transcript_buffer(force=True)
                    elif event_type in {"error", "session.error"}:
                        self._add_runtime_log("OPENAI", f"Realtime API error: {json.dumps(event, ensure_ascii=False)}")
                    elif event_type == "session.updated":
                        self._add_runtime_log("OPENAI", "Realtime session configured")

            except Exception as exc:
                if self._stop_event.is_set():
                    break
                self._add_runtime_log("OPENAI", f"Realtime connection failed: {exc}")
                time.sleep(self.reconnect_seconds)
            finally:
                if ws is not None:
                    try:
                        ws.close()
                    except Exception:
                        pass
                self._flush_transcript_buffer(force=True)
                if sender_thread is not None and sender_thread.is_alive():
                    sender_thread.join(timeout=1.0)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return

        self._clear_queue(self._audio_queue)
        self._stop_event.clear()
        with self._transcript_lock:
            self._transcript_buffer = ""
            self._last_delta_monotonic = 0.0

        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        try:
            self._audio_queue.put_nowait(self._stop_sentinel)
        except queue.Full:
            self._clear_queue(self._audio_queue)
            try:
                self._audio_queue.put_nowait(self._stop_sentinel)
            except queue.Full:
                pass

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def enqueue_audio_chunk(self, chunk: np.ndarray, capture_sample_rate: int) -> None:
        if capture_sample_rate <= 0 or len(chunk) == 0:
            return

        resampled = self._resample_audio(chunk, capture_sample_rate, self.target_sample_rate)
        encoded = self._float_audio_to_pcm16_base64(resampled)
        if not encoded:
            return

        try:
            self._audio_queue.put_nowait(encoded)
        except queue.Full:
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._audio_queue.put_nowait(encoded)
            except queue.Full:
                pass
