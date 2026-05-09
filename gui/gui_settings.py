from typing import Iterable, List, Tuple, Dict, Any, Optional
import time

import numpy as np
import sounddevice as sd
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from gui.gui_common import ensure_qt_app


class _SettingsDialog(QDialog):
    def __init__(
        self,
        settings: Dict[str, Any],
        input_devices: List[Tuple[int, Dict[str, Any]]],
        default_settings: Dict[str, Any],
        model_choices: Iterable[str],
        device_choices: Iterable[str],
        compute_choices: Iterable[str],
        task_choices: Iterable[str],
    ) -> None:
        super().__init__()
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.setMinimumWidth(700)

        self.selected_settings: Dict[str, Any] = {}

        def get_value(key: str, fallback: Any) -> Any:
            return settings.get(key, default_settings.get(key, fallback))

        self.device_indices = [idx for idx, _dev in input_devices]
        self.device_names = [dev["name"] for _idx, dev in input_devices]

        self._monitor_stream: Optional[sd.InputStream] = None
        self._monitor_rms: float = 0.0
        self._monitor_active_until: float = 0.0
        self._monitor_error: str = ""
        self._monitor_threshold: float = float(get_value("audio_activity_threshold", 0.003))

        root_layout = QVBoxLayout(self)

        tabs = QTabWidget(self)
        root_layout.addWidget(tabs)

        # Whisper tab
        whisper_tab = QWidget(self)
        whisper_tab_layout = QVBoxLayout(whisper_tab)

        whisper_layout = QFormLayout()
        whisper_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)

        device_options = [
            f"[{idx}] {dev['name']} ({dev.get('max_input_channels', 0)} ch)"
            for idx, dev in input_devices
        ]
        self.device_combo = QComboBox(whisper_tab)
        self.device_combo.addItems(device_options)
        self.device_combo.setEditable(False)
        default_device_name = get_value("audio_device_name", "")
        if default_device_name in self.device_names:
            self.device_combo.setCurrentIndex(self.device_names.index(default_device_name))
        else:
            self.device_combo.setCurrentIndex(0)
        whisper_layout.addRow(QLabel("Audio input device:"), self.device_combo)

        self.model_combo = QComboBox(whisper_tab)
        self.model_combo.addItems(list(model_choices))
        self.model_combo.setEditable(True)
        default_model = str(get_value("model_name", "medium"))
        if default_model in [self.model_combo.itemText(i) for i in range(self.model_combo.count())]:
            self.model_combo.setCurrentText(default_model)
        else:
            self.model_combo.setEditText(default_model)
        whisper_layout.addRow(QLabel("Model:"), self.model_combo)

        self.device_type_combo = QComboBox(whisper_tab)
        self.device_type_combo.addItems(list(device_choices))
        self.device_type_combo.setEditable(False)
        default_device_type = str(get_value("device", "cpu"))
        if default_device_type in [self.device_type_combo.itemText(i) for i in range(self.device_type_combo.count())]:
            self.device_type_combo.setCurrentText(default_device_type)
        elif self.device_type_combo.count() > 0:
            self.device_type_combo.setCurrentIndex(0)
        whisper_layout.addRow(QLabel("Compute device:"), self.device_type_combo)

        self.task_combo = QComboBox(whisper_tab)
        self.task_combo.addItems(list(task_choices))
        self.task_combo.setEditable(False)
        default_task = str(get_value("task", "translate"))
        if default_task in [self.task_combo.itemText(i) for i in range(self.task_combo.count())]:
            self.task_combo.setCurrentText(default_task)
        elif self.task_combo.count() > 0:
            self.task_combo.setCurrentIndex(0)
        whisper_layout.addRow(QLabel("Task:"), self.task_combo)

        whisper_tab_layout.addLayout(whisper_layout)

        whisper_advanced_group = QGroupBox("Advanced settings", whisper_tab)
        whisper_advanced_layout = QFormLayout(whisper_advanced_group)
        whisper_advanced_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)

        self.compute_type_combo = QComboBox(whisper_tab)
        self.compute_type_combo.addItems(list(compute_choices))
        self.compute_type_combo.setEditable(True)
        default_compute = str(get_value("compute_type", "int8"))
        if default_compute in [self.compute_type_combo.itemText(i) for i in range(self.compute_type_combo.count())]:
            self.compute_type_combo.setCurrentText(default_compute)
        else:
            self.compute_type_combo.setEditText(default_compute)
        whisper_advanced_layout.addRow(QLabel("Compute type:"), self.compute_type_combo)

        self.beam_size_edit = QLineEdit(str(get_value("beam_size", 3)), whisper_tab)
        whisper_advanced_layout.addRow(QLabel("Beam size:"), self.beam_size_edit)

        self.language_edit = QLineEdit(str(get_value("language", "")), whisper_tab)
        whisper_advanced_layout.addRow(QLabel("Language (optional):"), self.language_edit)

        self.context_seconds_edit = QLineEdit(str(get_value("context_seconds", 10)), whisper_tab)
        whisper_advanced_layout.addRow(QLabel("Context seconds:"), self.context_seconds_edit)

        self.update_interval_edit = QLineEdit(str(get_value("update_interval_seconds", 2)), whisper_tab)
        whisper_advanced_layout.addRow(QLabel("Update interval (s):"), self.update_interval_edit)

        self.audio_activity_threshold_edit = QLineEdit(str(get_value("audio_activity_threshold", 0.003)), whisper_tab)
        whisper_advanced_layout.addRow(QLabel("Audio activity threshold (RMS):"), self.audio_activity_threshold_edit)

        self.audio_indicator_label = QLabel("⚪ Idle", whisper_tab)
        self.audio_indicator_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        whisper_advanced_layout.addRow(QLabel("Live input indicator:"), self.audio_indicator_label)

        whisper_tab_layout.addWidget(whisper_advanced_group)
        tabs.addTab(whisper_tab, "Whisper")

        # Ollama tab
        ollama_tab = QWidget(self)
        ollama_tab_layout = QVBoxLayout(ollama_tab)

        ollama_layout = QFormLayout()
        ollama_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)

        self.use_ollama_cleanup_checkbox = QCheckBox(ollama_tab)
        self.use_ollama_cleanup_checkbox.setChecked(bool(get_value("use_ollama_cleanup", True)))
        ollama_layout.addRow(QLabel("LLM subtitle cleanup:"), self.use_ollama_cleanup_checkbox)

        self.ollama_device_combo = QComboBox(ollama_tab)
        self.ollama_device_combo.addItems(["CPU", "GPU"])
        self.ollama_device_combo.setEditable(False)
        default_ollama_device = str(get_value("ollama_device", "CPU"))
        if default_ollama_device in [self.ollama_device_combo.itemText(i) for i in range(self.ollama_device_combo.count())]:
            self.ollama_device_combo.setCurrentText(default_ollama_device)
        ollama_layout.addRow(QLabel("Ollama compute:"), self.ollama_device_combo)

        ollama_tab_layout.addLayout(ollama_layout)

        ollama_advanced_group = QGroupBox("Advanced settings", ollama_tab)
        ollama_advanced_layout = QFormLayout(ollama_advanced_group)
        ollama_advanced_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)

        self.ollama_model_edit = QLineEdit(str(get_value("ollama_model", "qwen2.5:7b-instruct")), ollama_tab)
        ollama_advanced_layout.addRow(QLabel("Ollama model:"), self.ollama_model_edit)

        self.ollama_context_edit = QLineEdit(str(get_value("ollama_context_window", 6)), ollama_tab)
        ollama_advanced_layout.addRow(QLabel("Context window (segments):"), self.ollama_context_edit)

        self.ollama_batch_edit = QLineEdit(str(get_value("ollama_raw_batch_size", 3)), ollama_tab)
        ollama_advanced_layout.addRow(QLabel("Batch size (lines per LLM call):"), self.ollama_batch_edit)

        ollama_tab_layout.addWidget(ollama_advanced_group)
        tabs.addTab(ollama_tab, "Ollama")

        # OpenAI Realtime tab
        openai_tab = QWidget(self)
        openai_tab_layout = QVBoxLayout(openai_tab)

        openai_layout = QFormLayout()
        openai_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)

        self.use_openai_realtime_checkbox = QCheckBox(openai_tab)
        self.use_openai_realtime_checkbox.setChecked(bool(get_value("use_openai_realtime_translate", False)))
        openai_layout.addRow(QLabel("Use OpenAI realtime translation:"), self.use_openai_realtime_checkbox)

        self.openai_api_key_edit = QLineEdit(str(get_value("openai_api_key", "")), openai_tab)
        self.openai_api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.openai_api_key_edit.setPlaceholderText("sk-...")
        openai_layout.addRow(QLabel("OpenAI API key:"), self.openai_api_key_edit)

        self.openai_output_language_edit = QLineEdit(str(get_value("openai_output_language", "es")), openai_tab)
        self.openai_output_language_edit.setPlaceholderText("es")
        openai_layout.addRow(QLabel("Target language code:"), self.openai_output_language_edit)

        self.openai_model_edit = QLineEdit(str(get_value("openai_model", "gpt-realtime-translate")), openai_tab)
        self.openai_model_edit.setPlaceholderText("gpt-realtime-translate")
        openai_layout.addRow(QLabel("Realtime model:"), self.openai_model_edit)

        self.openai_safety_identifier_edit = QLineEdit(str(get_value("openai_safety_identifier", "")), openai_tab)
        self.openai_safety_identifier_edit.setPlaceholderText("optional hashed-user-id")
        openai_layout.addRow(QLabel("OpenAI-Safety-Identifier (optional):"), self.openai_safety_identifier_edit)

        openai_tab_layout.addLayout(openai_layout)

        self.openai_hint_label = QLabel(
            "When enabled, source audio is streamed to OpenAI /v1/realtime/translations and subtitle SSE events are produced from realtime transcript output. Ollama cleanup is bypassed.",
            openai_tab,
        )
        self.openai_hint_label.setWordWrap(True)
        self.openai_hint_label.setStyleSheet("font-size: 12px; color: #9aa0a6;")
        openai_tab_layout.addWidget(self.openai_hint_label)

        tabs.addTab(openai_tab, "OpenAI Realtime")

        button_layout = QHBoxLayout()
        root_layout.addLayout(button_layout)
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        button_layout.addWidget(button_box)

        self.device_combo.currentIndexChanged.connect(self._restart_monitor_stream)
        self.audio_activity_threshold_edit.textChanged.connect(self._on_threshold_changed)
        self.use_openai_realtime_checkbox.toggled.connect(self._sync_backend_controls)

        self._monitor_timer = QTimer(self)
        self._monitor_timer.setInterval(120)
        self._monitor_timer.timeout.connect(self._refresh_audio_indicator)
        self._monitor_timer.start()

        self._restart_monitor_stream()
        self._sync_backend_controls(self.use_openai_realtime_checkbox.isChecked())
        self._refresh_audio_indicator()

    def _warn(self, title: str, text: str) -> None:
        QMessageBox.warning(self, title, text)

    def _on_threshold_changed(self, text: str) -> None:
        try:
            parsed = float(text.strip())
            if parsed > 0:
                self._monitor_threshold = parsed
        except ValueError:
            pass

    def _sync_backend_controls(self, use_openai: bool) -> None:
        self.openai_api_key_edit.setEnabled(use_openai)
        self.openai_output_language_edit.setEnabled(use_openai)
        self.openai_model_edit.setEnabled(use_openai)
        self.openai_safety_identifier_edit.setEnabled(use_openai)

        self.use_ollama_cleanup_checkbox.setEnabled(not use_openai)
        self.ollama_device_combo.setEnabled(not use_openai)
        self.ollama_model_edit.setEnabled(not use_openai)
        self.ollama_context_edit.setEnabled(not use_openai)
        self.ollama_batch_edit.setEnabled(not use_openai)

        if use_openai:
            self.use_ollama_cleanup_checkbox.setChecked(False)

    def _pick_monitor_sample_rate(self, device_index: int, preferred_rate: int) -> Optional[int]:
        common_rates: List[int] = [48000, 44100, 32000, 24000, 22050, 16000, 12000, 8000]
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
        return None

    def _monitor_callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        if status:
            self._monitor_error = f"Audio status: {status}"
        if indata is None or len(indata) == 0:
            return

        chunk = indata[:, 0]
        rms = float(np.sqrt(np.mean(np.square(chunk))))
        self._monitor_rms = rms
        if rms >= self._monitor_threshold:
            self._monitor_active_until = time.monotonic() + 0.6

    def _refresh_audio_indicator(self) -> None:
        if self._monitor_error:
            self.audio_indicator_label.setText(f"⚠ {self._monitor_error}")
            self.audio_indicator_label.setStyleSheet("color: #f28b82;")
            return

        active = time.monotonic() <= self._monitor_active_until
        rms_text = f"{self._monitor_rms:.5f}"
        if active:
            self.audio_indicator_label.setText(f"🟢 Audio detected (RMS {rms_text})")
            self.audio_indicator_label.setStyleSheet("color: #8fd18f;")
        else:
            self.audio_indicator_label.setText(f"⚪ Idle (RMS {rms_text})")
            self.audio_indicator_label.setStyleSheet("color: #b0b0b0;")

    def _stop_monitor_stream(self) -> None:
        stream = self._monitor_stream
        self._monitor_stream = None
        if stream is None:
            return
        try:
            stream.stop()
        except Exception:
            pass
        try:
            stream.close()
        except Exception:
            pass

    def _restart_monitor_stream(self, *_args: Any) -> None:
        self._stop_monitor_stream()
        self._monitor_error = ""
        self._monitor_rms = 0.0
        self._monitor_active_until = 0.0

        selection = self.device_combo.currentIndex()
        if selection < 0 or selection >= len(self.device_indices):
            self._monitor_error = "No input device selected."
            return

        device_index = self.device_indices[selection]
        try:
            device_info = sd.query_devices(device_index)
        except Exception as exc:
            self._monitor_error = f"Could not read device info: {exc}"
            return

        preferred_rate = int(float(device_info.get("default_samplerate", 48000)))
        if preferred_rate <= 0:
            preferred_rate = 48000

        sample_rate = self._pick_monitor_sample_rate(device_index, preferred_rate)
        if sample_rate is None:
            self._monitor_error = "No supported sample rate for monitor stream."
            return

        blocksize = max(256, int(sample_rate * 0.1))
        try:
            stream = sd.InputStream(
                device=device_index,
                channels=1,
                samplerate=sample_rate,
                dtype="float32",
                callback=self._monitor_callback,
                blocksize=blocksize,
            )
            stream.start()
            self._monitor_stream = stream
        except Exception as exc:
            self._monitor_error = f"Unable to start monitor: {exc}"

    def accept(self) -> None:
        selection = self.device_combo.currentIndex()
        if selection < 0:
            self._warn("Select a device", "Please select an audio input device.")
            return

        model_name = self.model_combo.currentText().strip()
        if not model_name:
            self._warn("Model required", "Please select or enter a model name.")
            return

        try:
            beam_size = int(self.beam_size_edit.text().strip())
            if beam_size <= 0:
                raise ValueError
        except ValueError:
            self._warn("Invalid beam size", "Beam size must be a positive integer.")
            return

        try:
            context_seconds = float(self.context_seconds_edit.text().strip())
            if context_seconds <= 0:
                raise ValueError
        except ValueError:
            self._warn("Invalid context seconds", "Context seconds must be a positive number.")
            return

        try:
            update_interval_seconds = float(self.update_interval_edit.text().strip())
            if update_interval_seconds <= 0:
                raise ValueError
        except ValueError:
            self._warn("Invalid update interval", "Update interval must be a positive number.")
            return

        try:
            audio_activity_threshold = float(self.audio_activity_threshold_edit.text().strip())
            if audio_activity_threshold <= 0:
                raise ValueError
        except ValueError:
            self._warn("Invalid audio threshold", "Audio activity threshold must be a positive number.")
            return

        try:
            ollama_context_window = int(self.ollama_context_edit.text().strip())
            if ollama_context_window <= 0:
                raise ValueError
        except ValueError:
            self._warn("Invalid context window", "Context window must be a positive integer.")
            return

        try:
            ollama_raw_batch_size = int(self.ollama_batch_edit.text().strip())
            if ollama_raw_batch_size <= 0:
                raise ValueError
        except ValueError:
            self._warn("Invalid batch size", "Batch size must be a positive integer.")
            return

        use_openai_realtime = self.use_openai_realtime_checkbox.isChecked()
        openai_api_key = self.openai_api_key_edit.text().strip()
        openai_output_language = self.openai_output_language_edit.text().strip()
        openai_model = self.openai_model_edit.text().strip() or "gpt-realtime-translate"
        openai_safety_identifier = self.openai_safety_identifier_edit.text().strip()

        if use_openai_realtime and not openai_api_key:
            self._warn("OpenAI API key required", "Please provide your OpenAI API key to use realtime translation.")
            return
        if use_openai_realtime and not openai_output_language:
            self._warn("Target language required", "Please provide a target language code (example: es, fr, ja).")
            return

        self.selected_settings = {
            "audio_device_name": self.device_names[selection],
            "model_name": model_name,
            "device": self.device_type_combo.currentText().strip() or "cpu",
            "compute_type": self.compute_type_combo.currentText().strip() or "int8",
            "task": self.task_combo.currentText().strip() or "translate",
            "beam_size": beam_size,
            "language": self.language_edit.text().strip(),
            "context_seconds": context_seconds,
            "update_interval_seconds": update_interval_seconds,
            "audio_activity_threshold": audio_activity_threshold,
            "use_ollama_cleanup": self.use_ollama_cleanup_checkbox.isChecked() and not use_openai_realtime,
            "ollama_device": self.ollama_device_combo.currentText(),
            "ollama_model": self.ollama_model_edit.text().strip(),
            "ollama_context_window": ollama_context_window,
            "ollama_raw_batch_size": ollama_raw_batch_size,
            "use_openai_realtime_translate": use_openai_realtime,
            "openai_api_key": openai_api_key,
            "openai_output_language": openai_output_language or "es",
            "openai_model": openai_model,
            "openai_safety_identifier": openai_safety_identifier,
        }
        self._monitor_timer.stop()
        self._stop_monitor_stream()
        super().accept()

    def reject(self) -> None:
        self._monitor_timer.stop()
        self._stop_monitor_stream()
        super().reject()


def select_settings(
    settings: Dict[str, Any],
    input_devices: List[Tuple[int, Dict[str, Any]]],
    default_settings: Dict[str, Any],
    model_choices: Iterable[str],
    device_choices: Iterable[str],
    compute_choices: Iterable[str],
    task_choices: Iterable[str],
) -> Dict[str, Any]:
    if not input_devices:
        raise RuntimeError("No audio input devices found.")

    ensure_qt_app()

    dialog = _SettingsDialog(
        settings=settings,
        input_devices=input_devices,
        default_settings=default_settings,
        model_choices=model_choices,
        device_choices=device_choices,
        compute_choices=compute_choices,
        task_choices=task_choices,
    )
    result = dialog.exec()

    if result != int(QDialog.DialogCode.Accepted) or not dialog.selected_settings:
        raise SystemExit("No settings selected.")
    return dialog.selected_settings


def prompt_input_sample_rate(device_index: int, common_rates: Iterable[int]) -> int:
    ensure_qt_app()
    rates = list(common_rates)
    while True:
        prompt = (
            "Enter an input sample rate in Hz.\n"
            f"Common values: {', '.join(str(r) for r in rates)}"
        )
        raw, ok = QInputDialog.getText(None, "Select Sample Rate", prompt)
        if not ok:
            raise sd.PortAudioError("No supported input sample rate found for selected device.")

        raw = raw.strip()
        if not raw:
            continue

        try:
            rate = int(float(raw))
        except ValueError:
            QMessageBox.warning(None, "Invalid value", "Sample rate must be a number.")
            continue

        try:
            sd.check_input_settings(device=device_index, channels=1, samplerate=rate, dtype="float32")
            return rate
        except sd.PortAudioError:
            QMessageBox.warning(
                None,
                "Unsupported sample rate",
                f"{rate} Hz is not supported by the selected device.",
            )
