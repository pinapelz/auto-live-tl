from typing import Iterable, List, Tuple, Dict, Any, cast
import sounddevice as sd
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
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

        self.device_names = [dev["name"] for _idx, dev in input_devices]

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

        button_layout = QHBoxLayout()
        root_layout.addLayout(button_layout)
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        button_layout.addWidget(button_box)

    def _warn(self, title: str, text: str) -> None:
        QMessageBox.warning(self, title, text)

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
            "use_ollama_cleanup": self.use_ollama_cleanup_checkbox.isChecked(),
            "ollama_device": self.ollama_device_combo.currentText(),
            "ollama_model": self.ollama_model_edit.text().strip(),
            "ollama_context_window": ollama_context_window,
            "ollama_raw_batch_size": ollama_raw_batch_size,
        }
        super().accept()


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

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    app = cast(QApplication, app)
    app.setFont(QFont("Calibri", 12))

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
