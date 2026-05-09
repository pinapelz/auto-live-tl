from typing import Any, Callable, Dict, List

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QGroupBox, QLabel, QTextEdit, QVBoxLayout, QWidget

from gui.gui_common import ensure_qt_app


AudioActivityProvider = Callable[[], Dict[str, Any]]
RuntimeLogLinesProvider = Callable[[], List[str]]
SubtitleLinesProvider = Callable[[], List[str]]


class _RuntimeDashboard(QWidget):
    def __init__(
        self,
        get_audio_activity: AudioActivityProvider,
        get_runtime_logs: RuntimeLogLinesProvider,
        get_subtitle_lines: SubtitleLinesProvider,
        on_close: Callable[[], None],
    ) -> None:
        super().__init__()
        self._get_audio_activity = get_audio_activity
        self._get_runtime_logs = get_runtime_logs
        self._get_subtitle_lines = get_subtitle_lines
        self._on_close = on_close
        self._closed = False
        self._last_rendered_runtime_logs: str = ""
        self._last_rendered_final_logs: str = ""

        self.setWindowTitle("auto-live-tl")
        self.setMinimumSize(1100, 700)

        layout = QVBoxLayout(self)

        title = QLabel("auto-live-tl", self)
        title.setStyleSheet("font-size: 22px; font-weight: 700; color: #000000;")
        layout.addWidget(title)

        self.audio_indicator = QLabel("⚪ Idle", self)
        self.audio_indicator.setStyleSheet("font-size: 16px; color: #b0b0b0; font-weight: 600;")
        layout.addWidget(self.audio_indicator)

        self.audio_details = QLabel("RMS 0.00000 | threshold 0.00300", self)
        self.audio_details.setStyleSheet("font-size: 13px; color: #9aa0a6;")
        layout.addWidget(self.audio_details)

        raw_group = QGroupBox("Debug Log (It's recommended to fetch the final data via the SSE API, see the README)", self)
        raw_group_layout = QVBoxLayout(raw_group)

        raw_title = QLabel("System / Raw Output", raw_group)
        raw_group_layout.addWidget(raw_title)

        self.runtime_log_view = QTextEdit(raw_group)
        self.runtime_log_view.setReadOnly(True)
        self.runtime_log_view.setPlaceholderText("Waiting for raw Whisper output...")
        self.runtime_log_view.setStyleSheet(
            """
            QTextEdit {
                background: #111417;
                color: #d8dee9;
                border: 1px solid #2f3742;
                border-radius: 8px;
                padding: 8px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 13px;
                line-height: 1.4;
            }
            """
        )
        raw_group_layout.addWidget(self.runtime_log_view, 3)

        final_title = QLabel("Final (Sent via SSE)", raw_group)
        raw_group_layout.addWidget(final_title)

        self.final_log_view = QTextEdit(raw_group)
        self.final_log_view.setReadOnly(True)
        self.final_log_view.setPlaceholderText("Waiting for FINAL output...")
        self.final_log_view.setStyleSheet(
            """
            QTextEdit {
                background: #0f1410;
                color: #dcf9dd;
                border: 1px solid #2f4a35;
                border-radius: 8px;
                padding: 8px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 14px;
                font-weight: 700;
                line-height: 1.6;
            }
            """
        )
        raw_group_layout.addWidget(self.final_log_view, 2)

        layout.addWidget(raw_group, 1)

        self._timer = QTimer(self)
        self._timer.setInterval(150)
        self._timer.timeout.connect(self._refresh)
        self._timer.start()
        self._refresh()

    def _shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._timer.stop()
        try:
            self._on_close()
        except Exception:
            pass

    def closeEvent(self, event: Any) -> None:  # type: ignore[override]
        self._shutdown()
        super().closeEvent(event)

    def _refresh(self) -> None:
        try:
            activity = self._get_audio_activity()
        except Exception:
            activity = {}

        active = bool(activity.get("active", False))
        try:
            rms = float(activity.get("rms", 0.0))
        except (TypeError, ValueError):
            rms = 0.0
        try:
            threshold = float(activity.get("threshold", 0.0))
        except (TypeError, ValueError):
            threshold = 0.0

        if active:
            self.audio_indicator.setText("🟢 Audio detected")
            self.audio_indicator.setStyleSheet("font-size: 16px; color: #8fd18f; font-weight: 600;")
        else:
            self.audio_indicator.setText("⚪ Idle")
            self.audio_indicator.setStyleSheet("font-size: 16px; color: #b0b0b0; font-weight: 600;")
        self.audio_details.setText(f"RMS {rms:.5f} | threshold {threshold:.5f}")

        try:
            logs = self._get_runtime_logs()
        except Exception:
            logs = []
        runtime_lines = [line for line in logs if "[FINAL]" not in line]
        final_lines = [line for line in logs if "[FINAL]" in line]

        joined_runtime_logs = "\n".join(runtime_lines)
        if joined_runtime_logs != self._last_rendered_runtime_logs:
            self._last_rendered_runtime_logs = joined_runtime_logs
            self.runtime_log_view.setPlainText(joined_runtime_logs)
            log_scroll = self.runtime_log_view.verticalScrollBar()
            log_scroll.setValue(log_scroll.maximum())

        joined_final_logs = "\n\n".join(final_lines)
        if joined_final_logs != self._last_rendered_final_logs:
            self._last_rendered_final_logs = joined_final_logs
            self.final_log_view.setPlainText(joined_final_logs)
            final_scroll = self.final_log_view.verticalScrollBar()
            final_scroll.setValue(final_scroll.maximum())


def run_runtime_dashboard(
    get_audio_activity: AudioActivityProvider,
    get_runtime_logs: RuntimeLogLinesProvider,
    get_subtitle_lines: SubtitleLinesProvider,
    on_close: Callable[[], None],
) -> None:
    app = ensure_qt_app()

    dashboard = _RuntimeDashboard(
        get_audio_activity=get_audio_activity,
        get_runtime_logs=get_runtime_logs,
        get_subtitle_lines=get_subtitle_lines,
        on_close=on_close,
    )
    dashboard.show()
    app.exec()
