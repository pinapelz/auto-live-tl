from typing import Any, Callable, List, Optional, Tuple, TypeVar, cast
from queue import Empty, Queue
import threading
import time

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QProgressBar, QVBoxLayout

from gui.gui.gui_common import ensure_qt_app


T = TypeVar("T")
StatusCallback = Callable[[str], None]


class _LoadingDialog(QDialog):
    def __init__(self, title: str, initial_message: str) -> None:
        super().__init__()
        self.setWindowTitle(title)
        self.setModal(True)
        self.setFixedWidth(440)

        layout = QVBoxLayout(self)

        self._spinner_frames: List[str] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self._spinner_index = 0

        spinner_row = QHBoxLayout()
        self._spinner_label = QLabel(self._spinner_frames[0], self)
        self._spinner_label.setStyleSheet("font-size: 20px; font-weight: 700; color: #8fd18f;")
        spinner_row.addWidget(self._spinner_label)

        self._message_label = QLabel(initial_message, self)
        self._message_label.setWordWrap(True)
        self._message_label.setStyleSheet("font-size: 13px;")
        spinner_row.addWidget(self._message_label, 1)
        layout.addLayout(spinner_row)

        self._progress = QProgressBar(self)
        self._progress.setRange(0, 0)
        self._progress.setTextVisible(False)
        layout.addWidget(self._progress)

        self._hint_label = QLabel("Please wait…", self)
        self._hint_label.setStyleSheet("font-size: 12px; color: #9aa0a6;")
        layout.addWidget(self._hint_label)

        self._spinner_timer = QTimer(self)
        self._spinner_timer.setInterval(90)
        self._spinner_timer.timeout.connect(self._tick_spinner)
        self._spinner_timer.start()

    def _tick_spinner(self) -> None:
        self._spinner_index = (self._spinner_index + 1) % len(self._spinner_frames)
        self._spinner_label.setText(self._spinner_frames[self._spinner_index])

    def set_message(self, message: str) -> None:
        self._message_label.setText(message)


def run_with_loading_popup(
    title: str,
    initial_message: str,
    task: Callable[[StatusCallback], T],
) -> T:
    app = ensure_qt_app()

    dialog = _LoadingDialog(title=title, initial_message=initial_message)
    events: Queue[Tuple[str, Any]] = Queue()

    def publish_status(message: str) -> None:
        events.put(("status", message))

    def worker() -> None:
        try:
            result = task(publish_status)
            events.put(("result", result))
        except Exception as exc:
            events.put(("error", exc))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    dialog.show()
    dialog.raise_()
    dialog.activateWindow()

    done = False
    result_value: Optional[T] = None
    error: Optional[Exception] = None

    while not done:
        app.processEvents()

        while True:
            try:
                event_type, payload = events.get_nowait()
            except Empty:
                break

            if event_type == "status":
                dialog.set_message(str(payload))
            elif event_type == "result":
                result_value = cast(T, payload)
                done = True
            elif event_type == "error":
                error = cast(Exception, payload)
                done = True

        if thread.is_alive() and not done:
            time.sleep(0.03)
            continue

        if not thread.is_alive():
            try:
                event_type, payload = events.get_nowait()
                if event_type == "status":
                    dialog.set_message(str(payload))
                elif event_type == "result":
                    result_value = cast(T, payload)
                elif event_type == "error":
                    error = cast(Exception, payload)
            except Empty:
                pass
            done = True

    dialog.close()
    app.processEvents()

    if error is not None:
        raise error

    return cast(T, result_value)
