from typing import cast
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication


def ensure_qt_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    app = cast(QApplication, app)
    app.setFont(QFont("Calibri", 12))
    return app
