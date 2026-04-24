from typing import Callable, Generator, Optional

from typing_extensions import Any
from flask import Blueprint, Response, stream_with_context, Flask

SSEGenerator = Callable[[], Generator[str, None, None]]


def make_sse_blueprint(
    event_stream_func: SSEGenerator,
    blueprint_name: str = "sse_routes",
    url_prefix: Optional[str] = None,
) -> Blueprint:
    bp = Blueprint(blueprint_name, __name__, url_prefix=url_prefix)

    @bp.get("/events")
    def events() -> Response:
        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        }
        return Response(stream_with_context(event_stream_func()), mimetype="text/event-stream", headers=headers)

    @bp.get("/health")
    def health() -> Response:
        resp = Response("ok", mimetype="text/plain")
        resp.headers["Access-Control-Allow-Origin"] = "*"
        return resp

    return bp


def register_routes(app: Flask, event_stream_func: Any, url_prefix: Optional[str] = None) -> None:
    bp = make_sse_blueprint(event_stream_func, url_prefix=url_prefix)
    app.register_blueprint(bp)
