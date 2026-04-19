# auto-live-tl
A basic LOCAL translation backend that listens to an audio sink via PCM and runs translation via faster-whisper

Translations a

`server.py` serves a backend for translating incoming audio data. It expects some other client to hit the `/events` endpoint to fetch the translated data.
