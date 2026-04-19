# auto-live-tl
A basic LOCAL translation backend that listens to an audio sink via PCM and runs translation via faster-whisper

Translations and trascriptions are transformers based, inaccuracies and hallucinations will occur.

`server.py` serves a backend for translating incoming audio data. It expects some other client to hit the `/events` endpoint to fetch the translated data.

`youtube-subtitle.user.js` is one such example client that can fetch data from this endpoint and render it beneath a YouTube video. You can install it as a userscript.

<img width="1210" height="109" alt="image" src="https://github.com/user-attachments/assets/2bffde45-bc61-4d63-b779-b7a8cd183bc0" />
