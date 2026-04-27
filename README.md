# auto-live-tl
A basic LOCAL translation backend that listens to an audio sink via PCM and runs translation via faster-whisper. Also supports the option to use `qwen2.5-7b-instruct` to format/clean-up subtitles based on sliding window context.

Translations and trascriptions are transformers based, inaccuracies and hallucinations will occur.


# Setup
> It's highly recommended that you run this with a GPU, running with CPU is possible but inference will be very slow outside of using tiny models (which compromise accuracy)
>
> For this, you will need to install a Nvidia CUDA 12 toolkit. I am running with [CUDA Toolkit 12.9](https://developer.nvidia.com/cuda-12-9-0-download-archive)

```
uv sync
uv run server.py
```
A GUI is available for configuration

`server.py` serves a backend for translating incoming audio data. It expects some other client to hit the `/events` endpoint to fetch the translated data.


# Clients:

`youtube-subtitle.user.js` is one such example client that can fetch data from this endpoint and render it beneath a YouTube video. You can install it as a userscript.

<img width="1210" height="109" alt="image" src="https://github.com/user-attachments/assets/2bffde45-bc61-4d63-b779-b7a8cd183bc0" />
