# attenlabs-sas-demo
Python SDK for [Attention Labs](https://attentionlabs.ai) real-time selective auditory attention.

Every voice pipeline has the same problem: the microphone hears everything, but your ASR should only process speech directed at the device. Wake words solve this with a rigid trigger phrase. SAS solves it without one.

attenlabs-sas streams mic and webcam data to the SAS inference server over WebSocket and emits typed events: attention predictions, voice activity, conversation state, and ready-to-forward speech audio. LLM routing is left to you. This is a CLI sample app that demonstrates it.


[reachy_demo.webm](https://github.com/user-attachments/assets/14c5a350-9059-4ac7-bba9-92dca01feb69)

## What you'll need

- A SAS auth token (sign up on the dashboard [here](https://attentionlabs.ai/dashboard/))
- Python 3.10+
- A microphone and webcam
- An OpenAI API key with Realtime access *(optional — omit to run without the LLM stage and just see live predictions)*

## Install

```bash
pip install attenlabs-sas cv2-enumerate-cameras
# or, from a clone of this repo:
pip install .
```

If you're on Linux and don't already have PortAudio, install it first:

```bash
sudo apt-get install libportaudio2
```

## Run

```bash
python main.py --token YOUR_TOKEN --openai-key sk-...
```

Or via environment variables:

```bash
export OPENAI_API_KEY=sk-...
python main.py --token YOUR_TOKEN
```

## CLI options

```
--token             SAS auth token (required)
--url               Override the default SAS server URL (for self-hosted SAS)
--openai-key        OpenAI API key; falls back to OPENAI_API_KEY env var
--camera-index      Webcam device index (default 0)
--mic-device        Mic device name or numeric index (default: system default)
--threshold         Device-class trigger threshold 0..1 (default 0.7)
--no-video          Disable webcam capture
--no-audio          Disable mic capture
--no-llm            Disable LLM stage even if a key is set
--log-level         DEBUG, INFO, WARNING, ERROR (default WARNING)
```

## SDK docs

Full API reference — constructor, methods, events, types, threading model — lives in [**DOCS.md**](DOCS.md).

## How it works

1. [`main.py`](main.py) constructs an `AttentionClient` from [`attenlabs-sas`](https://pypi.org/project/attenlabs-sas/), which starts mic + webcam capture threads and opens a WebSocket to the SAS server.
2. The SDK emits events — `prediction`, `vad`, `state`, and `speech_ready` — which the CLI prints in real time.
3. On `speech_ready`, `main.py` hands the PCM16 audio to [`llm.py`](llm.py), a small OpenAI Realtime bridge that sends it to OpenAI and plays the response back through your speaker.
4. While the LLM is speaking, `main.py` calls `client.mute()` + `client.mark_responding(True)` so the server stops emitting predictions until playback ends.

The LLM bridge is deliberately part of this demo, not the SDK — swap in whichever provider you like.

## Security note

This demo reads the OpenAI API key from a CLI arg or env var and uses it directly from the local process. Fine for personal use. For anything multi-user, proxy the Realtime connection through a server you control so keys never leave your backend.

## License

MIT
