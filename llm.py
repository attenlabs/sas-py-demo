"""OpenAI Realtime bridge — sample-only, NOT part of attenlabs-sas.

Takes base64 PCM16 audio from `sas.AttentionClient`'s `speech_ready` event,
forwards it to OpenAI's Realtime API, and plays the response back through the
local speaker via sounddevice.
"""

from __future__ import annotations

import base64
import json
import logging
import threading
import time
from typing import Callable, Optional

import numpy as np
import sounddevice as sd
import websocket

DEFAULT_URL = "wss://api.openai.com/v1/realtime?model=gpt-realtime-2025-08-28"
DEFAULT_VOICE = "sage"
OUTPUT_SAMPLE_RATE = 24000  # OpenAI Realtime's output rate
DEFAULT_GAIN_DB = 6.0

logger = logging.getLogger("sas_demo.llm")


class RealtimeLLMBridge:
    def __init__(
        self,
        api_key: str,
        *,
        url: str = DEFAULT_URL,
        voice: str = DEFAULT_VOICE,
        instructions: str = "You are a helpful assistant.",
        gain_db: float = DEFAULT_GAIN_DB,
        temperature: float = 0.8,
        output_device: Optional[int] = None,
    ):
        if not api_key:
            raise ValueError("RealtimeLLMBridge: api_key required")
        self.api_key = api_key
        self.url = url
        self.voice = voice
        self.instructions = instructions
        self.gain_db = gain_db
        self.temperature = temperature
        self.output_device = output_device

        self.ws: Optional[websocket.WebSocketApp] = None
        self.ws_thread: Optional[threading.Thread] = None
        self.session_ready = False
        self.audio_chunks: list[str] = []
        self.pending_audio: Optional[str] = None
        self.response_timer: Optional[float] = None
        self.closed = False

        self._listeners: dict[str, list[Callable]] = {}

    def on(self, event: str, func: Callable) -> Callable:
        self._listeners.setdefault(event, []).append(func)
        return func

    def _emit(self, event: str, *args) -> None:
        for fn in self._listeners.get(event, []):
            try:
                fn(*args)
            except Exception:
                logger.exception("llm listener '%s' raised", event)

    def send_audio_b64(self, audio_b64: str) -> None:
        self.pending_audio = audio_b64
        self.closed = False
        if self.session_ready and self.ws is not None:
            self._flush()
            return
        self._connect()

    def _connect(self) -> None:
        if self.ws_thread is not None and self.ws_thread.is_alive():
            return
        self.session_ready = False
        headers = [
            f"Authorization: Bearer {self.api_key}",
            "OpenAI-Beta: realtime=v1",
        ]
        self.ws = websocket.WebSocketApp(
            self.url,
            header=headers,
            on_open=self._on_open,
            on_message=self._on_message,
            on_close=self._on_close,
            on_error=self._on_error,
        )
        self.ws_thread = threading.Thread(
            target=self.ws.run_forever, daemon=True, name="llm-ws",
        )
        self.ws_thread.start()

    def _on_open(self, ws) -> None:
        ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "modalities": ["audio", "text"],
                "instructions": self.instructions,
                "voice": self.voice,
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {"model": "whisper-1"},
                "turn_detection": None,
                "tool_choice": "auto",
                "temperature": self.temperature,
                "max_response_output_tokens": "inf",
            },
        }))

    def _on_message(self, ws, message) -> None:
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            return
        t = data.get("type")
        if t in ("session.created", "session.updated"):
            if not self.session_ready:
                self.session_ready = True
                self._flush()
        elif t == "response.audio.delta":
            delta = data.get("delta")
            if delta:
                self.audio_chunks.append(delta)
        elif t == "response.audio.done":
            self._playback()
        elif t == "response.audio_transcript.done":
            self._emit("transcript", data.get("transcript", ""))
        elif t == "error":
            err = data.get("error") or {}
            self._emit("error", {
                "title": "LLM Error",
                "message": err.get("message") or str(data),
            })
            self._emit("speaking_end")

    def _on_close(self, ws, code, reason) -> None:
        self.session_ready = False
        if self.pending_audio and not self.closed:
            self.pending_audio = None
            self._emit("error", {
                "title": "LLM Disconnected",
                "message": "LLM connection dropped mid-request.",
            })
            self._emit("speaking_end")

    def _on_error(self, ws, error) -> None:
        logger.debug("llm ws error: %s", error)

    def _flush(self) -> None:
        if not self.pending_audio or self.ws is None:
            return
        audio = self.pending_audio
        self.pending_audio = None
        self.response_timer = time.monotonic()
        try:
            self.ws.send(json.dumps({
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_audio", "audio": audio}],
                },
            }))
            self.ws.send(json.dumps({"type": "response.create"}))
        except Exception as e:
            self._emit("error", {"title": "LLM Send Error", "message": str(e)})
            self._emit("speaking_end")

    def _playback(self) -> None:
        chunks = self.audio_chunks
        self.audio_chunks = []
        if not chunks:
            self._emit("speaking_end")
            return

        raw = b"".join(base64.b64decode(c) for c in chunks)
        pcm16 = np.frombuffer(raw, dtype=np.int16).copy()
        if pcm16.size == 0:
            self._emit("speaking_end")
            return

        gain = 10 ** (self.gain_db / 20.0)
        out = np.clip(pcm16.astype(np.float32) * gain, -32768, 32767).astype(np.int16)

        if self.response_timer is not None:
            dt = time.monotonic() - self.response_timer
            logger.info("llm response time: %.2fs", dt)

        self._emit("speaking_start")

        def play():
            try:
                sd.play(out, samplerate=OUTPUT_SAMPLE_RATE, device=self.output_device)
                sd.wait()
            except Exception:
                logger.exception("llm playback error")
            finally:
                self._emit("speaking_end")

        threading.Thread(target=play, daemon=True, name="llm-playback").start()

    def close(self) -> None:
        self.closed = True
        self.pending_audio = None
        if self.ws is not None:
            try:
                self.ws.close()
            except Exception:
                pass
            self.ws = None
