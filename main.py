#!/usr/bin/env python3
"""CLI demo for attenlabs-sas.

Streams mic + webcam to the SD Attention Server, forwards detected speech to
OpenAI Realtime, and plays the response back through the local speaker.
"""

from __future__ import annotations

import argparse
import collections
import logging
import os
import re
import shutil
import sys
import threading
import time

import sounddevice as sd

from sas import AttentionClient, CameraConfig, MicConfig

from llm import RealtimeLLMBridge

CLASS_LABELS = {0: "silent", 1: "human", 2: "device"}

LLM_INSTRUCTIONS = (
    "You are a helpful assistant. Respond concisely in 1 sentence. "
    "If a device/TV command is spoken to you, respond as if you were controlling a TV."
)


# ── Terminal UI ─────────────────────────────────────────────────


def _vlen(s: str) -> int:
    """Visible string length, excluding ANSI escape sequences."""
    return len(re.sub(r'\033\[[0-9;]*m', '', s))


class TerminalUI:
    """Persistent ASCII dashboard for SD Attention predictions.

    Stays hidden during model warmup. Activates on first real prediction
    and redraws in-place every update.
    """

    _LABELS = {0: "SILENT", 1: "HUMAN", 2: "DEVICE"}
    _CLR = {"SILENT": "\033[90m", "HUMAN": "\033[33m", "DEVICE": "\033[36m"}
    _STATE_CLR = {
        "IDLE": "\033[90m",
        "LISTENING": "\033[32m",
        "SENDING": "\033[33m",
        "CANCELLED": "\033[31m",
    }
    _LLM_CLR = {"Idle": "\033[90m", "Processing": "\033[33m", "Speaking": "\033[36m"}
    RST = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    BAR_W = 20
    MAX_HISTORY = 10
    MAX_LOGS = 5

    def __init__(self):
        self._lock = threading.Lock()
        self._active = False
        self._history: collections.deque = collections.deque(maxlen=self.MAX_HISTORY)
        self._current: dict | None = None
        self._conv_state = "IDLE"
        self._llm_state = "Idle"
        self._logs: collections.deque = collections.deque(maxlen=self.MAX_LOGS)
        self._drawn = 0

    @property
    def active(self) -> bool:
        return self._active

    def activate(self):
        with self._lock:
            if self._active:
                return
            self._active = True
            sys.stdout.write("\033[?25l\033[2J\033[H")  # hide cursor, clear, home
            sys.stdout.flush()
            self._render()

    def deactivate(self):
        with self._lock:
            if not self._active:
                return
            self._active = False
            sys.stdout.write(f"\033[{self._drawn}B\033[?25h\n")
            sys.stdout.flush()

    def update_prediction(self, cls: int, confidence: float | None, faces: int = 0):
        label = self._LABELS.get(cls, "?")
        entry = {"label": label, "confidence": confidence, "faces": faces}
        with self._lock:
            self._current = entry
            self._history.appendleft(entry)
            if self._active:
                self._render()

    def update_conv_state(self, state: str):
        with self._lock:
            self._conv_state = state.upper()
            if self._active:
                self._render()

    def update_llm_state(self, state: str):
        with self._lock:
            self._llm_state = state
            if self._active:
                self._render()

    def log(self, msg: str):
        """Append to the log panel. Falls back to print() before UI is active."""
        with self._lock:
            ts = time.strftime("%H:%M:%S")
            self._logs.appendleft(f"{ts}  {msg}")
            if self._active:
                self._render()
            else:
                print(f"[{ts}] {msg}")

    def _bar(self, pct: float) -> str:
        filled = round(pct / 100 * self.BAR_W)
        return "█" * filled + "░" * (self.BAR_W - filled)

    def _pad(self, s: str, w: int) -> str:
        return s + " " * max(0, w - _vlen(s))

    def _center(self, s: str, w: int) -> str:
        gap = max(0, w - _vlen(s))
        left = gap // 2
        return " " * left + s + " " * (gap - left)

    def _row(self, content: str, w: int) -> str:
        return "│ " + self._pad(content, w - 4) + " │"

    def _sep(self, w: int, l: str = "├", r: str = "┤") -> str:
        return l + "─" * (w - 2) + r

    def _render(self):
        cols = shutil.get_terminal_size().columns
        w = max(min(cols, 60), 54)

        if self._drawn:
            sys.stdout.write(f"\033[{self._drawn}A\033[G")

        L: list[str] = []

        L.append(self._sep(w, "┌", "┐"))
        title = f"{self.BOLD}SD Attention · Monitor{self.RST}"
        L.append("│" + self._center(title, w - 2) + "│")

        L.append(self._sep(w))
        sc = self._STATE_CLR.get(self._conv_state, "")
        lc = self._LLM_CLR.get(self._llm_state, "")
        status = (f"State: {sc}{self._conv_state}{self.RST}"
                  f"    LLM: {lc}{self._llm_state}{self.RST}")
        L.append(self._row(status, w))

        L.append(self._sep(w))
        if self._current:
            p = self._current
            lbl = p["label"]
            c = self._CLR.get(lbl, "")
            pct = (p["confidence"] or 0) * 100
            cur = (f"▶ {c}{self.BOLD}{lbl:<7}{self.RST}"
                   f" {pct:5.1f}%  {self._bar(pct)}"
                   f"  faces: {p['faces']}")
            L.append(self._row(cur, w))
        else:
            L.append(self._row("▶ Waiting…", w))

        L.append(self._sep(w))
        L.append(self._row(f"{self.DIM}History{self.RST}", w))
        for p in self._history:
            lbl = p["label"]
            c = self._CLR.get(lbl, "")
            pct = (p["confidence"] or 0) * 100
            line = f"  {c}{lbl:<7}{self.RST} {pct:5.1f}%  {self._bar(pct)}"
            L.append(self._row(line, w))
        for _ in range(self.MAX_HISTORY - len(self._history)):
            L.append(self._row("", w))

        L.append(self._sep(w))
        L.append(self._row(f"{self.DIM}Log{self.RST}", w))
        logs = list(self._logs)[:self.MAX_LOGS]
        max_entry_len = w - 8
        for entry in logs:
            if len(entry) > max_entry_len:
                entry = entry[: max_entry_len - 1] + "…"
            L.append(self._row(f"  {self.DIM}{entry}{self.RST}", w))
        for _ in range(self.MAX_LOGS - len(logs)):
            L.append(self._row("", w))

        L.append(self._sep(w, "└", "┘"))

        sys.stdout.write("\n".join(L) + "\n")
        sys.stdout.flush()
        self._drawn = len(L)


# ── Device selection ────────────────────────────────────────────


def _list_microphones() -> list[dict]:
    """Return list of input audio devices."""
    devices = sd.query_devices()
    inputs = []
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            inputs.append({"index": i, "name": d["name"],
                           "channels": d["max_input_channels"],
                           "rate": d["default_samplerate"]})
    return inputs


def _list_cameras() -> list[dict]:
    """Enumerate cameras using cv2_enumerate_cameras."""
    from cv2_enumerate_cameras import enumerate_cameras
    cameras = []
    for cam in enumerate_cameras():
        cameras.append({"index": cam.index, "name": cam.name})
    return cameras


def _pick(label: str, items: list[dict], key: str = "index",
          name_key: str = "name") -> int | None:
    """Interactive numbered picker. Returns chosen value or None to skip."""
    if not items:
        print(f"  No {label} found.")
        return None

    for i, item in enumerate(items):
        extra = " | ".join(f"{k}={v}" for k, v in item.items()
                           if k not in (key, name_key))
        print(f"  [{i}] {item[name_key]}" + (f"  ({extra})" if extra else ""))
    print(f"  [s] Skip {label}")

    while True:
        choice = input(f"  Select {label} [0]: ").strip().lower()
        if choice == "s":
            return None
        if choice == "":
            return items[0][key]
        try:
            idx = int(choice)
            if 0 <= idx < len(items):
                return items[idx][key]
        except ValueError:
            pass
        print("  Invalid choice, try again.")


def _select_devices(args: argparse.Namespace) -> tuple[int | None, int | None]:
    """Interactive device selection. Returns (mic_index, camera_index)."""
    mic_index = None
    cam_index = None

    if not args.no_audio:
        print("\nAvailable microphones:")
        mics = _list_microphones()
        mic_index = _pick("microphone", mics)
        if mic_index is None:
            print("  Audio disabled.")

    if not args.no_video:
        print("\nAvailable cameras:")
        cams = _list_cameras()
        cam_index = _pick("camera", cams)
        if cam_index is None:
            print("  Video disabled.")

    print()
    return mic_index, cam_index


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="attenlabs-sas CLI demo")
    p.add_argument("--token", default=os.environ.get("SAS_TOKEN"),
                   help="SAS auth token (or set SAS_TOKEN env var)")
    p.add_argument("--url", default=None,
                   help="Override the SAS server URL (default: wss://server.attentionlabs.ai/ws)")
    p.add_argument("--openai-key", default=os.environ.get("OPENAI_API_KEY"),
                   help="OpenAI API key with Realtime access (env: OPENAI_API_KEY)")
    p.add_argument("--camera-index", type=int, default=None, help="Webcam device index (skip selector)")
    p.add_argument("--mic-device", default=None,
                   help="Mic device name or index (system default if unset)")
    p.add_argument("--threshold", type=float, default=0.7,
                   help="Device-class trigger threshold 0..1")
    p.add_argument("--no-video", action="store_true", help="Disable webcam capture")
    p.add_argument("--no-audio", action="store_true", help="Disable mic capture")
    p.add_argument("--no-llm", action="store_true",
                   help="Disable LLM stage even if --openai-key is set")
    p.add_argument("--log-level", default="WARNING",
                   help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.WARNING),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # Interactive device selection when flags aren't explicit
    mic_device = args.mic_device
    cam_index = args.camera_index

    if mic_device is not None:
        try:
            mic_device = int(mic_device)
        except ValueError:
            pass  # treat as device name string

    needs_mic_select = mic_device is None and not args.no_audio
    needs_cam_select = cam_index is None and not args.no_video

    if needs_mic_select or needs_cam_select:
        sel_mic, sel_cam = _select_devices(args)
        if needs_mic_select:
            mic_device = sel_mic
        if needs_cam_select:
            cam_index = sel_cam

    enable_audio = not args.no_audio and mic_device is not None
    enable_video = not args.no_video and cam_index is not None

    if not enable_audio and not enable_video:
        print("Both audio and video disabled — nothing to stream.")
        return 1

    # Set sounddevice default input device
    if enable_audio:
        sd.default.device[0] = mic_device

    if not args.token:
        print("Error: --token is required (or set SAS_TOKEN env var).")
        return 1

    client = AttentionClient(
        url=args.url,
        token=args.token,
        video=CameraConfig(device_index=cam_index if cam_index is not None else 0),
        audio=MicConfig(device=mic_device),
        initial_threshold=args.threshold,
        enable_audio=enable_audio,
        enable_video=enable_video,
    )

    ui = TerminalUI()
    warmup = {"count": 0}

    use_llm = bool(args.openai_key) and not args.no_llm
    llm: RealtimeLLMBridge | None = None
    llm_state = {"s": "idle"}

    if use_llm:
        llm = RealtimeLLMBridge(
            api_key=args.openai_key,
            instructions=LLM_INSTRUCTIONS,
        )

        def on_speaking_start():
            llm_state["s"] = "speaking"
            ui.update_llm_state("Speaking")
            ui.log("LLM speaking")
            client.mute()
            client.mark_responding(True)

        def on_speaking_end():
            llm_state["s"] = "idle"
            ui.update_llm_state("Idle")
            ui.log("LLM done")
            client.unmute()
            client.mark_responding(False)

        llm.on("speaking_start", on_speaking_start)
        llm.on("speaking_end", on_speaking_end)
        llm.on("transcript", lambda t: ui.log(f"LLM: {t[:60]}"))
        llm.on("error", lambda e: ui.log(f"LLM error: {e['title']}: {e['message']}"))
    else:
        ui.log("LLM disabled — set --openai-key or OPENAI_API_KEY to enable")

    @client.on_connected
    def _(): ui.log("ws connected")

    @client.on_started
    def _(): ui.log("server started")

    @client.on_warmup_complete
    def _():
        ui.activate()
        ui.log("warmup complete — streaming live")

    @client.on_prediction
    def _(event):
        if not ui.active:
            warmup["count"] += 1
            if warmup["count"] == 1 or warmup["count"] % 5 == 0:
                print(f"  warming up model... ({warmup['count']}/~50)")
            return
        ui.update_prediction(event.cls, event.confidence, event.num_faces)

    @client.on_state
    def _(event):
        ui.update_conv_state(event.state)
        ui.log(f"state -> {event.state}")

    @client.on_speech_ready
    def _(event):
        ui.log(f"speech ready ({event.duration_sec:.2f}s)")
        if llm is not None:
            llm_state["s"] = "processing"
            ui.update_llm_state("Processing")
            llm.send_audio_b64(event.audio_base64)

    @client.on_stats
    def _(event):
        rtt = f"{event.rtt_ms:.0f}ms" if event.rtt_ms is not None else "n/a"
        ui.log(f"stats rtt={rtt} v={event.sent_video}(-{event.skipped_video}) a={event.sent_audio}")

    @client.on_error
    def _(event):
        ui.log(f"ERROR {event.title}: {event.message}"
               + (f" | {event.detail}" if event.detail else ""))

    @client.on_disconnected
    def _(event):
        ui.log(f"disconnected code={event.code} reason={event.reason or 'none'}")

    print("starting... (Ctrl-C to stop)")
    try:
        client.start()
    except Exception as e:
        print(f"start failed: {e}", file=sys.stderr)
        if llm is not None:
            llm.close()
        return 1

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        ui.deactivate()
        print("\nstopping...")
        if llm is not None:
            llm.close()
        client.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
