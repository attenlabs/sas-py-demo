#!/usr/bin/env python3
"""CLI demo for attenlabs-sas.

Streams mic + webcam to the SD Attention Server, forwards detected speech to
OpenAI Realtime, and plays the response back through the local speaker.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import sounddevice as sd

from sas import AttentionClient, CameraConfig, MicConfig

from llm import RealtimeLLMBridge

CLASS_LABELS = {0: "silent", 1: "human", 2: "device"}

LLM_INSTRUCTIONS = (
    "You are a helpful assistant. Respond concisely in 1 sentence. "
    "If a device/TV command is spoken to you, respond as if you were controlling a TV."
)


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
            print("[llm] speaking")
            client.mute()
            client.mark_responding(True)

        def on_speaking_end():
            llm_state["s"] = "idle"
            print("[llm] done")
            client.unmute()
            client.mark_responding(False)

        llm.on("speaking_start", on_speaking_start)
        llm.on("speaking_end", on_speaking_end)
        llm.on("transcript", lambda t: print(f"[llm] transcript: {t}"))
        llm.on("error", lambda e: print(f"[llm error] {e['title']}: {e['message']}"))
    else:
        print("llm disabled — set --openai-key or OPENAI_API_KEY env var to enable")

    @client.on_connected
    def _(): print("ws connected")

    @client.on_started
    def _(): print("server warmup complete")

    @client.on_warmup_complete
    def _(): print("first prediction received — streaming live")

    @client.on_prediction
    def _(event):
        if llm_state["s"] in ("speaking", "processing"):
            return
        label = CLASS_LABELS.get(event.cls, f"cls{event.cls}")
        pct = int(event.confidence * 100)
        print(f"prediction: {label:7}  {pct:3}%  faces={event.num_faces}  src={event.source}")

    @client.on_vad
    def _(event):
        if event.probability > 0.5:
            print(f"vad: {event.probability:.2f}")

    @client.on_state
    def _(event):
        print(f"conv state: {event.state}")

    @client.on_speech_ready
    def _(event):
        print(f"speech ready ({event.duration_sec:.2f}s)")
        if llm is not None:
            llm_state["s"] = "processing"
            llm.send_audio_b64(event.audio_base64)

    @client.on_stats
    def _(event):
        rtt = f"{event.rtt_ms:.0f}ms" if event.rtt_ms is not None else "n/a"
        print(f"stats: rtt={rtt} video={event.sent_video}(skip {event.skipped_video}) audio={event.sent_audio}")

    @client.on_error
    def _(event):
        print(f"ERROR: {event.title}: {event.message}"
              + (f" | {event.detail}" if event.detail else ""))

    @client.on_disconnected
    def _(event):
        print(f"disconnected: code={event.code} reason={event.reason or 'none'}")

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
        print("\nstopping...")
    finally:
        if llm is not None:
            llm.close()
        client.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
