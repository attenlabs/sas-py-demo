#!/usr/bin/env python3
"""CLI demo for sas-py.

Streams mic + webcam to the SD Attention Server, forwards detected speech to
OpenAI Realtime, and plays the response back through the local speaker.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

from sas import AttentionClient, CameraConfig, MicConfig

from llm import RealtimeLLMBridge

CLASS_LABELS = {0: "silent", 1: "human", 2: "device"}

LLM_INSTRUCTIONS = (
    "You are a helpful assistant. Respond concisely in 1 sentence. "
    "If a device/TV command is spoken to you, respond as if you were controlling a TV."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="sas-py CLI demo")
    p.add_argument("--token", required=True, help="SAS auth token")
    p.add_argument("--url", default=None,
                   help="Override the SAS server URL (default: wss://server.attentionlabs.ai/ws)")
    p.add_argument("--openai-key", default=os.environ.get("OPENAI_API_KEY"),
                   help="OpenAI API key with Realtime access (env: OPENAI_API_KEY)")
    p.add_argument("--camera-index", type=int, default=0, help="Webcam device index")
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

    mic_device = args.mic_device
    if mic_device is not None:
        try:
            mic_device = int(mic_device)
        except ValueError:
            pass  # treat as device name string

    client = AttentionClient(
        url=args.url,
        token=args.token,
        video=CameraConfig(device_index=args.camera_index),
        audio=MicConfig(device=mic_device),
        initial_threshold=args.threshold,
        enable_audio=not args.no_audio,
        enable_video=not args.no_video,
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
