#!/usr/bin/env python3
"""
Lab Partner — Raspberry Pi Gamepad Hold-to-Talk Assistant
"""

import os
import json
import tempfile
import subprocess
import threading
from pathlib import Path
from datetime import datetime

import numpy as np
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv
from openai import OpenAI
from evdev import InputDevice, ecodes

# ----------------------------
# DEFAULT CONFIG
# ----------------------------

DEFAULT_CONFIG = {
    "device_path": "/dev/input/by-id/usb-0079_USB_Gamepad-event-joystick",
    "ptt_key": "BTN_THUMB",
    "sample_rate": 16000,
    "models": {"stt": "whisper-1", "chat": "gpt-4o"},
    "tts": {"engine": "espeak-ng", "voice": "en-us", "rate": 175},
    "logging": {"enabled": True, "path": "logs/conversation.jsonl", "max_turns_in_memory": 12},
}

# ----------------------------
# CONFIG LOADER
# ----------------------------


def _deep_merge(base, override):
    result = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(path="config.json"):
    config = dict(DEFAULT_CONFIG)
    p = Path(path)

    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            user_config = json.load(f)
        config = _deep_merge(config, user_config)

    if not isinstance(config["sample_rate"], int) or config["sample_rate"] <= 0:
        raise ValueError("sample_rate must be a positive integer")

    key_name = config.get("ptt_key", "BTN_THUMB")
    if isinstance(key_name, str):
        if not hasattr(ecodes, key_name):
            raise ValueError(f"Unknown ptt_key '{key_name}'. Try something like 'BTN_THUMB'.")
        config["ptt_key_code"] = getattr(ecodes, key_name)
    else:
        config["ptt_key_code"] = int(key_name)

    return config


def append_jsonl(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ----------------------------
# INITIALIZE
# ----------------------------

load_dotenv()
client = OpenAI()

cfg = load_config()

DEVICE_PATH = cfg["device_path"]
PTT_KEY = cfg["ptt_key_code"]
SAMPLE_RATE = cfg["sample_rate"]

STT_MODEL = cfg["models"]["stt"]
CHAT_MODEL = cfg["models"]["chat"]

TTS_ENGINE = cfg["tts"]["engine"]
TTS_VOICE = cfg["tts"]["voice"]
TTS_RATE = cfg["tts"]["rate"]

LOG_ENABLED = cfg["logging"]["enabled"]
LOG_PATH = cfg["logging"]["path"]
MAX_TURNS = cfg["logging"]["max_turns_in_memory"]

messages = []


# ----------------------------
# AUDIO FUNCTIONS
# ----------------------------


def record_audio_until(stop_event: threading.Event) -> np.ndarray:
    """Record mic audio until stop_event is set. Returns float32 mono array shape (N, 1)."""
    print("Recording...")
    chunks: list[np.ndarray] = []

    def callback(indata, frames, time_info, status):
        if status:
            # xruns etc. Not fatal.
            print(status)
        chunks.append(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback):
        while not stop_event.is_set():
            sd.sleep(50)

    if not chunks:
        return np.zeros((0, 1), dtype=np.float32)

    return np.concatenate(chunks, axis=0)


def transcribe_audio(audio: np.ndarray) -> str:
    if audio is None or audio.size == 0:
        return ""

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio, SAMPLE_RATE)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model=STT_MODEL,
                file=f,
            )
        return transcript.text.strip()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def chat_with_model(user_text: str) -> str:
    global messages

    messages.append({"role": "user", "content": user_text})
    messages = messages[-MAX_TURNS:]

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
    )

    reply = response.choices[0].message.content.strip()
    messages.append({"role": "assistant", "content": reply})
    return reply


def speak(text: str) -> None:
    if not text:
        return
    subprocess.run(
        [TTS_ENGINE, "-v", TTS_VOICE, "-s", str(TTS_RATE), text],
        check=False,
    )


# ----------------------------
# MAIN LOOP
# ----------------------------

print("==============================================")
print(" Lab Partner — Raspberry Pi Voice Assistant ")
print("==============================================")
print(" Hold button to speak. Release to send.")
print(" Ctrl+C to quit.")
print()

device = InputDevice(DEVICE_PATH)

# Recording state
is_recording = False
stop_event: threading.Event | None = None
recording_thread: threading.Thread | None = None
audio_data: np.ndarray | None = None


def _start_recording():
    global stop_event, recording_thread, audio_data, is_recording

    if is_recording:
        return

    is_recording = True
    stop_event = threading.Event()
    audio_data = None

    def runner():
        global audio_data
        audio_data = record_audio_until(stop_event)

    recording_thread = threading.Thread(target=runner, daemon=True)
    recording_thread.start()


def _stop_recording_and_process():
    global stop_event, recording_thread, is_recording, audio_data

    if not is_recording:
        return

    is_recording = False
    if stop_event is not None:
        stop_event.set()

    if recording_thread is not None:
        recording_thread.join(timeout=2.0)

    try:
        transcript = transcribe_audio(audio_data)

        if not transcript:
            print("\n(heard nothing — try again)\n")
            return

        print(f"\nYou: {transcript}")

        if LOG_ENABLED:
            append_jsonl(
                LOG_PATH,
                {"ts": datetime.now().isoformat(), "role": "user", "text": transcript},
            )

        reply = chat_with_model(transcript)
        print(f"Assistant: {reply}\n")

        if LOG_ENABLED:
            append_jsonl(
                LOG_PATH,
                {"ts": datetime.now().isoformat(), "role": "assistant", "text": reply},
            )

        speak(reply)

    except Exception as e:
        print(f"Error: {e}")


try:
    for event in device.read_loop():
        # Only pay attention to our configured PTT key
        if event.type != ecodes.EV_KEY or event.code != PTT_KEY:
            continue

        # Press
        if event.value == 1:
            _start_recording()

        # Release
        elif event.value == 0:
            _stop_recording_and_process()

except KeyboardInterrupt:
    print("\nExiting...")