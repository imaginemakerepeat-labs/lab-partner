#!/usr/bin/env python3
"""
Lab Partner — Raspberry Pi Gamepad Hold-to-Talk Assistant
"""

import os
import json
import tempfile
import subprocess
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
    "models": {
        "stt": "whisper-1",
        "chat": "gpt-4o"
    },
    "tts": {
        "engine": "espeak-ng",
        "voice": "en-us",
        "rate": 175
    },
    "logging": {
        "enabled": True,
        "path": "logs/conversation.jsonl",
        "max_turns_in_memory": 12
    }
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

    # Validate sample rate
    if not isinstance(config["sample_rate"], int) or config["sample_rate"] <= 0:
        raise ValueError("sample_rate must be a positive integer")

    # Map string key to evdev code
    key_name = config.get("ptt_key", "BTN_THUMB")
    if isinstance(key_name, str):
        if not hasattr(ecodes, key_name):
            raise ValueError(f"Unknown ptt_key '{key_name}'")
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

def record_audio():
    print("Recording...")
    recording = []

    def callback(indata, frames, time, status):
        recording.append(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback):
        while button_held:
            sd.sleep(50)

    audio = np.concatenate(recording, axis=0)
    return audio


def transcribe_audio(audio):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio, SAMPLE_RATE)
        tmp_path = tmp.name

    with open(tmp_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model=STT_MODEL,
            file=f
        )

    os.unlink(tmp_path)
    return transcript.text


def chat_with_model(user_text):
    global messages

    messages.append({"role": "user", "content": user_text})
    messages = messages[-MAX_TURNS:]

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages
    )

    reply = response.choices[0].message.content
    messages.append({"role": "assistant", "content": reply})

    return reply


def speak(text):
    subprocess.run(
        [TTS_ENGINE, "-v", TTS_VOICE, "-s", str(TTS_RATE), text],
        check=False
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

button_held = False

for event in device.read_loop():
    if event.type == ecodes.EV_KEY and event.code == PTT_KEY:
        if event.value == 1:  # Button pressed
            button_held = True
            audio_data = record_audio()

        elif event.value == 0:  # Button released
            button_held = False

            try:
                transcript = transcribe_audio(audio_data)
                print(f"\nYou: {transcript}")

                if LOG_ENABLED:
                    append_jsonl(LOG_PATH, {
                        "ts": datetime.now().isoformat(),
                        "role": "user",
                        "text": transcript
                    })

                reply = chat_with_model(transcript)
                print(f"Assistant: {reply}\n")

                if LOG_ENABLED:
                    append_jsonl(LOG_PATH, {
                        "ts": datetime.now().isoformat(),
                        "role": "assistant",
                        "text": reply
                    })

                speak(reply)

            except Exception as e:
                print(f"Error: {e}")