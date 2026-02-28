#!/usr/bin/env python3
"""
Lab Partner — Dual PTT + Interrupt
Red   = OpenAI (PTT hold)
Green = Ollama (PTT hold)
Yellow/Interrupt = BTN_PINKIE (press to interrupt speech + cancel current turn)

Features:
- OpenAI tool: get_time_date()
- Shared SYSTEM_PERSONA for OpenAI + Ollama
"""

import os
import json
import tempfile
import subprocess
import threading
from pathlib import Path
from datetime import datetime

import requests
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
    },

    "context": {
        "local_turns": 6
    },

    "buttons": {
        "red_ptt": "BTN_THUMB",
        "green_ptt": "BTN_TOP",
        "interrupt": "BTN_PINKIE"
    },

    "routes": {
        "red": {
            "backend": "openai",
            "chat_model": "gpt-4o"
        },
        "green": {
            "backend": "ollama",
            "ollama_model": "qwen2.5:1.5b-instruct"
        }
    }
}


# ----------------------------
# SYSTEM PERSONA
# ----------------------------

SYSTEM_PERSONA = (
    "You are Lab Partner, a concise workshop assistant. "
    "You help with electronics, fabrication, coding, and practical problem solving. "
    "Be direct. Be clear. Avoid unnecessary verbosity."
    "Your name is Query."
)


# ----------------------------
# CONFIG HELPERS
# ----------------------------

def _deep_merge(base, override):
    result = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _ecode(name: str) -> int:
    if not hasattr(ecodes, name):
        raise ValueError(f"Unknown button '{name}'")
    return getattr(ecodes, name)


def load_config(path="config.json"):
    config = dict(DEFAULT_CONFIG)
    p = Path(path)

    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            config = _deep_merge(config, json.load(f))

    buttons = config["buttons"]
    config["red_key"] = _ecode(buttons["red_ptt"])
    config["green_key"] = _ecode(buttons["green_ptt"])
    config["interrupt_key"] = _ecode(buttons["interrupt"])

    return config


# ----------------------------
# INIT
# ----------------------------

load_dotenv()
client = OpenAI()

cfg = load_config()

DEVICE_PATH = cfg["device_path"]
SAMPLE_RATE = cfg["sample_rate"]
LOCAL_TURNS = cfg["context"]["local_turns"]

RED_KEY = cfg["red_key"]
GREEN_KEY = cfg["green_key"]
INTERRUPT_KEY = cfg["interrupt_key"]

ROUTES = cfg["routes"]

MAX_TURNS = cfg["logging"]["max_turns_in_memory"]

TTS_ENGINE = cfg["tts"]["engine"]
TTS_VOICE = cfg["tts"]["voice"]
TTS_RATE = cfg["tts"]["rate"]

messages = []

cancel_turn = threading.Event()
tts_proc = None
tts_lock = threading.Lock()

record_thread = None
audio_buffer = None


# ----------------------------
# LOCAL TOOL (OpenAI)
# ----------------------------

def tool_get_time_date() -> str:
    now = datetime.now()
    return now.strftime("%A, %B %d, %Y — %I:%M %p")


OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_time_date",
            "description": "Get the current local date and time.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False
            }
        }
    }
]

TOOL_DISPATCH = {
    "get_time_date": lambda _args: tool_get_time_date()
}


# ----------------------------
# AUDIO
# ----------------------------

def record_audio(stop_event: threading.Event):
    chunks = []

    def callback(indata, frames, time_info, status):
        chunks.append(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback):
        while not stop_event.is_set():
            sd.sleep(50)

    if not chunks:
        return np.zeros((0, 1), dtype=np.float32)

    return np.concatenate(chunks, axis=0)


def transcribe(audio) -> str:
    if audio is None or getattr(audio, "size", 0) == 0:
        return ""

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio, SAMPLE_RATE)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as f:
            resp = client.audio.transcriptions.create(
                model=cfg["models"]["stt"],
                file=f
            )
        return (resp.text or "").strip()
    finally:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass


# ----------------------------
# CHAT
# ----------------------------

def _trim():
    global messages
    if MAX_TURNS and len(messages) > MAX_TURNS:
        messages = messages[-MAX_TURNS:]


def chat_openai(text: str, model: str) -> str:
    # Inject system persona once
    if not any(m["role"] == "system" for m in messages):
        messages.append({"role": "system", "content": SYSTEM_PERSONA})

    messages.append({"role": "user", "content": text})
    _trim()

    for _ in range(4):
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=OPENAI_TOOLS,
            tool_choice="auto",
        )

        msg = resp.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None)

        if tool_calls:
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        },
                    }
                    for tc in tool_calls
                ]
            })
            _trim()

            for tc in tool_calls:
                name = tc.function.name
                args = {}
                handler = TOOL_DISPATCH.get(name)

                if handler:
                    try:
                        tool_out = handler(args)
                    except Exception as e:
                        tool_out = f"Tool error: {e}"
                else:
                    tool_out = f"Unknown tool: {name}"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(tool_out),
                })
                _trim()

            continue

        reply = (msg.content or "").strip()
        messages.append({"role": "assistant", "content": reply})
        _trim()
        return reply

    return "Sorry — I got stuck."


def _local_prompt() -> str:
    lines = [SYSTEM_PERSONA + "\n"]
    for m in messages[-LOCAL_TURNS:]:
        if m["role"] == "tool":
            continue
        role = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{role}: {m.get('content','')}")
    lines.append("Assistant:")
    return "\n".join(lines)


def chat_ollama(text: str, model: str) -> str:
    messages.append({"role": "user", "content": text})
    _trim()

    prompt = _local_prompt()
    print("Thinking (local)...", flush=True)

    r = requests.post(
        "http://127.0.0.1:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=180,
    )
    r.raise_for_status()

    reply = (r.json().get("response") or "").strip()
    messages.append({"role": "assistant", "content": reply})
    _trim()
    return reply


def generate(text: str, route: dict):
    backend = route["backend"]
    if backend == "openai":
        return chat_openai(text, route["chat_model"]), "openai"
    if backend == "ollama":
        return chat_ollama(text, route["ollama_model"]), "ollama"
    raise ValueError(f"Unknown backend: {backend}")


# ----------------------------
# SPEECH
# ----------------------------

def speak(text: str):
    global tts_proc
    if not text:
        return

    with tts_lock:
        if tts_proc and tts_proc.poll() is None:
            try:
                tts_proc.terminate()
            except Exception:
                pass

        tts_proc = subprocess.Popen(
            [TTS_ENGINE, "-v", TTS_VOICE, "-s", str(TTS_RATE), text],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )


def interrupt():
    global tts_proc
    cancel_turn.set()

    with tts_lock:
        if tts_proc and tts_proc.poll() is None:
            try:
                tts_proc.terminate()
            except Exception:
                pass

    print("interrupted", flush=True)


# ----------------------------
# MAIN LOOP
# ----------------------------

print("==============================================")
print(" Lab Partner — Dual PTT + Interrupt")
print("==============================================")
print("Red = OpenAI | Green = Local | Yellow = Interrupt")
print("Ctrl+C to quit.\n")

device = InputDevice(DEVICE_PATH)

is_recording = False
stop_event = None
active_route = None

try:
    for event in device.read_loop():

        if event.type != ecodes.EV_KEY:
            continue

        if event.code == INTERRUPT_KEY and event.value == 1:
            interrupt()
            continue

        if event.code in (RED_KEY, GREEN_KEY):

            if event.value == 1:
                cancel_turn.clear()
                is_recording = True
                stop_event = threading.Event()

                is_red = (event.code == RED_KEY)
                active_route = ROUTES["red"] if is_red else ROUTES["green"]

                label = "openai" if is_red else "local"
                print(f"recording ({label})...", flush=True)

                def runner():
                    global audio_buffer
                    audio_buffer = record_audio(stop_event)

                record_thread = threading.Thread(target=runner, daemon=True)
                record_thread.start()

            elif event.value == 0 and is_recording:
                is_recording = False

                if stop_event:
                    stop_event.set()

                if record_thread:
                    record_thread.join(timeout=2.0)

                text = transcribe(audio_buffer)
                if not text:
                    continue

                print("\nYou:", text, flush=True)

                reply, backend = generate(text, active_route)

                if cancel_turn.is_set():
                    print("(cancelled turn — ignoring reply)", flush=True)
                    continue

                print(f"Assistant ({backend}): {reply}\n", flush=True)
                speak(reply)

except KeyboardInterrupt:
    print("\nExiting...", flush=True)