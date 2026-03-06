#!/usr/bin/env python3
"""
Lab Partner — Dual PTT + Interrupt + HUD + Persona/KB + Keyboard + Maynard Mouth (Debug)

Gamepad:
- Red   = OpenAI (hold to record)
- Green = Local/Ollama (hold to record)
- Yellow/Interrupt = BTN_PINKIE (press to interrupt TTS + cancel current turn)

Keyboard (toggle mode; type then Enter):
- o  -> toggle OpenAI record start/stop
- l  -> toggle Local/Ollama record start/stop
- i  -> interrupt
- h  -> help
- q  -> quit

HUD:
- Uses hud.py's run_hud(queue)

Persona/Knowledge:
- Loads persona.txt and knowledge_base.txt if present; injects into system prompt.

Maynard mouth:
- Sends 'open'/'wide'/'close'/'clear' over UDP to Maynard.
- DEBUG format: "seq|timestamp|cmd"
- Prints sender-side logs so we can pinpoint cut-outs.
"""

import sys
import json
import time
import socket
import tempfile
import subprocess
import threading
from pathlib import Path
from queue import Queue, Empty

import requests
import numpy as np
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv
from openai import OpenAI
from evdev import InputDevice, ecodes
from config import load_config, append_jsonl, read_text_file

# ----------------------------
# INIT
# ----------------------------

load_dotenv()
client = OpenAI()
cfg = load_config()

DEVICE_PATH = cfg["device_path"]
SAMPLE_RATE = int(cfg["sample_rate"])

STT_MODEL = cfg["models"]["stt"]
OPENAI_CHAT_MODEL_DEFAULT = cfg["models"]["chat"]

TTS_ENGINE = cfg["tts"]["engine"]
TTS_VOICE = cfg["tts"]["voice"]
TTS_RATE = str(cfg["tts"]["rate"])

LOG_ENABLED = bool(cfg["logging"]["enabled"])
LOG_PATH = cfg["logging"]["path"]
MAX_TURNS = int(cfg["logging"]["max_turns_in_memory"])

LOCAL_TURNS = int(cfg["context"]["local_turns"])

RED_KEY = cfg["red_key"]
GREEN_KEY = cfg["green_key"]
INTERRUPT_KEY = cfg["interrupt_key"]

ROUTES = cfg["routes"]
OLLAMA_URL = cfg.get("ollama", {}).get("url", "http://127.0.0.1:11434/api/chat")

MAYNARD_ENABLED = bool(cfg.get("maynard", {}).get("enabled", True))
MAYNARD_IP = cfg.get("maynard", {}).get("ip", "10.0.0.4")
MAYNARD_PORT = int(cfg.get("maynard", {}).get("port", 9000))

PROMPT_PERSONA = cfg.get("prompt_files", {}).get("persona", "persona.txt")
PROMPT_KB = cfg.get("prompt_files", {}).get("knowledge_base", "knowledge_base.txt")


# ----------------------------
# HUD (your hud.py uses run_hud(queue))
# ----------------------------

hud_queue = None
hud_thread = None
hud_mod = None

def hud_put(payload: dict) -> None:
    global hud_queue
    if hud_queue is None:
        return
    try:
        hud_queue.put_nowait(payload)
    except Exception:
        pass

try:
    import hud as hud_mod
    hud_queue = Queue()
    hud_thread = threading.Thread(target=hud_mod.run_hud, args=(hud_queue,), daemon=True)
    hud_thread.start()
    print("HUD started", flush=True)
except Exception as e:
    print(f"HUD not started: {e}", flush=True)
    hud_queue = None
    hud_mod = None

def hud_state(state: str, status: str = "", backend: str = "", flash: bool = False, memory: int = 0) -> None:
    payload = {"state": state, "status": status, "flash": flash, "memory": memory}
    if backend:
        payload["backend"] = backend
    hud_put(payload)


# ----------------------------
# PERSONA / KB SYSTEM PROMPT
# ----------------------------

persona_txt = read_text_file(PROMPT_PERSONA)
kb_txt = read_text_file(PROMPT_KB)

if persona_txt:
    print(f"Loaded {PROMPT_PERSONA}: {len(persona_txt)} chars", flush=True)
if kb_txt:
    print(f"Loaded {PROMPT_KB}: {len(kb_txt)} chars", flush=True)

system_prompt = "\n\n".join([t for t in [persona_txt, kb_txt] if t]).strip()


# ----------------------------
# STATE
# ----------------------------

messages = []
if system_prompt:
    messages.append({"role": "system", "content": system_prompt})

cancel_turn = threading.Event()

tts_proc = None
tts_lock = threading.Lock()

is_recording = False
active_route_key = None   # "red" or "green"
stop_event = None
rec_thread = None
audio_holder = {"audio": np.array([], dtype=np.float32)}

mouth_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def trim_messages():
    if not messages:
        return
    sys_msg = messages[0] if messages[0]["role"] == "system" else None
    rest = messages[1:] if sys_msg else messages
    rest = rest[-(MAX_TURNS * 2):]
    messages.clear()
    if sys_msg:
        messages.append(sys_msg)
    messages.extend(rest)


# ----------------------------
# MAYNARD MOUTH DEBUG
# ----------------------------

MOUTH_DEBUG = True
MOUTH_DEBUG_EVERY = 1   # print every N sends (1=all; set 10 to reduce spam)
mouth_seq = 0
mouth_sent = 0

def maynard_send(cmd: str, why: str = "") -> None:
    """Send UDP mouth command with seq/timestamp (for debug)."""
    global mouth_seq, mouth_sent
    if not MAYNARD_ENABLED:
        return

    mouth_seq += 1
    mouth_sent += 1
    payload = f"{mouth_seq}|{time.time():.3f}|{cmd}"

    try:
        mouth_sock.sendto(payload.encode("utf-8"), (MAYNARD_IP, MAYNARD_PORT))
        if MOUTH_DEBUG and (mouth_sent % MOUTH_DEBUG_EVERY == 0):
            print(f"[MOUTH->] seq={mouth_seq} cmd={cmd} why={why}", flush=True)
    except Exception as e:
        print(f"[MOUTH!!] send failed seq={mouth_seq} cmd={cmd} err={e}", flush=True)


def _char_to_viseme(ch: str):
    c = ch.lower()
    if c in "mbp":
        return "close"
    if c in "ei":
        return "wide"
    if c in "aou":
        return "open"
    if c == "y":
        return "wide"
    return None


def mouth_ticker_loop(stop_evt: threading.Event) -> None:
    """
    Loop animation while speaking:
    Keeps sending packets until stop_evt or cancel_turn is set.
    """
    print("[MOUTH] loop start", flush=True)
    cycle = ["open", "wide", "open", "close"]
    idx = 0

    try:
        while not stop_evt.is_set() and not cancel_turn.is_set():
            cmd = cycle[idx % len(cycle)]
            maynard_send(cmd, why="loop")
            idx += 1
            time.sleep(0.08)  # ~12.5 fps; try 0.06 for snappier
    except Exception as e:
        print(f"[MOUTH!!] loop exception: {e}", flush=True)

    maynard_send("close", why="loop_end")
    maynard_send("clear", why="loop_end")
    print("[MOUTH] loop end", flush=True)

# ----------------------------
# AUDIO
# ----------------------------

def record_audio(stop_evt: threading.Event) -> np.ndarray:
    frames = []

    def callback(indata, frame_count, time_info, status):
        frames.append(indata.copy())
        if stop_evt.is_set():
            raise sd.CallbackStop()

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=callback):
            while not stop_evt.is_set():
                sd.sleep(50)
    except Exception:
        return np.array([], dtype=np.float32)

    if not frames:
        return np.array([], dtype=np.float32)

    return np.concatenate(frames, axis=0).flatten().astype(np.float32)


def transcribe(audio: np.ndarray) -> str:
    if audio is None or getattr(audio, "size", 0) == 0:
        return ""

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio, SAMPLE_RATE)
        wav_path = f.name

    try:
        with open(wav_path, "rb") as f:
            resp = client.audio.transcriptions.create(model=STT_MODEL, file=f)
        return (resp.text or "").strip()
    finally:
        try:
            Path(wav_path).unlink()
        except Exception:
            pass


# ----------------------------
# CHAT
# ----------------------------

def chat_openai(user_text: str, model: str) -> str:
    cancel_turn.clear()
    messages.append({"role": "user", "content": user_text})
    trim_messages()

    resp = client.chat.completions.create(model=model, messages=messages)

    if cancel_turn.is_set():
        return ""

    out = (resp.choices[0].message.content or "").strip()
    messages.append({"role": "assistant", "content": out})
    trim_messages()
    return out


def chat_ollama(user_text: str, model: str) -> str:
    cancel_turn.clear()

    sys_msg = messages[0] if (messages and messages[0]["role"] == "system") else None
    recent = [m for m in messages if m["role"] != "system"][-(LOCAL_TURNS * 2):]
    local_msgs = ([sys_msg] if sys_msg else []) + recent + [{"role": "user", "content": user_text}]

    payload = {"model": model, "messages": local_msgs, "stream": False}

    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=180)
        r.raise_for_status()
        data = r.json()
        out = (data.get("message", {}) or {}).get("content", "")
    except Exception as e:
        out = f"(ollama error) {e}"

    if cancel_turn.is_set():
        return ""

    messages.append({"role": "user", "content": user_text})
    messages.append({"role": "assistant", "content": (out or "").strip()})
    trim_messages()

    return (out or "").strip()


def generate(route_key: str, user_text: str) -> tuple[str, str]:
    route = ROUTES[route_key]
    backend = route["backend"]

    if backend == "openai":
        model = route.get("chat_model", OPENAI_CHAT_MODEL_DEFAULT)
        return chat_openai(user_text, model=model), "OPENAI"

    if backend == "ollama":
        model = route["ollama_model"]
        return chat_ollama(user_text, model=model), "LOCAL"

    raise ValueError(f"Unknown backend: {backend}")


# ----------------------------
# SPEAK + INTERRUPT
# ----------------------------

def speak(text: str, backend_label: str):
    global tts_proc
    if not text:
        return

    with tts_lock:
        if tts_proc and tts_proc.poll() is None:
            try:
                tts_proc.terminate()
            except Exception:
                pass

        mouth_stop = threading.Event()
        threading.Thread(target=mouth_ticker_loop, args=(mouth_stop,), daemon=True).start()

        maynard_send("open", why="tts_start")

        if hud_mod:
            hud_state(
                state=getattr(hud_mod, "STATE_SPEAKING", "speaking"),
                status="Speaking...",
                backend=backend_label,
                memory=len(messages),
            )

        tts_proc = subprocess.Popen(
            [TTS_ENGINE, "-v", TTS_VOICE, "-s", TTS_RATE, text],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _cleanup(proc, stop_evt):
        try:
            proc.wait()
        finally:
            stop_evt.set()
            maynard_send("close", why="tts_end")
            maynard_send("clear", why="tts_end")
            if hud_mod:
                hud_state(
                    state=getattr(hud_mod, "STATE_IDLE", "idle"),
                    status="Idle",
                    backend=backend_label,
                    memory=len(messages),
                )

    threading.Thread(target=_cleanup, args=(tts_proc, mouth_stop), daemon=True).start()


def interrupt():
    global tts_proc
    cancel_turn.set()

    with tts_lock:
        if tts_proc and tts_proc.poll() is None:
            try:
                tts_proc.terminate()
            except Exception:
                pass

    maynard_send("close", why="interrupt")
    maynard_send("clear", why="interrupt")

    print("interrupted", flush=True)

    if hud_mod:
        hud_state(
            state=getattr(hud_mod, "STATE_INTERRUPTED", "interrupted"),
            status="Interrupted",
            flash=True,
            memory=len(messages),
        )


# ----------------------------
# RECORD CONTROL
# ----------------------------

def start_record(route_key: str):
    global is_recording, active_route_key, stop_event, rec_thread, audio_holder

    cancel_turn.clear()
    is_recording = True
    active_route_key = route_key
    stop_event = threading.Event()
    audio_holder = {"audio": np.array([], dtype=np.float32)}

    label = "openai" if route_key == "red" else "local"
    print(f"recording ({label})...", flush=True)

    if hud_mod:
        backend_label = getattr(hud_mod, "BACKEND_OPENAI", "OPENAI") if route_key == "red" else getattr(hud_mod, "BACKEND_LOCAL", "LOCAL")
        hud_state(
            state=getattr(hud_mod, "STATE_RECORDING", "recording"),
            status=f"Recording ({backend_label})...",
            backend=backend_label,
            memory=len(messages),
        )

    def runner():
        audio_holder["audio"] = record_audio(stop_event)

    rec_thread = threading.Thread(target=runner, daemon=True)
    rec_thread.start()


def stop_record_and_handle():
    global is_recording, stop_event, rec_thread, audio_holder, active_route_key

    is_recording = False
    if stop_event:
        stop_event.set()
    if rec_thread:
        rec_thread.join(timeout=2.0)

    if hud_mod:
        hud_state(
            state=getattr(hud_mod, "STATE_THINKING", "thinking"),
            status="Transcribing...",
            memory=len(messages),
        )

    audio = (audio_holder or {}).get("audio", np.array([], dtype=np.float32))
    text = transcribe(audio)

    if not text:
        print("(no audio captured)", flush=True)
        if hud_mod:
            hud_state(
                state=getattr(hud_mod, "STATE_IDLE", "idle"),
                status="Idle",
                memory=len(messages),
            )
        return

    print(f"\nYou: {text}", flush=True)

    if LOG_ENABLED:
        append_jsonl(LOG_PATH, {"ts": time.time(), "role": "user", "text": text, "route": active_route_key})

    if hud_mod:
        hud_state(
            state=getattr(hud_mod, "STATE_THINKING", "thinking"),
            status="Thinking...",
            memory=len(messages),
        )

    reply, backend_label = generate(active_route_key, text)

    if cancel_turn.is_set():
        print("(cancelled — ignoring reply)", flush=True)
        if hud_mod:
            hud_state(
                state=getattr(hud_mod, "STATE_IDLE", "idle"),
                status="Idle",
                memory=len(messages),
            )
        return

    print(f"Assistant ({backend_label}): {reply}\n", flush=True)

    if LOG_ENABLED:
        append_jsonl(LOG_PATH, {"ts": time.time(), "role": "assistant", "text": reply, "backend": backend_label})

    speak(reply, backend_label)


# ----------------------------
# KEYBOARD THREAD
# ----------------------------

kbd_q: Queue[str] = Queue()
quit_flag = threading.Event()

def keyboard_thread():
    while not quit_flag.is_set():
        try:
            line = sys.stdin.readline()
            if not line:
                time.sleep(0.05)
                continue
            cmd = line.strip().lower()
            if cmd:
                kbd_q.put(cmd)
        except Exception:
            time.sleep(0.1)

def print_help():
    print("\nKeyboard controls (type + Enter):")
    print("  o  -> toggle OpenAI record start/stop")
    print("  l  -> toggle Local/Ollama record start/stop")
    print("  i  -> interrupt")
    print("  q  -> quit")
    print("  h  -> help\n", flush=True)


# ----------------------------
# MAIN
# ----------------------------

print("==============================================")
print(" Lab Partner — Dual PTT + Interrupt + HUD + Persona/KB + Keyboard + Maynard (Debug)")
print("==============================================")
print("Gamepad: Red=OpenAI | Green=Local | Yellow=Interrupt")
print("Keyboard: type then Enter (h for help)")
print("Ctrl+C to quit.\n")

if hud_mod:
    hud_state(
        state=getattr(hud_mod, "STATE_IDLE", "idle"),
        status="Idle",
        memory=len(messages),
    )

threading.Thread(target=keyboard_thread, daemon=True).start()

device = InputDevice(DEVICE_PATH)

try:
    import select
except Exception:
    select = None

try:
    while True:
        # Keyboard commands
        try:
            cmd = kbd_q.get_nowait()
        except Empty:
            cmd = None

        if cmd:
            if cmd in ("h", "help", "?"):
                print_help()
            elif cmd in ("q", "quit", "exit"):
                print("Quitting...", flush=True)
                quit_flag.set()
                break
            elif cmd in ("i", "interrupt"):
                interrupt()
            elif cmd in ("o", "openai"):
                if not is_recording:
                    start_record("red")
                else:
                    stop_record_and_handle()
            elif cmd in ("l", "local", "ollama"):
                if not is_recording:
                    start_record("green")
                else:
                    stop_record_and_handle()
            else:
                print(f"(unknown cmd) {cmd} — type 'h' for help", flush=True)

        # Gamepad events without blocking keyboard responsiveness
        if select is None:
            time.sleep(0.01)
            continue

        rlist, _, _ = select.select([device.fd], [], [], 0.01)
        if not rlist:
            continue

        for event in device.read():
            if event.type != ecodes.EV_KEY:
                continue

            if event.code == INTERRUPT_KEY and event.value == 1:
                interrupt()
                continue

            if event.code in (RED_KEY, GREEN_KEY):
                if event.value == 1 and not is_recording:
                    start_record("red" if event.code == RED_KEY else "green")
                elif event.value == 0 and is_recording:
                    stop_record_and_handle()

except KeyboardInterrupt:
    print("\nExiting...", flush=True)
finally:
    quit_flag.set()
    try:
        device.close()
    except Exception:
        pass
    try:
        maynard_send("close", why="exit")
        maynard_send("clear", why="exit")
    except Exception:
        pass
    if hud_queue is not None:
        try:
            hud_queue.put({"cmd": "quit"})
        except Exception:
            pass