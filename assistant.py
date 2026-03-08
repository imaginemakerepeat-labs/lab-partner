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
import time
import threading
from queue import Queue, Empty

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from evdev import InputDevice, ecodes

from config import load_config, append_jsonl, read_text_file
from audio import record_audio, transcribe
from memory import trim_messages
from mouth import MouthController
from tts import TTSController

from backends.openai_backend import chat_openai
from backends.ollama_backend import chat_ollama
from skills import run_skill
from wake_word import WakeWordListener


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

tts = TTSController(TTS_ENGINE, TTS_VOICE, TTS_RATE)

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

WAKE_CFG = cfg.get("wake_word", {})
WAKE_ENABLED = bool(WAKE_CFG.get("enabled", False))
WAKE_MODEL = WAKE_CFG.get("model_path", "")
WAKE_THRESHOLD = float(WAKE_CFG.get("threshold", 0.6))
WAKE_COOLDOWN = float(WAKE_CFG.get("cooldown", 2.0))
WAKE_FOLLOWUP_SECONDS = float(WAKE_CFG.get("followup_seconds", 4))


# ----------------------------
# HUD
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
quit_flag = threading.Event()

tts_proc = None
tts_lock = threading.Lock()

is_recording = False
active_route_key = None  # "red" or "green"
stop_event = None
rec_thread = None
audio_holder = {"audio": np.array([], dtype=np.float32)}

wake_busy = threading.Event()
tts_active = threading.Event()

mouth = MouthController(MAYNARD_ENABLED, MAYNARD_IP, MAYNARD_PORT)

kbd_q: Queue[str] = Queue()

device = None
wake_listener = None


# ----------------------------
# CHAT
# ----------------------------

def generate(route_key: str, user_text: str) -> tuple[str, str]:
    global messages

    route = ROUTES[route_key]
    backend = route["backend"]

    if backend == "openai":
        model = route.get("chat_model", OPENAI_CHAT_MODEL_DEFAULT)
        reply = chat_openai(client, messages, cancel_turn, user_text, model)
        messages[:] = trim_messages(messages, MAX_TURNS)
        return reply, "OPENAI"

    if backend == "ollama":
        model = route["ollama_model"]
        reply = chat_ollama(messages, cancel_turn, user_text, model, OLLAMA_URL, LOCAL_TURNS)
        messages[:] = trim_messages(messages, MAX_TURNS)
        return reply, "LOCAL"

    raise ValueError(f"Unknown backend: {backend}")


def answer_from_web_context(user_text: str, payload: dict) -> tuple[str, str]:
    results = payload.get("results", [])
    page = payload.get("page")

    evidence = []

    for i, r in enumerate(results, start=1):
        evidence.append(
            f"[Result {i}]\n"
            f"Title: {r.get('title', '')}\n"
            f"URL: {r.get('url', '')}\n"
            f"Snippet: {r.get('snippet', '')}"
        )

    if page:
        evidence.append(
            f"[Top Page]\n"
            f"Title: {page.get('title', '')}\n"
            f"URL: {page.get('url', '')}\n"
            f"Text:\n{page.get('text', '')}"
        )

    evidence_text = "\n\n".join(evidence)

    temp_messages = [
        {
            "role": "system",
            "content": (
                "Answer the user's question using only the web evidence below. "
                "Do not invent facts. If evidence is weak or incomplete, say so. "
                "Be concise and helpful."
            ),
        },
        {
            "role": "user",
            "content": f"Question: {user_text}\n\nEvidence:\n{evidence_text}",
        },
    ]

    reply = chat_openai(
        client,
        temp_messages,
        cancel_turn,
        user_text="",
        model=OPENAI_CHAT_MODEL_DEFAULT,
    )

    return reply, "WEB+OPENAI"


# ----------------------------
# SPEAK + INTERRUPT
# ----------------------------

def speak(text: str, backend_label: str):
    global tts_proc

    if not text:
        return

    tts_active.set()

    mouth_stop = threading.Event()
    threading.Thread(target=mouth.ticker_loop, args=(mouth_stop, cancel_turn), daemon=True).start()

    mouth.send("open", why="tts_start")

    if hud_mod:
        hud_state(
            state=getattr(hud_mod, "STATE_SPEAKING", "speaking"),
            status="Speaking...",
            backend=backend_label,
            memory=len(messages),
        )

    tts_proc = tts.speak(text)

    def _cleanup(proc, stop_evt):
        try:
            proc.wait()
        finally:
            tts_active.clear()
            stop_evt.set()

            mouth.send("close", why="tts_end")
            mouth.send("clear", why="tts_end")

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

    mouth.send("close", why="interrupt")
    mouth.send("clear", why="interrupt")

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
        backend_label = (
            getattr(hud_mod, "BACKEND_OPENAI", "OPENAI")
            if route_key == "red"
            else getattr(hud_mod, "BACKEND_LOCAL", "LOCAL")
        )
        hud_state(
            state=getattr(hud_mod, "STATE_RECORDING", "recording"),
            status=f"Recording ({backend_label})...",
            backend=backend_label,
            memory=len(messages),
        )

    def runner():
        audio_holder["audio"] = record_audio(stop_event, SAMPLE_RATE)

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
    text = transcribe(client, audio, SAMPLE_RATE, STT_MODEL)

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

    handled, skill_reply = run_skill(text)

    if LOG_ENABLED:
        append_jsonl(LOG_PATH, {"ts": time.time(), "role": "user", "text": text, "route": active_route_key})

    if hud_mod:
        hud_state(
            state=getattr(hud_mod, "STATE_THINKING", "thinking"),
            status="Thinking...",
            memory=len(messages),
        )

    if handled:
        if isinstance(skill_reply, dict) and skill_reply.get("mode") == "web_context":
            reply, backend_label = answer_from_web_context(text, skill_reply)
        else:
            reply = skill_reply
            backend_label = "SKILL"
    else:
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


def run_wake_word_turn():
    global audio_holder

    if is_recording or tts_active.is_set() or wake_busy.is_set() or quit_flag.is_set():
        return

    wake_busy.set()
    try:
        cancel_turn.clear()

        print("[WAKE] triggered", flush=True)

        if hud_mod:
            hud_state(
                state=getattr(hud_mod, "STATE_SPEAKING", "speaking"),
                status="Wake acknowledged",
                backend="WAKE",
                memory=len(messages),
            )

        # Stage 1: acknowledge wake word
        speak("Yes?", "WAKE")

        start_wait = time.time()
        while tts_active.is_set():
            if cancel_turn.is_set() or quit_flag.is_set():
                if hud_mod:
                    hud_state(
                        state=getattr(hud_mod, "STATE_IDLE", "idle"),
                        status="Idle",
                        memory=len(messages),
                    )
                return
            if time.time() - start_wait > 3.0:
                break
            time.sleep(0.05)

        time.sleep(0.25)

        if cancel_turn.is_set() or quit_flag.is_set():
            if hud_mod:
                hud_state(
                    state=getattr(hud_mod, "STATE_IDLE", "idle"),
                    status="Idle",
                    memory=len(messages),
                )
            return

        if hud_mod:
            hud_state(
                state=getattr(hud_mod, "STATE_RECORDING", "recording"),
                status="Listening for command",
                backend="WAKE",
                memory=len(messages),
            )

        stop_evt = threading.Event()
        audio_holder = {"audio": np.array([], dtype=np.float32)}

        def stop_later():
            time.sleep(WAKE_FOLLOWUP_SECONDS)
            stop_evt.set()

        threading.Thread(target=stop_later, daemon=True).start()

        audio = record_audio(stop_evt, SAMPLE_RATE)
        audio_holder["audio"] = audio

        duration = len(audio) / SAMPLE_RATE if audio is not None else 0

        if duration < 0.12:
            print(f"[WAKE] ignoring short audio ({duration:.3f}s)", flush=True)
            if hud_mod:
                hud_state(
                    state=getattr(hud_mod, "STATE_IDLE", "idle"),
                    status="Idle",
                    memory=len(messages),
                )
            return

        if hud_mod:
            hud_state(
                state=getattr(hud_mod, "STATE_THINKING", "thinking"),
                status="Transcribing...",
                memory=len(messages),
            )

        text = transcribe(client, audio, SAMPLE_RATE, STT_MODEL)

        if not text:
            print("(no wake follow-up captured)", flush=True)
            if hud_mod:
                hud_state(
                    state=getattr(hud_mod, "STATE_IDLE", "idle"),
                    status="Idle",
                    memory=len(messages),
                )
            return

        print(f"\nYou: {text}", flush=True)

        handled, skill_reply = run_skill(text)

        if handled:
            if isinstance(skill_reply, dict) and skill_reply.get("mode") == "web_context":
                reply, backend_label = answer_from_web_context(text, skill_reply)
            else:
                reply = skill_reply
                backend_label = "SKILL"
        else:
            reply, backend_label = generate("red", text)

        if cancel_turn.is_set():
            print("(cancelled — ignoring wake reply)", flush=True)
            if hud_mod:
                hud_state(
                    state=getattr(hud_mod, "STATE_IDLE", "idle"),
                    status="Idle",
                    memory=len(messages),
                )
            return

        print(f"Assistant ({backend_label}): {reply}\n", flush=True)

        if LOG_ENABLED:
            append_jsonl(LOG_PATH, {"ts": time.time(), "role": "user", "text": text, "route": "wake_followup"})
            append_jsonl(LOG_PATH, {"ts": time.time(), "role": "assistant", "text": reply, "backend": backend_label})

        speak(reply, backend_label)

    except Exception as e:
        print(f"[WAKE] callback error: {e}", flush=True)
    finally:
        wake_busy.clear()


# ----------------------------
# KEYBOARD THREAD
# ----------------------------

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
# CLEAN SHUTDOWN
# ----------------------------

def shutdown():
    global device, wake_listener, tts_proc

    quit_flag.set()
    cancel_turn.set()

    with tts_lock:
        try:
            if tts_proc and tts_proc.poll() is None:
                tts_proc.terminate()
                tts_proc.wait(timeout=1.0)
        except Exception:
            pass

    if wake_listener:
        try:
            wake_listener.stop()
        except Exception:
            pass

    if device:
        try:
            device.close()
        except Exception:
            pass

    try:
        mouth.send("close", why="exit")
        mouth.send("clear", why="exit")
    except Exception:
        pass

    if hud_queue is not None:
        try:
            hud_queue.put({"cmd": "quit"})
        except Exception:
            pass

    time.sleep(0.3)


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

if WAKE_ENABLED:
    try:
        wake_listener = WakeWordListener(
            model_name=WAKE_MODEL,
            threshold=WAKE_THRESHOLD,
            cooldown=WAKE_COOLDOWN,
            sample_rate=SAMPLE_RATE,
            callback=run_wake_word_turn,
        )
        wake_listener.start()
    except Exception as e:
        print(f"[WAKE] failed to start: {e}", flush=True)

try:
    import select
except Exception:
    select = None

try:
    while True:
        try:
            cmd = kbd_q.get_nowait()
        except Empty:
            cmd = None

        if cmd:
            if cmd in ("h", "help", "?"):
                print_help()
            elif cmd in ("q", "quit", "exit"):
                print("Quitting...", flush=True)
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

        if select is None:
            time.sleep(0.01)
            continue

        if device is None:
            break

        rlist, _, _ = select.select([device.fd], [], [], 0.01)
        if not rlist:
            continue

        try:
            events = device.read()
        except OSError as e:
            if e.errno == 9:
                print("[INPUT] device closed", flush=True)
                break
            raise

        for event in events:
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
    shutdown()