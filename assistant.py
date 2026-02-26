#!/usr/bin/env python3
"""
Work Assistant — Raspberry Pi / Linux (Gamepad Hold-to-Talk)

Device: DragonRise/0079 USB Gamepad (event-joystick)
PTT: Hold BTN_THUMB (your "RED" button on event9) to record.
Release to transcribe (Whisper) and send to GPT-4o, then speak (espeak-ng/espeak).
"""

import os
import sys
import tempfile
import subprocess

import numpy as np
import sounddevice as sd
import soundfile as sf

from dotenv import load_dotenv
from openai import OpenAI

from evdev import InputDevice, ecodes

# ----------------------------
# CONFIG
# ----------------------------

# Your stable by-id symlink (from your ls output)
DEVICE_PATH = "/dev/input/by-id/usb-0079_USB_Gamepad-event-joystick"

# Your RED button on event9 shows up as BTN_THUMB (code 289)
PTT_KEY = ecodes.BTN_THUMB  # 289

SAMPLE_RATE = 16000
CHANNELS = 1

MODEL_CHAT = "gpt-4o"
MODEL_STT = "whisper-1"

# If you record silence, set this to a specific input device index.
# To list: python -c "import sounddevice as sd; print(sd.query_devices())"
AUDIO_DEVICE_INDEX = None

# ----------------------------
# Setup
# ----------------------------

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("\n❌ Error: OPENAI_API_KEY not found. Put it in .env or export it.\n")
    sys.exit(1)

client = OpenAI(api_key=api_key)

# Conversation history (persists for the session)
conversation_history = [
    {
        "role": "system",
        "content": (
            "You are a sharp, efficient work assistant helping with podcast production. "
            "You help analyze content, suggest debate angles, summarize ideas, and support "
            "the creative workflow. Keep responses concise and spoken-word friendly — "
            "no markdown, no bullet symbols, just clean natural speech."
        )
    }
]


# ----------------------------
# TTS
# ----------------------------

def speak(text: str):
    """Try Linux TTS; always print."""
    print(f"\n🤖 Assistant: {text}\n")

    # Prefer espeak-ng, then espeak. If neither installed, just print.
    for cmd in (["espeak-ng", text], ["espeak", text]):
        try:
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        except FileNotFoundError:
            continue


# ----------------------------
# OpenAI: chat + transcription
# ----------------------------

def chat(user_input: str) -> str:
    """Send message to GPT and get response."""
    conversation_history.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model=MODEL_CHAT,
        messages=conversation_history,
        temperature=0.7,
        max_tokens=500,
    )

    reply = response.choices[0].message.content
    conversation_history.append({"role": "assistant", "content": reply})
    return reply


def transcribe_audio(file_path: str) -> str:
    """Send recorded audio to OpenAI Whisper and return transcript."""
    print("📝 Transcribing audio...")
    with open(file_path, "rb") as f:
        result = client.audio.transcriptions.create(
            model=MODEL_STT,
            file=f,
        )
    text = getattr(result, "text", "") or ""
    print(f"🗣️  You (via STT): {text}\n")
    return text


# ----------------------------
# Hold-to-talk recorder
# ----------------------------

class HoldRecorder:
    def __init__(self, samplerate=16000, channels=1, device_index=None):
        self.samplerate = samplerate
        self.channels = channels
        self.device_index = device_index
        self.stream = None
        self.frames = []
        self.recording = False

    def _callback(self, indata, frames, time_info, status):
        # status may contain xruns/overflows; we keep going.
        if self.recording:
            self.frames.append(indata.copy())

    def start(self):
        if self.recording:
            return

        self.frames = []
        self.recording = True

        self.stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            dtype="float32",
            callback=self._callback,
            device=self.device_index,
        )
        self.stream.start()
        print("🔴 Recording... (hold RED / BTN_THUMB)")

    def stop_save_wav(self) -> str | None:
        if not self.recording:
            return None

        self.recording = False
        try:
            if self.stream:
                self.stream.stop()
                self.stream.close()
        finally:
            self.stream = None

        if not self.frames:
            print("⚠️ No audio captured.")
            return None

        audio = np.concatenate(self.frames, axis=0)

        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        sf.write(path, audio, self.samplerate)
        print(f"✅ Saved: {path}")
        return path


# ----------------------------
# Main loop: gamepad events
# ----------------------------

def main():
    if not os.path.exists(DEVICE_PATH):
        print(f"\n❌ Device path not found:\n  {DEVICE_PATH}\n")
        print("Check with: ls -l /dev/input/by-id/\n")
        sys.exit(1)

    dev = InputDevice(DEVICE_PATH)

    print("=" * 62)
    print("  Raspberry Pi Work Assistant (Gamepad Hold-to-Talk)")
    print("=" * 62)
    print(f"  Input device : {dev.path} ({dev.name})")
    print("  Action       : Hold RED (BTN_THUMB) to talk → release to send")
    print("  Quit         : Ctrl+C")
    print("=" * 62)
    print("✅ Ready.\n")

    recorder = HoldRecorder(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        device_index=AUDIO_DEVICE_INDEX,
    )

    try:
        for event in dev.read_loop():
            if event.type != ecodes.EV_KEY:
                continue

            if event.code != PTT_KEY:
                continue

            # event.value: 1=down, 0=up, 2=repeat/hold (ignore 2)
            if event.value == 1:
                print("🎮 PTT down (BTN_THUMB)")
                recorder.start()

            elif event.value == 0:
                print("🎮 PTT up (BTN_THUMB)")
                wav_path = recorder.stop_save_wav()
                if not wav_path:
                    continue

                try:
                    user_text = transcribe_audio(wav_path).strip()
                    if not user_text:
                        print("⚠️ No speech detected.")
                        continue

                    print("⏳ Thinking...")
                    reply = chat(user_text)
                    speak(reply)

                except Exception as e:
                    print(f"❌ Error processing audio: {e}")

                finally:
                    try:
                        os.remove(wav_path)
                    except OSError:
                        pass

    except KeyboardInterrupt:
        print("\n👋 Exiting assistant. Goodbye!")


if __name__ == "__main__":
    main()