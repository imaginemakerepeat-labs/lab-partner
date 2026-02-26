# Lab Partner

Lab Partner is a Raspberry Pi–based, push-to-talk voice assistant designed for hands-on environments.

It uses a USB gamepad as a physical input device: hold a button to record, release to transcribe your speech via OpenAI Whisper, generate a response with GPT-4o, and play it back locally using text-to-speech.

Built for workshops, maker spaces, and experimental lab setups where keyboard interaction is inconvenient or impractical.

---

## Core Flow

1. Hold gamepad button → audio recording begins  
2. Release button → audio is saved locally  
3. Audio sent to Whisper for transcription  
4. Transcript sent to GPT-4o  
5. Response spoken aloud using `espeak-ng`

---

## Design Goals

- Minimal interface (physical button control)
- Clear hardware-to-AI pipeline
- Local audio handling
- Explicit configuration (no hidden magic)
- Raspberry Pi friendly

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/imaginemakerepeat-labs/lab-partner.git
cd lab-partner
```

---

### 2. Install System Dependencies (Raspberry Pi / Debian)

These are required for audio capture and text-to-speech:

```bash
sudo apt update
sudo apt install -y espeak-ng portaudio19-dev libsndfile1
```

---

### 3. Create a Python Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -U pip
```

---

### 4. Install Python Dependencies

```bash
pip install -r requirements.txt
```

---

### 5. Configure Your OpenAI API Key

Create a `.env` file:

```bash
cp .env.example .env
```

Edit `.env` and add:

```bash
OPENAI_API_KEY=your_api_key_here
```

---

### 6. Verify Gamepad Device

Confirm your USB gamepad appears:

```bash
ls -l /dev/input/by-id/
```

If necessary, update `DEVICE_PATH` in `assistant.py` to match your device.

---

### 7. Run Lab Partner

Activate your environment:

```bash
source venv/bin/activate
```

Start the assistant:

```bash
python assistant.py
```

You should see:

```
Ready. Hold the button to speak. Release to get a reply.
```

---

## Usage

Press and hold the configured USB gamepad button to begin recording.

Release the button to:
- Save audio locally
- Transcribe speech using Whisper
- Generate a response with GPT-4o
- Play the response via `espeak-ng`

Designed for physical environments where keyboard interaction is inconvenient.

---

## Features

- 🎮 Hold-to-talk via USB gamepad (BTN_THUMB default)
- 🎤 Local microphone capture (16 kHz)
- 🧠 Speech-to-text via OpenAI Whisper (`whisper-1`)
- 💬 Conversational responses via GPT-4o
- 🔊 Local text-to-speech using `espeak-ng`
- 🧰 Lightweight and Raspberry Pi optimized

---

## Hardware Requirements

- Raspberry Pi (tested on Raspberry Pi OS / Debian)
- USB gamepad (e.g., DragonRise / 0079)
- USB or onboard microphone
- Speaker or audio output device

---

## Software Requirements

System packages:

```bash
sudo apt update
sudo apt install -y espeak-ng portaudio19-dev libsndfile1
```

Python packages are listed in `requirements.txt`.

---

## Security Notes

- `.env` is excluded via `.gitignore`
- Never commit API keys
- Rotate keys immediately if accidentally exposed

---

## Roadmap

- Persistent conversation memory
- Config file instead of hardcoded device path
- Optional wake-word mode
- Optional local LLM mode via Ollama
- Logging and transcript saving

---

## License

MIT License