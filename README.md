# Work Assistant

A Mac CLI assistant triggered by a global hotkey. Press **Ctrl + Shift + Space** to activate, type your message, and hear the response spoken aloud via macOS TTS.

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add your OpenAI API key

```bash
cp .env.example .env
# Edit .env and paste your key
```

### 3. Grant Accessibility permissions (required for global hotkey)

Go to: **System Settings → Privacy & Security → Accessibility**
Add your terminal app (Terminal, iTerm2, etc.) to the allowed list.

### 4. Run

```bash
python assistant.py
```

## Usage

| Action | How |
|---|---|
| Trigger assistant | `Ctrl + Shift + Space` |
| Type your message | At the `You:` prompt |
| Hear response | Spoken via macOS Samantha voice |
| Quit | Press `Escape` |

## Customization

- **Change voice**: Edit the `say -v Samantha` line in `speak()` — run `say -v ?` to list all available voices
- **Change model**: Swap `gpt-4o` for `gpt-4-turbo` or `gpt-3.5-turbo` in `chat()`
- **Change system prompt**: Edit the `system` message in `conversation_history` to adjust the assistant's personality and focus
- **Change hotkey**: Modify the key combo in `on_press()`

## Next Steps (from your scrum list)

- [ ] Load and search podcast transcripts
- [ ] Add Whisper speech-to-text input
- [ ] Timestamp-based clip pulling
- [ ] Structured reasoning / debate mode
- [ ] Export conversation + clip metadata
