# audio.py
import os
import tempfile

import numpy as np
import sounddevice as sd
import soundfile as sf


def record_audio(stop_event, sample_rate: int):
    chunks = []

    def callback(indata, frames, time_info, status):
        chunks.append(indata.copy())

    with sd.InputStream(samplerate=sample_rate, channels=1, callback=callback):
        while not stop_event.is_set():
            sd.sleep(50)

    if not chunks:
        return np.zeros((0, 1), dtype=np.float32)

    return np.concatenate(chunks, axis=0)


def transcribe(client, audio, sample_rate: int, stt_model: str) -> str:
    if audio is None or getattr(audio, "size", 0) == 0:
        return ""

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio, sample_rate)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as f:
            resp = client.audio.transcriptions.create(
                model=stt_model,
                file=f,
            )
        return (resp.text or "").strip()
    finally:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
