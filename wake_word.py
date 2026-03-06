import threading
import time
from typing import Optional, Callable

import numpy as np
import sounddevice as sd


class WakeWordListener:
    def __init__(self, model_name, threshold, cooldown, sample_rate, callback: Callable[[], None]):
        self.model_name = model_name
        self.threshold = threshold
        self.cooldown = cooldown
        self.sample_rate = sample_rate
        self.callback = callback

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_fire = 0.0
        self.model = None

    def start(self):
        # Lazy import so normal assistant startup does not explode
        import openwakeword
        from openwakeword.model import Model

        openwakeword.utils.download_models()
        self.model = Model(wakeword_models=[self.model_name])

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print(f"[WAKE] listening for '{self.model_name}'", flush=True)

    def stop(self):
        self._stop.set()

    def _fire(self):
        now = time.time()
        if now - self._last_fire < self.cooldown:
            return
        self._last_fire = now

        try:
            self.callback()
        except Exception as e:
            print(f"[WAKE] callback error: {e}", flush=True)

    def _run(self):
        blocksize = int(self.sample_rate * 0.08)

        def audio_callback(indata, frames, time_info, status):
            if self.model is None:
                return

            audio = np.clip(indata[:, 0], -1, 1)
            pcm = (audio * 32767).astype(np.int16)

            pred = self.model.predict(pcm)
            score = max(pred.values()) if pred else 0.0

            if score >= self.threshold:
                self._fire()

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            blocksize=blocksize,
            dtype="float32",
            callback=audio_callback,
        ):
            while not self._stop.is_set():
                time.sleep(0.2)