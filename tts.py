# tts.py
import subprocess
import threading
from typing import Optional, Callable


class TTSController:
    """
    Owns the TTS subprocess + interrupt behavior.
    Optional hooks let caller integrate HUD + mouth without duplicating logic.
    """

    def __init__(
        self,
        engine: str,
        voice: str,
        rate: int,
        on_speaking: Optional[Callable[[], None]] = None,
        on_idle: Optional[Callable[[], None]] = None,
        on_interrupt: Optional[Callable[[], None]] = None,
    ):
        self.engine = engine
        self.voice = voice
        self.rate = rate

        self.on_speaking = on_speaking or (lambda: None)
        self.on_idle = on_idle or (lambda: None)
        self.on_interrupt = on_interrupt or (lambda: None)

        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    def speak(self, text: str):
        if not text:
            return

        with self._lock:
            # stop any currently speaking process
            if self._proc and self._proc.poll() is None:
                try:
                    self._proc.terminate()
                except Exception:
                    pass

            self._proc = subprocess.Popen(
                [self.engine, "-v", self.voice, "-s", str(self.rate), text],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        self.on_speaking()

        # watcher: when TTS finishes, go idle
        def watcher(proc: subprocess.Popen):
            try:
                proc.wait()
            except Exception:
                pass
            self.on_idle()

        threading.Thread(target=watcher, args=(self._proc,), daemon=True).start()

    def interrupt(self):
        with self._lock:
            if self._proc and self._proc.poll() is None:
                try:
                    self._proc.terminate()
                except Exception:
                    pass
        self.on_interrupt()
