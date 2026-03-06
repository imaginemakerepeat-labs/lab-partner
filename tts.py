import subprocess
import threading


class TTSController:
    def __init__(self, engine: str, voice: str, rate: str):
        self.engine = engine
        self.voice = voice
        self.rate = rate
        self.proc = None
        self.lock = threading.Lock()

    def speak(self, text: str):
        if not text:
            return None

        with self.lock:
            if self.proc and self.proc.poll() is None:
                try:
                    self.proc.terminate()
                except Exception:
                    pass

            self.proc = subprocess.Popen(
                [self.engine, "-v", self.voice, "-s", str(self.rate), text],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return self.proc

    def interrupt(self):
        with self.lock:
            if self.proc and self.proc.poll() is None:
                try:
                    self.proc.terminate()
                except Exception:
                    pass

    def current_proc(self):
        return self.proc