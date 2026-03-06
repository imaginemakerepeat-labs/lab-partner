# mouth.py
import socket
import threading
import time
from typing import Optional


class MouthController:
    """
    UDP mouth driver + simple ticker animation.
    Commands emitted: open / wide / close / clear
    Payload: "<seq>|<epoch>|<cmd>"
    """

    def __init__(self, enabled: bool, ip: str, port: int, interval: float = 0.12):
        self.enabled = enabled
        self.ip = ip
        self.port = port
        self.interval = interval

        self._seq = 0
        self._sock: Optional[socket.socket] = None

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        if self.enabled:
            try:
                self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            except Exception:
                self._sock = None
                self.enabled = False

    def send(self, cmd: str, why: str = ""):
        if not self.enabled or not self._sock:
            return

        self._seq += 1
        payload = f"{self._seq}|{time.time():.3f}|{cmd}"

        try:
            self._sock.sendto(payload.encode(), (self.ip, self.port))
            if why:
                print(f"[MOUTH->] seq={self._seq} cmd={cmd} why={why}", flush=True)
            else:
                print(f"[MOUTH->] seq={self._seq} cmd={cmd}", flush=True)
        except Exception:
            pass

    def stop_ticker(self):
        self._stop.set()

    def start_ticker(self):
        if not self.enabled or not self._sock:
            return

        # stop any existing ticker
        self.stop_ticker()
        self._stop.clear()

        def run():
            self.send("open", "tts_start")

            pattern = ["wide", "close", "open", "wide", "open"]
            i = 0

            while not self._stop.is_set():
                self.send(pattern[i % len(pattern)], "viseme")
                i += 1
                time.sleep(self.interval)

            self.send("close", "ticker_end")
            self.send("clear", "tts_end")

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
