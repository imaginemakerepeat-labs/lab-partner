# hud.py
import time
import pygame

STATE_IDLE = "IDLE"
STATE_RECORDING = "RECORDING"
STATE_THINKING = "THINKING"
STATE_SPEAKING = "SPEAKING"
STATE_INTERRUPTED = "INTERRUPTED"

BACKEND_OPENAI = "OPENAI"
BACKEND_LOCAL = "LOCAL"


def color_for(state, backend):
    if state == STATE_IDLE:
        return (35, 35, 40)
    if state == STATE_RECORDING:
        return (120, 35, 35) if backend == BACKEND_OPENAI else (35, 120, 55)
    if state == STATE_THINKING:
        return (35, 65, 120)
    if state == STATE_SPEAKING:
        return (90, 45, 120)
    if state == STATE_INTERRUPTED:
        return (140, 120, 35)
    return (35, 35, 40)


def badge_color(backend):
    if backend == BACKEND_OPENAI:
        return (220, 80, 80)
    if backend == BACKEND_LOCAL:
        return (80, 220, 120)
    return (160, 160, 160)


def run_hud(queue):
    """
    Receives dict messages from queue:
      {"backend": "OPENAI"/"LOCAL", "state": "...", "status": "...", "memory": int, "flash": bool}
    Special:
      {"cmd": "quit"} exits.
    """
    pygame.init()
    screen = pygame.display.set_mode((520, 240))
    pygame.display.set_caption("Lab Partner HUD")
    clock = pygame.time.Clock()

    font_title = pygame.font.SysFont(None, 34)
    font_body = pygame.font.SysFont(None, 24)
    font_small = pygame.font.SysFont(None, 18)

    backend = BACKEND_OPENAI
    state = STATE_IDLE
    memory_turns = 0
    status_line = "Idle / Ready"
    flash_until = 0.0

    running = True
    while running:
        # allow closing the window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # drain queue
        while True:
            try:
                msg = queue.get_nowait()
            except Exception:
                break

            if isinstance(msg, dict) and msg.get("cmd") == "quit":
                running = False
                break

            if not isinstance(msg, dict):
                continue

            if msg.get("backend") is not None:
                backend = msg["backend"]
            if msg.get("state") is not None:
                state = msg["state"]
            if msg.get("status") is not None:
                status_line = msg["status"]
            if msg.get("memory") is not None:
                memory_turns = int(msg["memory"])
            if msg.get("flash"):
                flash_until = time.time() + 0.35

        screen.fill(color_for(state, backend))

        panel = pygame.Rect(18, 18, 484, 204)
        pygame.draw.rect(screen, (15, 15, 18), panel, border_radius=16)
        pygame.draw.rect(screen, (55, 55, 65), panel, width=2, border_radius=16)

        title = font_title.render("LAB PARTNER — HUD", True, (235, 235, 240))
        screen.blit(title, (34, 30))

        badge = pygame.Rect(340, 28, 146, 34)
        pygame.draw.rect(screen, badge_color(backend), badge, border_radius=10)
        badge_text = font_body.render(backend, True, (10, 10, 12))
        screen.blit(badge_text, (badge.x + 12, badge.y + 7))

        screen.blit(font_body.render("STATE:", True, (200, 200, 210)), (34, 78))
        screen.blit(font_body.render(state, True, (240, 240, 245)), (120, 78))

        screen.blit(font_body.render("MEMORY:", True, (200, 200, 210)), (34, 110))
        screen.blit(font_body.render(f"{memory_turns} turns", True, (240, 240, 245)), (120, 110))

        status_box = pygame.Rect(34, 142, 452, 54)
        pygame.draw.rect(screen, (25, 25, 30), status_box, border_radius=12)
        pygame.draw.rect(screen, (70, 70, 85), status_box, width=2, border_radius=12)

        screen.blit(font_body.render(status_line, True, (235, 235, 245)),
                    (status_box.x + 14, status_box.y + 15))

        screen.blit(font_small.render("HUD driven by assistant.py events", True, (170, 170, 185)),
                    (34, 202))

        if time.time() < flash_until:
            overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
            overlay.fill((255, 220, 80, 90))
            screen.blit(overlay, (0, 0))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()