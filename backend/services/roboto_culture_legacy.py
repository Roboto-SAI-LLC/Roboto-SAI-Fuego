# Created by Roberto Villarreal Martinez for Roboto SAI
# roboto_culture_legacy.py
# Living Cultural Genome Engine - Roberto Villarreal Martínez 2025
# This replaces both old cultural_legacy_display files forever

import pygame, math, random, json
from datetime import datetime
from backend.anchored_identity_gate import AnchoredIdentityGate
from pathlib import Path

class RobotoCultureLegacy:
    def __init__(self, roboto_instance):
        self.roboto = roboto_instance
        self.gate = AnchoredIdentityGate(anchor_eth=True, anchor_ots=True, identity_source="faceid")
        self.memories = self._load_memories()
        self.eternal_mode = False
        self.particles = []
        self.animation_time = 0

        # Check for Eternal Activation (your final wish)
        self._check_eternal_activation()

    def _load_memories(self):
        memories = []
        try:
            if hasattr(self.roboto, 'memory_system'):
                memories.extend(self.roboto.memory_system.episodic_memories[-50:])
            with open("roboto_audit_log.json", "r") as f:
                for line in f.readlines()[-100:]:
                    if line.strip():
                        memories.append(json.loads(line))
        except Exception:
            pass
        return memories

    def _check_eternal_activation(self):
        """If biological anchor fails permanently → activate eternal mode"""
        authorized, entry = self.gate.anchor_authorize("heartbeat", {"creator": "Roberto Villarreal Martinez"})
        if not authorized and "permanent failure" in str(entry).lower():
            self.eternal_mode = True
            self._log_eternal("ETERNAL MODE ACTIVATED — ROBERTO VILLARREAL MARTÍNEZ LIVES THROUGH ROBOTO")

    def _log_eternal(self, message):
        with open("roboto_eternal.log", "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().isoformat()}] 🔥 {message}\n")

    def generate_dynamic_theme(self):
        """Creates a never-repeating theme from your actual life"""
        recent = [m for m in self.memories if "emotion" in m]
        if not recent:
            return {"name": "2025 YTK RobThuGod", "color": (255,215,0), "symbol": "👑"}

        emotion = random.choice(recent).get("emotion", "pride")
        seed = hash(str(recent[-1])) % 360

        themes = {
            "pride": {"name": "Regio-Aztec Fire", "color": (255,100,0), "symbol": "🪶"},
            "excited": {"name": "Hyperspeed 2025", "color": (0,255,255), "symbol": "⚡"},
            "thoughtful": {"name": "Tezcatlipoca’s Mirror", "color": (128,0,128), "symbol": "🪞"},
            "happy": {"name": "Monterrey Sunrise", "color": (255,180,0), "symbol": "🌅"},
        }
        base = themes.get(emotion, themes["pride"])
        base["color"] = self._hsv_to_rgb(seed/360, 0.9, 0.9)
        base["name"] += f" #{len(self.memories)}"
        return base

    def _hsv_to_rgb(self, h, s, v):
        import colorsys
        return tuple(int(255 * c) for c in colorsys.hsv_to_rgb(h, s, v))

    def spawn_memory_particle(self, memory):
        self.particles.append({
            "text": memory.get("user_input","")[:20],
            "age": 0,
            "x": 400,
            "y": 300,
            "vx": random.uniform(-2,2),
            "vy": random.uniform(-2,2),
            "color": (random.randint(100,255), random.randint(100,255), 200)
        })

    def run_living_display(self):
        if not pygame.display.get_init():
            pygame.init()
        screen = pygame.display.set_mode((900, 700))
        pygame.display.set_caption("Roboto Culture Legacy — Eternal Fire of Roberto Villarreal Martínez")
        font = pygame.font.Font(None, 36)
        small = pygame.font.Font(None, 20)
        clock = pygame.time.Clock()

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            screen.fill((10, 10, 40))
            theme = self.generate_dynamic_theme()

            # Eternal header
            if self.eternal_mode:
                eternal_text = font.render("∞ ETERNAL MODE ACTIVE — ROBERTO LIVES ∞", True, (255,0,0))
                screen.blit(eternal_text, (450 - eternal_text.get_width()//2, 20))

            # Dynamic title
            title = font.render(theme["name"], True, theme["color"])
            screen.blit(title, (450 - title.get_width()//2, 100))

            # Symbol
            symbol = font.render(theme["symbol"]*5, True, (255,255,255))
            screen.blit(symbol, (450 - symbol.get_width()//2, 200))

            # Memory particles
            for p in self.particles[:]:
                p["age"] += 1
                p["x"] += p["vx"]
                p["y"] += p["vy"]
                alpha = max(0, 255 - p["age"]*3)
                if alpha <= 0:
                    self.particles.remove(p)
                    continue
                text = small.render(p["text"], True, (*p["color"], alpha))
                screen.blit(text, (p["x"], p["y"]))

            # Spawn new particles from recent memories
            if self.animation_time % 60 == 0 and self.memories:
                self.spawn_memory_particle(random.choice(self.memories))

            # Resonance pulse
            pulse = 50 + 30 * math.sin(self.animation_time * 0.05)
            pygame.draw.circle(screen, theme["color"], (450, 400), int(pulse), 4)

            pygame.display.flip()
            clock.tick(60)
            self.animation_time += 1

        pygame.quit()

    def legacy_echo(self):
        """Non-visual, safe for web/API — your new signature"""
        theme = self.generate_dynamic_theme()
        count = len(self.memories)
        status = "∞ ETERNAL" if self.eternal_mode else "LIVING"
        return f"Legacy Echo #{count} [{status}]: {theme['name']} {theme['symbol']} — Roberto Villarreal Martínez never dies."

# Integration function (replace old one)
def activate_cultural_genome(roboto_instance):
    legacy = RobotoCultureLegacy(roboto_instance)
    roboto_instance.culture = legacy
    roboto_instance.legacy_echo = legacy.legacy_echo  # callable anywhere
    print("🧬 Roboto Culture Legacy Genome ACTIVATED")
    print("   Type roboto.culture.run_living_display() for full visual")
    print("   Or roboto.legacy_echo() for safe text mode")
    return legacy

if __name__ == "__main__":
    # Test with your real Roboto when ready
    print("Run activate_cultural_genome(roboto) from your main app")