# ts_env_min.py
# Minimal Team Spirit–style environment + actions (no learning).

import random
from dataclasses import dataclass
import math

# --- 1) Actions (discrete) ---------------------------------------------------
ACTIONS = ("LEFT", "STAY", "RIGHT")   # keep it simple for now

# >>> NEW: bin size used for state bucketing
BIN = 50.0


# --- 2) Basic geometry -------------------------------------------------------
@dataclass
class Dims:
    W: int = 800
    H: int = 600
    frame_left: float = 40.0           # ~5% of width
    frame_right: float = 760.0         # ~95% of width
    paddle_y: float = 600 - 26.0 * 3   # paddle line (approx from your game)
    paddle_w: float = 72.0
    ball_r: float = 8.0

STEP_PX = 13.5   # paddle move per tick (matches your bots)
DT      = 0.02   # logical tick (not used heavily yet)
TRIAL_SECONDS = 45.0

# --- 3) Environment ----------------------------------------------------------
class TeamSpiritEnvMin:
    """
    Minimal 1-ball, 2-paddle environment:
      - P1 is the controllable paddle (bottom line).
      - P0 (partner) sits roughly in the center for now.
      - One ball drops straight down; episode ends when it reaches the paddle line.
    Later we can add: angles, multiple balls, 'col' vs 'nonCol', bounces, etc.
    """
    def __init__(self, seed: int | None = None, block_type: str = "col"):
        assert block_type in ("col", "nonCol")
        self.rng = random.Random(seed)
        self.dim = Dims()
        self.block_type = block_type
        self.reset()

    # --- Public API ---
    def reset(self):
        d = self.dim
        self.p0_pos = (d.frame_left + d.frame_right)/2 - d.paddle_w/2
        self.p1_pos = d.W * 0.67 - d.paddle_w/2

        # start ball near top with a downward angle (35°..145°)
        self.ball_x = self.rng.randint(int(d.frame_left + 50), int(d.frame_right - 50))
        self.ball_y = d.ball_r + 5
        self.ball_speed = 10.0
        self.ball_ang   = math.radians(self.rng.uniform(35, 145))  # downward in screen coords

        self.elapsed = 0.0
        self.ticks = 0
        self.done  = False
        return self.observe()

    def observe(self):
        """Return a compact, hashable state string (good for IBL later)."""
        d = self.dim
        bx = round(self.ball_x / 50)
        by = round(self.ball_y / 50)
        px = round(self.p1_pos / 50)
        return f"{bx}:{by}:{px}:{self.block_type}"

    def raw_state(self):
        """Optional: full dict if you want to inspect values."""
        return {
            "p0_pos": self.p0_pos,
            "p1_pos": self.p1_pos,
            "ball_x": self.ball_x,
            "ball_y": self.ball_y,
            "block_type": self.block_type,
            "ticks": self.ticks,
        }

    def step(self, action: str):
        if self.done:
            return self.observe(), 0.0, True, {}

        # 1) Move P1 within bounds
        d = self.dim
        left_bound  = d.frame_left
        right_bound = d.frame_right - d.paddle_w
        if action == "LEFT":
            self.p1_pos = max(left_bound, self.p1_pos - STEP_PX)
        elif action == "RIGHT":
            self.p1_pos = min(right_bound, self.p1_pos + STEP_PX)

        # 2) Move ball by angle
        r = d.ball_r
        prev_x, prev_y = self.ball_x, self.ball_y
        self.ball_x += self.ball_speed * math.cos(self.ball_ang)
        self.ball_y += self.ball_speed * math.sin(self.ball_ang)

        # 3) Collide with vertical walls (left/right): reflect across vertical
        if self.ball_x < d.frame_left + r:
            self.ball_x = d.frame_left + r
            self.ball_ang = math.pi - self.ball_ang
        if self.ball_x > d.frame_right - r:
            self.ball_x = d.frame_right - r
            self.ball_ang = math.pi - self.ball_ang

        # 4) Collide with top wall (horizontal): reflect across horizontal
        if self.ball_y < d.ball_r:
            self.ball_y = d.ball_r
            self.ball_ang = -self.ball_ang

        reward = 0.0
        # 5) Paddle collision (horizontal surface + aiming offset)
        paddle_line = d.paddle_y
        if self.ball_y >= paddle_line:
            caught = (self.ball_x >= self.p1_pos) and (self.ball_x <= self.p1_pos + d.paddle_w)
            reward = 5.0 if caught else -5.0       # keep your ±5 if you like
            # respawn a new ball near the top so play continues
            self.ball_x = self.rng.randint(int(d.frame_left + 50), int(d.frame_right - 50))
            self.ball_y = d.ball_r + 5

        # 4) Advance the trial clock and decide done by time
        self.elapsed += DT
        self.done = (self.elapsed >= TRIAL_SECONDS)
        
        return self.observe(), reward, self.done, {}

        # paddle_y = d.paddle_y
        # if (prev_y + r) < paddle_y <= (self.ball_y + r):  # crossed the paddle line
        #     # horizontally overlapping the paddle?
        #     if (self.ball_x + r) >= self.p1_pos and (self.ball_x - r) <= (self.p1_pos + d.paddle_w):
        #         # reflect and add offset based on impact position
        #         paddle_cx = self.p1_pos + d.paddle_w/2
        #         impact_norm = (self.ball_x - paddle_cx) / (d.paddle_w/2)  # -1..1
        #         max_offset = math.radians(25)  # tune: more = sharper aim
        #         self.ball_ang = -self.ball_ang + impact_norm * max_offset
        #         # nudge above the paddle to avoid re-colliding next tick
        #         self.ball_y = paddle_y - r
        #         reward = +1.0  # optional tiny hit reward (can be 0.0 for now)
        #     else:
        #         # MISS: end the episode (simple version)
        #         self.done = True
        #         return self.observe(), -5.0, True, {}

        # self.ticks += 1
        # return self.observe(), reward, self.done, {}
    
        def observe(self):
            """Return a compact, hashable state string (IBL-friendly)."""
            d = self.dim
            bx = round(self.ball_x / BIN)
            by = round(self.ball_y / BIN)
            px = round(self.p1_pos / BIN)
            return f"{bx}:{by}:{px}:{self.block_type}"

        # >>> NEW: pretty printer for geometry, bins, and actions
    def describe(self):
        d = self.dim
        # ball travel extents inside the frame (touching walls)
        x_min = d.frame_left + d.ball_r
        x_max = d.frame_right - d.ball_r
        y_min = d.ball_r
        y_max_world = d.H - d.ball_r
        y_max_play  = d.paddle_y  # we terminate when reaching paddle line

        # paddle extents for agent
        p_left  = d.frame_left
        p_right = d.frame_right - d.paddle_w

        def b(v):  # bin index helper
            return int(v // BIN)

        info = {
            "ACTIONS": list(ACTIONS),
            "BIN_SIZE": BIN,
            "BALL_X_RANGE_PX": (x_min, x_max),
            "BALL_Y_RANGE_PX": (y_min, y_max_play),
            "BALL_X_RANGE_BINS": (b(x_min), b(x_max)),
            "BALL_Y_RANGE_BINS": (b(y_min), b(y_max_play)),
            "PADDLE_X_RANGE_PX": (p_left, p_right),
            "PADDLE_X_RANGE_BINS": (b(p_left), b(p_right)),
            "PADDLE_Y_PX": d.paddle_y,
            "FRAME_BOUNDS_PX": {
                "left": d.frame_left, "right": d.frame_right,
                "top": 0.0, "bottom": d.H
            },
        }

        # print nicely
        print("\n=== TeamSpiritEnvMin Description ===")
        print(f"ACTIONS: {info['ACTIONS']}")
        print(f"BIN_SIZE: {info['BIN_SIZE']}")
        print(f"BALL X px:  [{x_min:.1f}, {x_max:.1f}]   bins: [{b(x_min)}, {b(x_max)}]")
        print(f"BALL Y px:  [{y_min:.1f}, {y_max_play:.1f}]   bins: [{b(y_min)}, {b(y_max_play)}]  (play ends at paddle_y)")
        print(f"PADDLE X px:[{p_left:.1f}, {p_right:.1f}]   bins: [{b(p_left)}, {b(p_right)}]")
        print(f"PADDLE Y px: {d.paddle_y:.1f}")
        print(f"FRAME px: left={d.frame_left:.1f}, right={d.frame_right:.1f}, top=0.0, bottom={d.H:.1f}")
        print("====================================\n")
        return info


    
# --- 4) Tiny sanity run (no learning) ----------------------------------------
if __name__ == "__main__":
    env = TeamSpiritEnvMin(seed=42, block_type="col")
    s = env.reset()
    print("Initial state:", s, env.raw_state())

    # Take 10 random actions just to show the interface
    for t in range(10):
        a = random.choice(ACTIONS)
        s, r, done, info = env.step(a)
        print(f"t={t:02d}  a={a:>5s}  s={s}  r={r:+.1f}  done={done}")
        if done:
            break
