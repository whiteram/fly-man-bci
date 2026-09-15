"""Generate a human-readable demo video as a stimulus source:
a bright ball bouncing on a textured background with a drifting dark
bar -- readable by eye, and rich enough (motion, contrast) to drive the
fly cascade.  Output: (frames, H, W) uint8 .npy consumed by
visual_inputs.json entries with kind="video"."""
import numpy as np
from pathlib import Path

FPS = 30.0
DUR_S = 10.5
H, W = 72, 96

frames = []
n = int(DUR_S * FPS)
bg_y, bg_x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
base = 110 + 25 * np.sin(bg_y / 9.0) * np.cos(bg_x / 11.0)   # soft texture

ball_r, ball_v = 7.0, np.array([2.6, 1.9])                   # px/frame
ball_p = np.array([20.0, 40.0])
bar_x = 0.0
for f in range(n):
    img = np.repeat(base[..., None], 1, axis=2).astype(np.float64)[..., 0]
    # drifting dark vertical bar
    bar_x = (bar_x + 1.1) % (W + 26) - 13
    img -= 55 * np.exp(-((bg_x - bar_x) ** 2) / (2 * 4.5 ** 2))
    # bouncing bright ball
    ball_p += ball_v
    for ax, lim in ((0, H), (1, W)):
        if ball_p[ax] < ball_r or ball_p[ax] > lim - ball_r:
            ball_v[ax] *= -1
            ball_p[ax] = np.clip(ball_p[ax], ball_r, lim - ball_r)
    ball = ((bg_y - ball_p[0]) ** 2 + (bg_x - ball_p[1]) ** 2) \
        <= ball_r ** 2
    img[ball] = 235
    frames.append(np.clip(img, 0, 255).astype(np.uint8))

out = Path(__file__).parent / "data" / "stimuli" / "demo_bounce.npy"
out.parent.mkdir(parents=True, exist_ok=True)
np.save(out, np.stack(frames))
print(f"wrote {out}: {n} frames @ {FPS:.0f} fps, {H}x{W}")
