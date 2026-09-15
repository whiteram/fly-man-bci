"""Convert a human video (mp4/avi/gif/mov) into the .npy frame stack the
visual-input pipeline consumes:

    python viz/make_stimulus_from_video.py clip.mp4 viz/data/stimuli/my_clip.npy \
        [--fps 30] [--width 96] [--start 0] [--duration 10.5]

Frames are center-cropped to 4:3, rescaled, converted to grayscale, and
stored as (frames, H, W) uint8.  Requires imageio + imageio-ffmpeg:
    pip install imageio imageio-ffmpeg
"""
import argparse
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("src")
ap.add_argument("dst")
ap.add_argument("--fps", type=float, default=30.0)
ap.add_argument("--width", type=int, default=96)
ap.add_argument("--start", type=float, default=0.0, help="seconds to skip")
ap.add_argument("--duration", type=float, default=10.5, help="seconds to take")
args = ap.parse_args()

try:
    import imageio.v3 as iio
except ImportError as e:
    raise SystemExit("需要 imageio：pip install imageio imageio-ffmpeg") from e

meta = iio.immeta(args.src, plugin="pyav")
src_fps = float(meta.get("fps", 30.0))
stride = max(1, round(src_fps / args.fps))
H = round(args.width * 3 / 4)
frames = []
t = args.start
while len(frames) < args.duration * args.fps:
    try:
        img = iio.imread(args.src, plugin="pyav", index=int(t * src_fps))
    except Exception:
        break
    h0, w0 = img.shape[:2]
    # center-crop 4:3 then rescale to (H, args.width)
    if w0 > h0 * 4 / 3:
        cw = int(h0 * 4 / 3)
        img = img[:, (w0 - cw) // 2:(w0 - cw) // 2 + cw]
    else:
        ch = int(w0 * 3 / 4)
        img = img[(h0 - ch) // 2:(h0 - ch) // 2 + ch]
    yy = np.linspace(0, img.shape[0] - 1, H).astype(int)
    xx = np.linspace(0, img.shape[1] - 1, args.width).astype(int)
    g = (img[yy][:, xx] * np.array([0.2126, 0.7152, 0.0722])).sum(-1)
    frames.append(np.clip(g, 0, 255).astype(np.uint8))
    t += stride / src_fps

out = np.stack(frames)
Path(args.dst).parent.mkdir(parents=True, exist_ok=True)
np.save(args.dst, out)
print(f"wrote {args.dst}: {out.shape[0]} frames, {H}x{args.width}, "
      f"{args.fps} fps")
