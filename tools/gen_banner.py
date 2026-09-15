import math

W, H = 1200, 630
OUT = "#2b3a55"          # outline navy
CREAM = "#fbf5ea"        # background
parts = []
parts.append('<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d" font-family="Segoe UI, Helvetica, Arial, sans-serif">' % (W, H, W, H))
parts.append('<rect width="%d" height="%d" fill="%s"/>' % (W, H, CREAM))
parts.append('<rect x="0" y="540" width="%d" height="90" fill="#f2e8d2"/>' % W)
parts.append('<line x1="0" y1="540" x2="%d" y2="540" stroke="#e0d2b4" stroke-width="3"/>' % W)


def rr(x, y, w, h, r, fill, stroke=None, sw=5):
    s = ' stroke="%s" stroke-width="%s"' % (stroke, sw) if stroke else ''
    return '<rect x="%s" y="%s" width="%s" height="%s" rx="%s" fill="%s"%s/>' % (x, y, w, h, r, fill, s)


def circle(cx, cy, r, fill, stroke=None, sw=4):
    s = ' stroke="%s" stroke-width="%s"' % (stroke, sw) if stroke else ''
    return '<circle cx="%s" cy="%s" r="%s" fill="%s"%s/>' % (cx, cy, r, fill, s)


def ellipse(cx, cy, rx, ry, fill, stroke=None, sw=4):
    s = ' stroke="%s" stroke-width="%s"' % (stroke, sw) if stroke else ''
    return '<ellipse cx="%s" cy="%s" rx="%s" ry="%s" fill="%s"%s/>' % (cx, cy, rx, ry, fill, s)


def path(d, stroke, sw, fill="none", cap="round"):
    return '<path d="%s" fill="%s" stroke="%s" stroke-width="%s" stroke-linecap="%s" stroke-linejoin="round"/>' % (d, fill, stroke, sw, cap)


def text(x, y, t, size, fill, weight="normal", anchor="start"):
    return '<text x="%s" y="%s" font-size="%s" fill="%s" font-weight="%s" text-anchor="%s">%s</text>' % (x, y, size, fill, weight, anchor, t)


def hexpath(cx, cy, r, rot=0.0):
    pts = []
    for i in range(6):
        a = math.radians(60 * i + rot)
        pts.append("%.1f,%.1f" % (cx + r * math.cos(a), cy + r * math.sin(a)))
    return "M" + " L".join(pts) + " Z"


GREEN = "#8aa14f"
GREEN_D = "#6f8440"
EYE = "#c65f4e"
SHIRT = "#f5a742"
SHIRT_D = "#de9433"
PANTS = "#3d5a80"
PANTS_D = "#33496b"
SHOE = "#8a5a44"

# ---------------- shadows ----------------
for cx, rx in [(504, 70), (730, 125), (904, 62)]:
    parts.append(ellipse(cx, 552, rx, 12, "#d8c8a6", sw=0))

# ---------------- title ----------------
parts.append('<text x="68" y="118" font-size="58" font-weight="800" fill="%s">FLY-MAN <tspan fill="#ec4899">BCI</tspan></text>' % OUT)
parts.append(text(70, 156, "stimulus \u2192 his fly-brain \u2192 scalp EEG \u2014 the fly-man\u2019s EEG generation model", 19, "#5b6b85"))
parts.append(text(70, 188, "蝇人脑机接口 · 蝇之脑，人身之躯——他急需一套脑机接口", 17, "#7a8aa3"))

# ---------------- stimulus monitor, seen from the side ----------------
parts.append('<defs><clipPath id="scr"><path d="M 472.8 221.5 L 535.2 237.5 L 535.2 341.5 L 472.8 349.5 Z"/></clipPath></defs>')
parts.append(path("M 504 358 L 504 528", OUT, 9))           # slim pole, same spec as the EEG stand
parts.append(ellipse(504, 532, 36, 10, OUT))                # small foot, same spec as the EEG stand
# trapezoid body (the bezel): raised so the screen sits at the fly-man's eye height
parts.append('<path d="M 465 205 L 543 225 L 543 355 L 465 365 Z" fill="%s" stroke="%s" stroke-width="5" stroke-linejoin="round"/>' % (OUT, OUT))
# screen window: a smaller parallel trapezoid, centered in the bezel
parts.append('<path d="M 472.8 221.5 L 535.2 237.5 L 535.2 341.5 L 472.8 349.5 Z" fill="#f4f0e6"/>')
parts.append('<g clip-path="url(#scr)">')
for i in range(9):                                          # checkerboard, clipped to the window
    for j in range(8):
        col = "#e2574c" if (i + j) % 2 == 0 else "#3a4a6b"
        parts.append('<rect x="%.1f" y="%.1f" width="9" height="17" fill="%s"/>' % (468 + i * 9, 216 + j * 17, col))
parts.append('</g>')

# ---------------- EEG amplifier on a single-pole stand ----------------
parts.append(path("M 904 324 L 904 528", OUT, 9))           # single pole
parts.append(ellipse(904, 532, 36, 10, OUT))                # foot (lands on the floor)
parts.append(rr(848, 246, 112, 78, 12, "#ffffff", OUT, 5))  # box, vertical center aligned with the stimulus screen
parts.append(rr(862, 258, 84, 36, 5, "#1b2540"))
ys = [279, 274, 282, 270, 284, 274, 280, 268, 283, 272, 281, 269, 282, 273, 279, 271, 283, 274, 278, 272, 281, 270, 282, 273, 279, 271, 282, 273]
xs = list(range(866, 942, 3))
wave = " ".join("%d,%d" % (x, y) for x, y in zip(xs, ys[: len(xs)]))
parts.append('<polyline points="%s" fill="none" stroke="#ec4899" stroke-width="2.4" stroke-linejoin="round"/>' % wave)
parts.append(text(904, 315, "EEG", 15, OUT, "700", "middle"))

# ---------------- chair (profile) ----------------
parts.append(rr(742, 302, 26, 140, 12, "#4f8a8b", OUT, 5))   # backrest
parts.append(rr(662, 430, 140, 27, 12, "#4f8a8b", OUT, 5))   # seat
parts.append(path("M 676 457 L 668 538", OUT, 8))
parts.append(path("M 788 457 L 798 538", OUT, 8))
parts.append(path("M 672 506 L 794 506", OUT, 6))

# ---------------- fly-man (true side view, facing left) ----------------
# far leg
parts.append(path("M 714 436 L 644 452 L 636 514", PANTS_D, 20))
parts.append(rr(612, 514, 44, 19, 9, "#6f4636", OUT, 4))
# thin profile torso, slight forward lean
parts.append('<path d="M 678 442 C 672 396 676 356 688 344 L 728 340 C 740 352 746 396 744 444 Z" fill="%s" stroke="%s" stroke-width="5" stroke-linejoin="round"/>' % (SHIRT, OUT))
# near leg
parts.append(path("M 702 438 L 628 450 L 618 514", PANTS, 20))
parts.append(rr(590, 514, 48, 20, 10, SHOE, OUT, 4))
# arm, outline pass then fill pass
parts.append(path("M 706 356 L 678 400 L 664 430", OUT, 22))
parts.append(path("M 706 356 L 678 400 L 664 430", SHIRT_D, 13))
parts.append(circle(662, 434, 11, GREEN, OUT, 4))
# neck + head, both on the torso centerline (x = 708)
parts.append(rr(695, 306, 26, 34, 8, GREEN_D, OUT, 4))
parts.append(circle(708, 266, 50, GREEN, OUT, 5))
# one big compound eye, on the front (left) of the head
parts.append(ellipse(686, 282, 24, 28, EYE, OUT, 5))
for hx, hy in [(686, 262), (678, 288), (694, 300)]:
    parts.append('<path d="%s" fill="none" stroke="#e39a8a" stroke-width="1.6"/>' % hexpath(hx, hy, 6.5))
parts.append(circle(676, 258, 5, "#ffffff"))
# small smile at the front of the face
parts.append(path("M 648 298 Q 658 308 670 306", OUT, 4))
# antennae through the cap, leaning forward
parts.append(path("M 692 220 Q 680 192 664 184", OUT, 4))
parts.append(circle(662, 183, 5.5, OUT))
parts.append(path("M 716 222 Q 710 188 694 178", OUT, 4))
parts.append(circle(692, 177, 5.5, OUT))
# EEG cap (dome + band)
parts.append('<path d="M 662 246 A 48 42 0 0 1 754 246 Z" fill="#ffffff" stroke="%s" stroke-width="5" stroke-linejoin="round"/>' % OUT)
parts.append(rr(654, 240, 108, 13, 6.5, "#e8edf5", OUT, 4))
for ex, ey, ec in [(686, 224, "#ec4899"), (708, 212, "#38bdf8"), (730, 224, "#f7c948")]:
    parts.append(circle(ex, ey, 6.5, ec, OUT, 3))
# wire: back of the cap -> amplifier
parts.append(path("M 760 244 C 796 236 832 234 856 246", OUT, 4.5))
parts.append(circle(760, 244, 6, OUT))

# ---------------- caption ----------------
parts.append(text(1134, 610, "166,700 neurons \u00b7 Lee Perry-Smith head scan \u00b7 45-241 ch 10-20 caps \u00b7 100% simulation", 13, "#a5946f", "600", "end"))
parts.append('</svg>')

svg = "\n".join(parts)
with open("assets/banner.svg", "w", encoding="utf-8") as f:
    f.write(svg)
print("banner.svg written,", len(svg), "bytes")
