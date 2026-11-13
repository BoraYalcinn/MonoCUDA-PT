# writes a procedural equirectangular sky (gradient + sun) as a little-endian PFM, same layout the renderer expects
import numpy as np, sys
W, H = 1024, 512
out = sys.argv[1] if len(sys.argv) > 1 else "env/sky.pfm"
u = (np.arange(W) + 0.5) / W
v = (np.arange(H) + 0.5) / H
phi = (u - 0.5) * 2 * np.pi          # same mapping as the renderer: u = atan2(z, x) / 2pi + 0.5
theta = v * np.pi                    # v = 0 is straight up (+y)
sinT = np.sin(theta)[:, None]
d = np.stack([np.cos(phi)[None, :] * sinT,
              np.cos(theta)[:, None] * np.ones((1, W)),
              np.sin(phi)[None, :] * sinT], axis=-1)   # (H, W, 3) = (x, y, z)

up = np.clip(d[..., 1], -1, 1)
t = np.clip(up, 0, 1) ** 0.5
zenith = np.array([0.20, 0.45, 1.00]); horizon = np.array([0.90, 0.95, 1.00])
sky = horizon * (1 - t[..., None]) + zenith * t[..., None]
ground = np.array([0.25, 0.22, 0.18])
img = np.where(up[..., None] >= 0, sky, ground) * 0.8

# sun: 2 degree radius disk, 50 degrees above the horizon, plus a soft glow around it
elev, azim = np.radians(50.0), np.radians(30.0)
sun = np.array([np.cos(elev) * np.cos(azim), np.sin(elev), np.cos(elev) * np.sin(azim)])
cosang = np.clip(d @ sun, -1, 1)
ang = np.arccos(cosang)
img += (ang < np.radians(2.0))[..., None] * np.array([600.0, 560.0, 480.0])
img += np.exp(-(ang / np.radians(12.0)) ** 2)[..., None] * np.array([1.2, 1.0, 0.7])

rows = np.flipud(img).astype('<f4')   # PFM stores the bottom row first
import os
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
with open(out, "wb") as f:
    f.write(b"PF\n%d %d\n-1.0\n" % (W, H))
    f.write(rows.tobytes())
print("wrote", out, W, "x", H, "max value", float(img.max()))
