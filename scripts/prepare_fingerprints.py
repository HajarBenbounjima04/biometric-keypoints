"""Generate 30 SYNTHETIC fingerprint-like ridge patterns into data/fingerprint."""
import os
import numpy as np, cv2

os.makedirs("data/fingerprint", exist_ok=True)
rng = np.random.default_rng(42)
for i in range(30):
    h = w = 256
    y, x = np.mgrid[0:h, 0:w]
    cx, cy = rng.uniform(100, 156, 2)
    r = np.hypot(x - cx, y - cy)
    theta = np.arctan2(y - cy, x - cx)
    f = rng.uniform(0.25, 0.35)
    img = np.sin(f * r + 2 * np.sin(theta * rng.integers(1, 3)) + rng.uniform(0, 6))
    img = img + 0.3 * rng.standard_normal((h, w))
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    cv2.imwrite(f"data/fingerprint/fp_synth_{i:03d}.png", img)
print("Wrote 30 synthetic fingerprints")
