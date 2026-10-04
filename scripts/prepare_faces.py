"""Download the public LFW dataset (via scikit-learn) and write 40 sample faces to data/faces."""
import os
import numpy as np, cv2
from sklearn.datasets import fetch_lfw_people

os.makedirs("data/faces", exist_ok=True)
lfw = fetch_lfw_people(min_faces_per_person=20, resize=1.0, color=True)
imgs = lfw.images
if imgs.max() <= 1.0:            # depending on the version, values are in [0,1] or [0,255]
    imgs = imgs * 255
step = max(1, len(imgs) // 40)   # evenly spaced samples for more variety
sample = imgs[::step][:40]
for i, img in enumerate(sample):
    img = np.clip(img, 0, 255).astype("uint8")
    img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)  # upscale to help detection
    cv2.imwrite(f"data/faces/face_{i:03d}.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
print("Wrote", len(sample), "face images")
