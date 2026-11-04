"""Keep the frames of a capture video so the video itself can be deleted.

For every window of `--window` frames (default 10, about 3 per second at 30 fps) the sharpest
frame (variance of the Laplacian) is saved at full resolution as a JPEG. Per-frame statistics
and the camera settings Blackmagic Camera wrote into the file are saved next to the frames.
The GPS location stored in the .mov is deliberately not copied.

usage: python extract_frames.py <video.mov> <out_dir> [--window 10] [--quality 95]
"""
import argparse, csv, json, os, re, subprocess, time

import cv2
import imageio_ffmpeg

KEEP_META = ("shutterSpeed", "shutterMode", "camera.iso", "whiteBalanceKelvin", "whiteBalanceTint",
             "camera.aperture", "lensType", "sensorFPS", "creationdate", "software", "model")

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("out")
ap.add_argument("--window", type=int, default=10)
ap.add_argument("--quality", type=int, default=95)
a = ap.parse_args()

frames_dir = os.path.join(a.out, "frames")
os.makedirs(frames_dir, exist_ok=True)

# camera settings from the container metadata, without location
probe = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", a.video],
                       capture_output=True, text=True, errors="replace").stderr
meta = {}
for line in probe.splitlines():
    m = re.match(r"\s+([\w.\-]+)\s*:\s(.*)$", line)
    if m and any(k in m.group(1) for k in KEEP_META) and "location" not in m.group(1):
        meta[m.group(1)] = m.group(2).strip()
for key in ("Duration", "Video:"):
    for line in probe.splitlines():
        if key in line:
            meta[key.strip(":")] = line.strip()
            break
meta["source_file"] = os.path.basename(a.video)
meta["window"] = a.window

# pass 1: sharpness of every frame
t0 = time.time()
cap = cv2.VideoCapture(a.video)
fps = cap.get(cv2.CAP_PROP_FPS)
stats = []
while True:
    ok, f = cap.read()
    if not ok:
        break
    g = cv2.cvtColor(cv2.resize(f, (f.shape[1] // 2, f.shape[0] // 2), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    stats.append((len(stats), len(stats) / fps, cv2.Laplacian(g, cv2.CV_64F).var(), float(g.mean())))
cap.release()
print(f"pass 1: {len(stats)} frames scored in {time.time() - t0:.0f}s", flush=True)

chosen = set()
for s in range(0, len(stats), a.window):
    seg = stats[s:s + a.window]
    chosen.add(max(seg, key=lambda r: r[2])[0])

# pass 2: write the chosen frames at full resolution
t0 = time.time()
cap = cv2.VideoCapture(a.video)
i, written = 0, 0
while True:
    ok, f = cap.read()
    if not ok:
        break
    if i in chosen:
        cv2.imwrite(os.path.join(frames_dir, f"frame_{i:05d}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, a.quality])
        written += 1
        h, w = f.shape[:2]
    i += 1
cap.release()
meta["frame_size"] = [w, h]
meta["fps"] = fps
meta["frames_in_video"] = len(stats)
meta["frames_kept"] = written
print(f"pass 2: {written} frames written in {time.time() - t0:.0f}s", flush=True)

with open(os.path.join(a.out, "frames.csv"), "w", newline="") as fh:
    wr = csv.writer(fh)
    wr.writerow(["frame", "t", "sharp", "luma", "kept"])
    for r in stats:
        wr.writerow([r[0], f"{r[1]:.3f}", f"{r[2]:.1f}", f"{r[3]:.1f}", int(r[0] in chosen)])
with open(os.path.join(a.out, "meta.json"), "w", encoding="utf-8") as fh:
    json.dump(meta, fh, indent=1, ensure_ascii=False)
print(json.dumps(meta, indent=1))
