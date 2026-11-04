"""Recover camera poses for a kept capture with COLMAP (pycolmap, CPU).

Works on the frames saved by extract_frames.py, so the original video is not needed.
Takes every `--step`-th kept frame, optionally drops a time range, downsizes by `--scale`,
and runs SIFT + sequential matching + incremental mapping.

Lesson from the first globe capture: a shared OPENCV camera (8 parameters) collapsed after the
first image pair and only 19/219 frames registered. SIMPLE_RADIAL with a focal-length prior
registered 182/219. The prior is 1.432 x image width (the iPhone 17 main lens as recorded by
Blackmagic Camera at 2160x1214); COLMAP refines it.

usage: python colmap_poses.py <capture_dir> [--step 2] [--scale 0.5] [--drop 122:146] [--tag half]
writes <capture_dir>/colmap_<tag>/{images,database.db,sparse/N,centres.csv,camera_path.png,summary.json}
"""
import argparse, csv, json, os, shutil, time

import cv2
import numpy as np
import pycolmap

ap = argparse.ArgumentParser()
ap.add_argument("capture")
ap.add_argument("--step", type=int, default=2)
ap.add_argument("--scale", type=float, default=0.5)
ap.add_argument("--drop", action="append", default=[], help="time range in seconds to leave out, e.g. 122:146")
ap.add_argument("--focal-ratio", type=float, default=1.432)
ap.add_argument("--overlap", type=int, default=15)
ap.add_argument("--tag", default="half")
a = ap.parse_args()

src = os.path.join(a.capture, "frames")
out = os.path.join(a.capture, f"colmap_{a.tag}")
img_dir = os.path.join(out, "images")
if os.path.exists(out):
    shutil.rmtree(out)
os.makedirs(img_dir)

fps = json.load(open(os.path.join(a.capture, "meta.json"))).get("fps", 30.0)
drops = [tuple(float(x) for x in d.split(":")) for d in a.drop]
names = sorted(os.listdir(src))
picked = []
for n in names[::a.step]:
    t = int(n[6:11]) / fps
    if any(lo <= t <= hi for lo, hi in drops):
        continue
    picked.append(n)
for n in picked:
    im = cv2.imread(os.path.join(src, n))
    if a.scale != 1:
        im = cv2.resize(im, (round(im.shape[1] * a.scale), round(im.shape[0] * a.scale)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(img_dir, n), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
H, W = im.shape[:2]
print(f"{len(picked)} frames at {W}x{H}", flush=True)

db = os.path.join(out, "database.db")
reader = pycolmap.ImageReaderOptions()
reader.camera_model = "SIMPLE_RADIAL"
reader.camera_params = f"{a.focal_ratio * W:.1f},{W / 2},{H / 2},0"
t0 = time.time()
pycolmap.extract_features(db, img_dir, camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader)
print(f"features {time.time() - t0:.0f}s", flush=True)
pair = pycolmap.SequentialPairingOptions()
pair.overlap = a.overlap
pair.quadratic_overlap = True
t0 = time.time()
pycolmap.match_sequential(db, pairing_options=pair)
print(f"matching {time.time() - t0:.0f}s", flush=True)
sparse = os.path.join(out, "sparse")
os.makedirs(sparse)
opts = pycolmap.IncrementalPipelineOptions()
opts.ba_refine_principal_point = False
t0 = time.time()
recs = pycolmap.incremental_mapping(db, img_dir, sparse, options=opts)
print(f"mapping {time.time() - t0:.0f}s, {len(recs)} model(s)", flush=True)

summary = {"frames_used": len(picked), "image_size": [W, H], "drop": a.drop, "models": []}
for k, rec in sorted(recs.items(), key=lambda kv: -kv[1].num_reg_images()):
    cam = next(iter(rec.cameras.values()))
    info = {"id": k, "registered": rec.num_reg_images(), "points": rec.num_points3D(),
            "reproj_px": round(rec.compute_mean_reprojection_error(), 3),
            "camera": [cam.model.name, *np.round(cam.params, 3).tolist()]}
    summary["models"].append(info)
    print(info, flush=True)

if recs:
    best_id, best = max(recs.items(), key=lambda kv: kv[1].num_reg_images())
    summary["best_model"] = best_id
    reg = {img.name for img in best.images.values()}
    missing = [n for n in picked if n not in reg]
    summary["unregistered_times_s"] = [round(int(n[6:11]) / fps, 1) for n in missing]
    rows = sorted(((img.name, *img.projection_center()) for img in best.images.values()))
    with open(os.path.join(out, "centres.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["name", "x", "y", "z"])
        wr.writerows(rows)

    # top and side view of the camera path, blue = start, red = end
    P = np.array([r[1:] for r in rows], float)
    Q = (P - P.mean(0)) @ np.linalg.svd(P - P.mean(0))[2].T
    img = np.full((440, 900, 3), 255, np.uint8)
    for x0, (xs, ys), title in ((0, (Q[:, 0], Q[:, 1]), "top view"), (450, (Q[:, 0], Q[:, 2]), "side view (height)")):
        s = min(390 / (np.ptp(xs) or 1), 370 / (np.ptp(ys) or 1))
        for k2, (x, y) in enumerate(zip(xs, ys)):
            t = k2 / max(1, len(xs) - 1)
            cv2.circle(img, (int(x0 + 30 + (x - xs.min()) * s), int(410 - (y - ys.min()) * s)), 4,
                       (int(255 * (1 - t)), 60, int(255 * t)), -1)
        cv2.putText(img, title, (x0 + 30, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, f"{len(P)}/{len(picked)} cameras; blue = start, red = end", (30, 434),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(out, "camera_path.png"), img)

with open(os.path.join(out, "summary.json"), "w") as fh:
    json.dump(summary, fh, indent=1)
print("unregistered at (s):", summary.get("unregistered_times_s"))
