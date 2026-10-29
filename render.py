"""Render the film.

    python render.py [--start 0] [--end 90] [--out out/zootopia_intro.mp4] [--workers 4] [--crf 17] [--scale 1]

Frames are rendered in parallel chunks, each piped into its own x264 encoder, then concatenated
and muxed with the (1:30-cut) soundtrack.
"""
import argparse
import multiprocessing as mp
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from engine.common import W, H, FPS, DUR  # noqa: E402

_world = None


def _init():
    global _world
    import director
    _world = director.build()


def _chunk(job):
    idx, f0, f1, path, crf, scale = job
    ow, oh = int(W * scale) // 2 * 2, int(H * scale) // 2 * 2
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS),
           "-i", "-"]
    if scale != 1:
        cmd += ["-vf", f"scale={ow}:{oh}:flags=lanczos"]
    cmd += ["-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p",
            "-x264-params", f"keyint={FPS}:min-keyint={FPS}:scenecut=0", "-bf", "2", path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.time()
    for f in range(f0, f1):
        img = _world.render(f / FPS)
        p.stdin.write(np.ascontiguousarray(img).tobytes())
    p.stdin.close()
    p.wait()
    return idx, f1 - f0, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=DUR)
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "zootopia_intro.mp4"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--crf", type=int, default=17)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--chunk", type=int, default=90, help="frames per chunk")
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    tmp = os.path.join(os.path.dirname(a.out), "_parts_" + os.path.basename(a.out).split(".")[0])
    os.makedirs(tmp, exist_ok=True)
    f_start, f_end = int(round(a.start * FPS)), int(round(a.end * FPS))
    jobs = []
    for i, f0 in enumerate(range(f_start, f_end, a.chunk)):
        f1 = min(f0 + a.chunk, f_end)
        jobs.append((i, f0, f1, os.path.join(tmp, f"part_{i:03d}.mp4"), a.crf, a.scale))
    print(f"{f_end - f_start} frames in {len(jobs)} chunks, {a.workers} workers", flush=True)
    t0 = time.time()
    done = 0
    with mp.Pool(a.workers, initializer=_init) as pool:
        for idx, n, dt in pool.imap_unordered(_chunk, jobs):
            done += n
            el = time.time() - t0
            print(f"chunk {idx:3d} done ({n} fr, {dt:.0f}s)  {done}/{f_end - f_start}  elapsed {el:.0f}s  "
                  f"eta {el / done * (f_end - f_start - done):.0f}s", flush=True)
    lst = os.path.join(tmp, "list.txt")
    with open(lst, "w") as fh:
        for j in jobs:
            fh.write(f"file '{os.path.abspath(j[3])}'\n")
    silent = os.path.join(tmp, "silent.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", silent],
                   check=True)
    audio = os.path.join(ROOT, "assets", "audio", "track_90s.wav")
    dur = (f_end - f_start) / FPS
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", silent, "-ss", f"{a.start}", "-t", f"{dur}", "-i", audio,
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
                    a.out], check=True)
    if not a.keep:
        for j in jobs:
            os.remove(j[3])
        os.remove(lst)
        os.remove(silent)
        os.rmdir(tmp)
    print("wrote", a.out, f"{time.time() - t0:.0f}s total", flush=True)


if __name__ == "__main__":
    main()
