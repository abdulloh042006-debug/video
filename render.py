"""Render the film.

    python render.py                      # the short edit  -> out/zootopia_intro.mp4
    python render.py --long               # the full 1:30 film (no edit)  -> out/zootopia_intro_long.mp4

Frames are rendered in parallel chunks, each piped into its own x264 encoder, then concatenated and muxed with
the soundtrack (the edit splices the audio on the beat, see edit.py).
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

_state = {}


def _init(long_mode):
    if long_mode:
        import director
        _state["world"] = director.build()
        _state["frame"] = lambda f: _state["world"].render(f / FPS)
    else:
        import edit
        ed = edit.Edit()
        _state["frame"] = ed.frame


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
        p.stdin.write(np.ascontiguousarray(_state["frame"](f)).tobytes())
    p.stdin.close()
    p.wait()
    return idx, f1 - f0, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--long", action="store_true", help="render the full 90 s film instead of the short edit")
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--crf", type=int, default=22)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--chunk", type=int, default=60, help="frames per chunk")
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()

    out = a.out or os.path.join(ROOT, "out", "zootopia_intro_long.mp4" if a.long else "zootopia_intro.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = os.path.join(os.path.dirname(out), "_parts_" + os.path.basename(out).split(".")[0])
    os.makedirs(tmp, exist_ok=True)
    wav = os.path.join(tmp, "audio.wav")
    if a.long:
        total = int(round(DUR * FPS))
        audio = os.path.join(ROOT, "assets", "audio", "track_90s.wav")
    else:
        import soundfile as sf
        import edit
        ed = edit.Edit()
        total = ed.total
        sf.write(wav, ed.audio(), edit.SR, subtype="PCM_16")
        audio = wav
        print("edit pieces:", [(p["name"], p["n"]) for p in ed.pieces], flush=True)
    jobs = []
    for i, f0 in enumerate(range(0, total, a.chunk)):
        jobs.append((i, f0, min(f0 + a.chunk, total), os.path.join(tmp, f"part_{i:03d}.mp4"), a.crf, a.scale))
    print(f"{total} frames ({total / FPS:.2f} s) in {len(jobs)} chunks, {a.workers} workers", flush=True)
    t0 = time.time()
    done = 0
    with mp.Pool(a.workers, initializer=_init, initargs=(a.long,)) as pool:
        for idx, n, dt in pool.imap_unordered(_chunk, jobs):
            done += n
            el = time.time() - t0
            print(f"chunk {idx:3d} done ({n} fr, {dt:.0f}s)  {done}/{total}  elapsed {el:.0f}s  eta {el / done * (total - done):.0f}s",
                  flush=True)
    lst = os.path.join(tmp, "list.txt")
    with open(lst, "w") as fh:
        for j in jobs:
            fh.write(f"file '{os.path.abspath(j[3])}'\n")
    silent = os.path.join(tmp, "silent.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", silent], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", silent, "-i", audio, "-t", f"{total / FPS}",
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", out],
                   check=True)
    if not a.keep:
        for f in os.listdir(tmp):
            os.remove(os.path.join(tmp, f))
        os.rmdir(tmp)
    print("wrote", out, f"{time.time() - t0:.0f}s total", flush=True)


if __name__ == "__main__":
    main()
