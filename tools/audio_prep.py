"""Audio prep: cut the supplied track at 1:30, level the quiet middle, build the beat map.

Input : the uploaded .m4a (path as argv[1])
Output: assets/audio/track_90s.wav (lossless working copy, git-ignored)
        assets/audio/beats.json     (beats, onsets and 30 fps audio features for the renderer)

The track has three very different sections:
    A   0.00 -20.22   loud, ~139 BPM
        20.22-20.91   total silence (comedic pause)
    B  20.91 -79.20   same kind of music but ~20 dB quieter, ~130 BPM
        79.20-79.78   total silence (anticipation)
    C  79.78 -90.00   medium loud, ~140 BPM
Gain is automated inside the two silences so nothing audibly pumps.
"""
import json
import os
import subprocess
import sys

import librosa
import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "assets", "audio")
os.makedirs(OUT, exist_ok=True)

DUR = 90.0
SR = 44100
FPS = 30
SIL1 = (20.22, 20.91)
SIL2 = (79.20, 79.78)
GAIN_DB = {"A": -1.5, "B": 14.0, "C": 9.0}
FADE_OUT = 0.9


def smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def gain_curve(n):
    t = np.arange(n) / SR
    g = np.full(n, GAIN_DB["A"])
    g = g + (GAIN_DB["B"] - GAIN_DB["A"]) * smoothstep((t - (SIL1[0] + 0.05)) / (SIL1[1] - SIL1[0] - 0.12))
    g = g + (GAIN_DB["C"] - GAIN_DB["B"]) * smoothstep((t - (SIL2[0] + 0.05)) / (SIL2[1] - SIL2[0] - 0.12))
    lin = 10 ** (g / 20)
    lin *= 1 - smoothstep((t - (DUR - FADE_OUT)) / FADE_OUT)
    return lin


def main(src):
    tmp = os.path.join(OUT, "_cut.wav")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-t", str(DUR), "-ac", "2",
                    "-ar", str(SR), tmp], check=True)
    x, sr = sf.read(tmp, always_2d=True)
    assert sr == SR
    x = x[: int(DUR * SR)]
    x = x * gain_curve(len(x))[:, None]
    # soft limiter: transparent below 0.8, tanh knee above
    k = 0.8
    ax = np.abs(x)
    x = np.where(ax > k, np.sign(x) * (k + (1 - k) * np.tanh((ax - k) / (1 - k))), x)
    sf.write(os.path.join(OUT, "track_90s.wav"), x, SR, subtype="PCM_16")
    os.remove(tmp)

    # ---- analysis on mono ----
    y = x.mean(1)
    y22 = librosa.resample(y, orig_sr=SR, target_sr=22050)
    sr2, hop = 22050, 220
    oenv = librosa.onset.onset_strength(y=y22, sr=sr2, hop_length=hop, aggregate=np.median, fmax=8000)
    tt = np.arange(len(oenv)) * hop / sr2

    beats = {}
    for name, (a, b) in {"A": (0.0, SIL1[0]), "B": (SIL1[1], SIL2[0]), "C": (SIL2[1], DUR)}.items():
        m = (tt >= a) & (tt < b)
        sbpm = {"A": 139, "B": 130, "C": 140}[name]
        _, bt = librosa.beat.beat_track(onset_envelope=oenv[m], sr=sr2, hop_length=hop, units="time",
                                        start_bpm=sbpm, tightness=200)
        beats[name] = [round(float(v + a), 3) for v in bt]

    # onsets with strength (all sections)
    pk = librosa.util.peak_pick(oenv, pre_max=6, post_max=6, pre_avg=14, post_avg=14,
                                delta=float(oenv.std()) * 0.35, wait=8)
    strength = oenv[pk] / np.percentile(oenv, 99.5)
    # normalise strength per section so the quiet part is usable too
    ons = []
    for i, f in enumerate(pk):
        ons.append([round(float(tt[f]), 3), round(float(min(1.5, strength[i])), 3)])

    # ---- 30 fps features ----
    n_fr = int(DUR * FPS)
    n_fft = 2048
    stft = np.abs(librosa.stft(y22, n_fft=n_fft, hop_length=sr2 // FPS))
    fr = librosa.fft_frequencies(sr=sr2, n_fft=n_fft)
    bands = {
        "low": stft[(fr >= 30) & (fr < 180)].sum(0),
        "mid": stft[(fr >= 180) & (fr < 2500)].sum(0),
        "high": stft[(fr >= 2500) & (fr < 10000)].sum(0),
    }
    sec_masks = {"A": (0, SIL1[0]), "B": (SIL1[1], SIL2[0]), "C": (SIL2[1], DUR)}
    feat = {}
    for k, v in bands.items():
        v = v[:n_fr]
        if len(v) < n_fr:
            v = np.pad(v, (0, n_fr - len(v)))
        out = np.zeros(n_fr)
        for a, b in sec_masks.values():
            i0, i1 = int(a * FPS), int(b * FPS)
            seg = v[i0:i1]
            out[i0:i1] = seg / (np.percentile(seg, 97) + 1e-9)
        feat[k] = [round(float(min(1.4, o)), 3) for o in out]
    json.dump({"fps": FPS, "dur": DUR, "sil1": SIL1, "sil2": SIL2, "beats": beats, "onsets": ons, "feat": feat},
              open(os.path.join(OUT, "beats.json"), "w"))
    for k, v in beats.items():
        print(k, len(v), "beats", v[:6], "...")
    print("onsets", len(ons))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "assets", "src", "music_original.m4a"))
