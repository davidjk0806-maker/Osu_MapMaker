"""Reading a song, splitting it into instruments, and re-packing it for osu!."""

import glob
import json
import os
import random
import shutil
import subprocess

import librosa
import numpy as np
import soundfile as sf

SR = 22050                    # analysis sample rate
STEMS = ("drums", "bass", "other", "vocals")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.environ.get("MAPMAKER_CACHE") or os.path.join(ROOT, ".cache")


class MapError(Exception):
    """A problem with the input that the person running this can fix."""


def read_audio(path, workdir):
    """Decode an audio file to float32 [frames, channels] at its own sample rate."""
    try:
        return sf.read(path, dtype="float32", always_2d=True)
    except Exception:
        pass
    wav = os.path.join(workdir, "decoded.wav")
    for cmd in (["afconvert", "-f", "WAVE", "-d", "LEI16", path, wav],
                ["ffmpeg", "-v", "error", "-y", "-i", path, wav]):
        if shutil.which(cmd[0]) is None:
            continue
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return sf.read(wav, dtype="float32", always_2d=True)
        except Exception:
            continue
    raise MapError("Couldn't decode that audio file. Try an .mp3, .ogg, .opus, "
                   ".m4a, .wav or .flac.")


def write_ogg(path, data, sr):
    """Vorbis at ~192 kbps. Written in blocks: libsndfile can choke on one huge write."""
    with sf.SoundFile(path, "w", samplerate=sr, channels=data.shape[1],
                      format="OGG", subtype="VORBIS", compression_level=0.4) as fh:
        for i in range(0, len(data), 16384):
            fh.write(data[i:i + 16384])


def decode_lag(path, data, sr, span=300):
    """How many samples the packed audio plays behind the audio we analysed."""
    back, _ = sf.read(path, dtype="float32", always_2d=True)
    a, b = data.mean(axis=1), back.mean(axis=1)
    n = min(len(a), len(b))
    lo = max(0, n // 2 - 5 * sr)
    hi = min(n, lo + 10 * sr)
    if hi - lo <= 4 * span:
        return 0
    corr = np.correlate(b[lo:hi], a[lo + span:hi - span], mode="valid")
    return int(np.argmax(corr)) - span


def pack_audio(data, sr, key):
    """The .ogg that goes into the .osz, made from the very samples that were analysed
    so it plays back in exactly the same place (an .mp3 does not promise that).
    Returns (path, ms the packed audio lags the analysed audio)."""
    path = os.path.join(CACHE, f"audio-{key}.ogg")
    note = path + ".json"
    if os.path.isfile(path) and os.path.isfile(note):
        with open(note) as fh:
            return path, json.load(fh)["lag_ms"]
    os.makedirs(CACHE, exist_ok=True)
    peak = float(np.abs(data).max())
    # keep peaks inside full scale so nothing clips on playback
    write_ogg(path, data * (0.98 / peak) if peak > 0.98 else data, sr)
    lag_ms = decode_lag(path, data, sr) * 1000.0 / sr
    with open(note, "w") as fh:
        json.dump({"lag_ms": lag_ms}, fh)
    return path, lag_ms


def preview_wav(mono, key):
    """A small mono .wav every browser can play, for the in-page chart preview."""
    path = os.path.join(CACHE, f"preview-{key}.wav")
    if not os.path.isfile(path):
        os.makedirs(CACHE, exist_ok=True)
        peak = float(np.abs(mono).max()) or 1.0
        sf.write(path, mono * min(1.0, 0.98 / peak), SR, subtype="PCM_16")
    return path


def separate(data, sr, key, progress=None):
    """Split the song into drums / bass / other / vocals with Demucs (mono, at SR)."""
    path = os.path.join(CACHE, f"stems-{key}.npz")
    if os.path.isfile(path):
        with np.load(path) as z:
            return {k: z[k].astype(np.float32) for k in STEMS}
    # keep the model download inside this project instead of the home folder, and once
    # it is here, load it without asking the network
    models = os.environ.setdefault("HF_HOME", os.path.join(CACHE, "models"))
    os.environ.setdefault("TORCH_HOME", models)
    if glob.glob(os.path.join(models, "hub", "models--*HTDemucs*", "snapshots", "*", "*")):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        import torch
        from demucs.api import Separator
    except ImportError as err:
        raise MapError("Demucs is missing. Run:  pip install -r requirements.txt") from err

    def report(state):
        if progress and state.get("audio_length"):
            progress(min(1.0, state["segment_offset"] / state["audio_length"]))

    random.seed(0)
    torch.manual_seed(0)
    wav = data[:, :2] if data.shape[1] >= 2 else np.repeat(data, 2, axis=1)
    wav = torch.from_numpy(np.ascontiguousarray(wav.T))
    devices = [d for d, ok in (("cuda", torch.cuda.is_available()),
                               ("mps", torch.backends.mps.is_available())) if ok] + ["cpu"]
    for device in devices:
        try:
            sep = Separator(model="htdemucs", device=device, callback=report)
            _, out = sep.separate_tensor(wav, sr)
            break
        except Exception:
            if device == "cpu":
                raise
    # float16 on disk; round-trip the fresh result too so every run sees the same numbers
    stems = {k: librosa.resample(out[k].cpu().numpy().mean(axis=0), orig_sr=sep.samplerate,
                                 target_sr=SR).astype(np.float16) for k in STEMS}
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(path, **stems)
    return {k: v.astype(np.float32) for k, v in stems.items()}
