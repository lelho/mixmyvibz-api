"""Correction de pitch v1 (expérimentale) : détecte la tonalité, aimante les notes sur la gamme.
Méthode : f0 par pYIN, découpage en segments de note stables, décalage par segment (phase vocoder),
retune_ms contrôle la fraction de correction appliquée (0 ms = totale, 60 ms = douce)."""
import numpy as np
import librosa

NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
KS_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KS_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def detect_key(y: np.ndarray, sr: int):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr).mean(axis=1)
    best = (-2.0, "C", "major")
    for r in range(12):
        for prof, mode in ((KS_MAJ, "major"), (KS_MIN, "minor")):
            c = np.corrcoef(chroma, np.roll(prof, r))[0, 1]
            if c > best[0]:
                best = (float(c), NOTES[r], mode)
    return {"key": best[1], "mode": best[2], "confidence": max(0.0, best[0])}


def scale_pitch_classes(key: str, mode: str):
    root = NOTES.index(key)
    steps = [0, 2, 4, 5, 7, 9, 11] if mode == "major" else [0, 2, 3, 5, 7, 8, 10]
    return {(root + s) % 12 for s in steps}


def autotune(y: np.ndarray, sr: int, key: str, mode: str, retune_ms: float = 30.0) -> np.ndarray:
    """y mono float32. Retourne y corrigé (même longueur)."""
    hop = 512
    f0, voiced, _ = librosa.pyin(y, fmin=70, fmax=900, sr=sr, frame_length=2048, hop_length=hop)
    pcs = scale_pitch_classes(key, mode)
    strength = float(np.clip(1.0 - retune_ms / 80.0, 0.15, 1.0))
    shifts = np.zeros(len(f0))
    for i, f in enumerate(f0):
        if not voiced[i] or np.isnan(f):
            continue
        midi = 69 + 12 * np.log2(f / 440.0)
        cands = [m for m in range(int(midi) - 2, int(midi) + 3) if m % 12 in pcs]
        target = min(cands, key=lambda m: abs(m - midi))
        shifts[i] = (target - midi) * strength
    # lissage : une valeur par segment de ~80 ms pour éviter le chorus de micro-décalages
    seg = max(int(0.08 * sr / hop), 8)  # au moins ~4096 échantillons par morceau (sinon le phase vocoder sort des NaN)
    out = np.copy(y)
    fade = int(0.01 * sr)
    for s in range(0, len(shifts), seg):
        chunk = shifts[s:s + seg]
        v = chunk[chunk != 0]
        if len(v) < seg // 2:
            continue
        n = float(np.median(v))
        if abs(n) < 0.05:
            continue
        a, b = s * hop, min((s + seg) * hop, len(y))
        a0, b0 = max(a - fade, 0), min(b + fade, len(y))
        try:
            piece = np.nan_to_num(librosa.effects.pitch_shift(y[a0:b0], sr=sr, n_steps=n))
        except Exception:
            continue
        w = np.ones(b0 - a0)
        w[:fade] = np.linspace(0, 1, min(fade, len(w)))
        w[-fade:] = np.linspace(1, 0, min(fade, len(w)))
        out[a0:b0] = out[a0:b0] * (1 - w) + piece[: b0 - a0] * w
    return out.astype(np.float32)


def detect_key_light(y: np.ndarray, sr: int):
    """Tonalité sur 40 s au milieu du morceau, à 22 kHz : rapide et sobre en mémoire."""
    n = int(40 * sr)
    if len(y) > n:
        a = (len(y) - n) // 2
        y = y[a:a + n]
    if sr > 22050:
        y = librosa.resample(y, orig_sr=sr, target_sr=22050)
        sr = 22050
    return detect_key(y, sr)


def autotune_long(y: np.ndarray, sr: int, key: str, mode: str, retune_ms: float = 30.0, chunk_s: float = 12.0, overlap_s: float = 0.5) -> np.ndarray:
    """Même correction, mais par morceaux de 12 s (fondu de 0,5 s entre les morceaux) : la mémoire ne dépend plus de la durée."""
    n, c, o = len(y), int(chunk_s * sr), int(overlap_s * sr)
    if n <= c + o:
        return autotune(y, sr, key, mode, retune_ms)
    out = np.zeros(n, dtype=np.float32)
    wsum = np.zeros(n, dtype=np.float32)
    s = 0
    while s < n:
        a, b = max(s - o, 0), min(s + c + o, n)
        piece = autotune(y[a:b], sr, key, mode, retune_ms)
        w = np.ones(b - a, dtype=np.float32)
        if a > 0:
            w[:min(2 * o, len(w))] = np.linspace(0, 1, min(2 * o, len(w)), dtype=np.float32)
        if b < n:
            w[-min(2 * o, len(w)):] = np.linspace(1, 0, min(2 * o, len(w)), dtype=np.float32)
        out[a:b] += piece[: b - a] * w
        wsum[a:b] += w
        s += c
    return (out / np.maximum(wsum, 1e-6)).astype(np.float32)
