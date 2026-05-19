"""
Classical Audio Feature Extraction - Full Quality (Refactored)
All handcrafted features for highest possible quality.
"""

from typing import Dict, List, Optional, Union

import numpy as np
import librosa
import torch
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================
#  Constants
# =============================================================
DEFAULT_HOP_LENGTH = 512
DEFAULT_FRAME_LENGTH = 2048
DEFAULT_N_MFCC = 13
MIN_AUDIO_SECONDS = 0.1
SILENCE_PERCENTILE = 20  # for adaptive silence threshold
MIN_PAUSE_DURATION = 0.05  # seconds


# =============================================================
#  Main API
# =============================================================
def extract_classical_features(
    waveform: Union[np.ndarray, torch.Tensor],
    sr: int = 16000,
    hop_length: int = DEFAULT_HOP_LENGTH,
    frame_length: int = DEFAULT_FRAME_LENGTH,
    n_mfcc: int = DEFAULT_N_MFCC,
    preemphasis: bool = True,
    normalize: bool = True,
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Extract all classical audio features from a waveform.

    Args:
        waveform: Audio as tensor or numpy array (1D)
        sr: Sample rate
        hop_length: Hop length for frame-based features
        frame_length: Frame length / n_fft for spectral features
        n_mfcc: Number of MFCC coefficients
        preemphasis: Apply pre-emphasis filter
        normalize: Peak-normalize the waveform

    Returns:
        Dict of features. Each value is either a scalar (float) or 1D ndarray.
    """
    # --- Prepare waveform ---
    y = _prepare_waveform(waveform, sr, preemphasis, normalize)

    features: Dict[str, Union[float, np.ndarray]] = {}

    # Compute STFT once and reuse
    stft = librosa.stft(y, n_fft=frame_length, hop_length=hop_length)
    S_power = np.abs(stft) ** 2

    # Each block is wrapped in try/except so one failure doesn't kill the rest
    _safe_update(features, _extract_mfcc, y, sr, hop_length, frame_length, n_mfcc)
    f0_clean, f0_full = _safe_pitch(features, y, sr, hop_length, frame_length)
    rms = _safe_energy(features, y, hop_length, frame_length)
    _safe_update(features, _extract_spectral, y, sr, S_power, hop_length)
    _safe_update(features, _extract_zcr, y, hop_length, frame_length)
    chroma = _safe_chroma(features, S_power, sr)
    _safe_update(features, _extract_tonnetz, chroma)
    _safe_update(features, _extract_tempo, y, sr, hop_length)
    _safe_update(features, _extract_voice_quality, y, sr, f0_clean,
                 frame_length, hop_length)
    _safe_update(features, _extract_pause_features, rms, sr, hop_length)

    return features


# =============================================================
#  Preparation
# =============================================================
def _prepare_waveform(
    waveform: Union[np.ndarray, torch.Tensor],
    sr: int,
    preemphasis: bool,
    normalize: bool,
) -> np.ndarray:
    """Convert to 1D float32 numpy array, validate, and optionally process."""
    if isinstance(waveform, torch.Tensor):
        y = waveform.detach().cpu().numpy()
    else:
        y = np.asarray(waveform, dtype=np.float32)

    y = np.squeeze(y).astype(np.float32, copy=False)

    if y.ndim != 1:
        raise ValueError(f"Expected 1D waveform, got shape {y.shape}")

    if len(y) < sr * MIN_AUDIO_SECONDS:
        raise ValueError(f"Audio too short: {len(y)/sr:.3f}s (min {MIN_AUDIO_SECONDS}s)")

    if normalize:
        peak = np.max(np.abs(y))
        if peak > 0:
            y = y / peak

    if preemphasis:
        y = librosa.effects.preemphasis(y)

    return y


def _safe_update(features: dict, fn, *args, **kwargs) -> None:
    """Run a feature-extraction function and merge its dict into features."""
    try:
        result = fn(*args, **kwargs)
        if result:
            features.update(result)
    except Exception as e:
        warnings.warn(f"{fn.__name__} failed: {e}")


# =============================================================
#  1. MFCC + Deltas
# =============================================================
def _extract_mfcc(
    y: np.ndarray, sr: int, hop_length: int, frame_length: int, n_mfcc: int,
) -> Dict[str, np.ndarray]:
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, hop_length=hop_length, n_fft=frame_length,
    )
    log_mel = librosa.power_to_db(mel_spec)
    mfcc = librosa.feature.mfcc(S=log_mel, n_mfcc=n_mfcc)

    out = {
        "mfcc_mean": np.mean(mfcc, axis=1),
        "mfcc_std": np.std(mfcc, axis=1),
    }

    # Delta and Delta-Delta need at least width=9 frames; fall back if shorter
    width = min(9, mfcc.shape[1] if mfcc.shape[1] % 2 == 1 else mfcc.shape[1] - 1)
    if width >= 3:
        delta = librosa.feature.delta(mfcc, width=width)
        delta2 = librosa.feature.delta(mfcc, order=2, width=width)
        out["delta_mfcc_mean"] = np.mean(delta, axis=1)
        out["delta_mfcc_std"] = np.std(delta, axis=1)
        out["delta2_mfcc_mean"] = np.mean(delta2, axis=1)
        out["delta2_mfcc_std"] = np.std(delta2, axis=1)
    else:
        zeros = np.zeros(n_mfcc, dtype=np.float32)
        out["delta_mfcc_mean"] = zeros.copy()
        out["delta_mfcc_std"] = zeros.copy()
        out["delta2_mfcc_mean"] = zeros.copy()
        out["delta2_mfcc_std"] = zeros.copy()

    return out


# =============================================================
#  2. Pitch
# =============================================================
def _safe_pitch(features: dict, y, sr, hop_length, frame_length):
    """Returns (f0_clean, f0_full) for downstream use."""
    try:
        f0, _, _ = librosa.pyin(
            y, fmin=50, fmax=500, sr=sr,
            hop_length=hop_length, frame_length=frame_length,
        )
        f0_clean = f0[~np.isnan(f0)]
        features.update(_pitch_stats(f0, f0_clean))
        return f0_clean, f0
    except Exception as e:
        warnings.warn(f"pitch extraction failed: {e}")
        features.update(_empty_pitch_stats())
        return np.array([]), np.array([])


def _pitch_stats(f0: np.ndarray, f0_clean: np.ndarray) -> Dict[str, float]:
    if len(f0_clean) == 0:
        return _empty_pitch_stats()

    voiced_ratio = (
        np.sum(~np.isnan(f0)) / len(f0) if len(f0) > 0 else 0.0
    )

    return {
        "pitch_mean": float(np.mean(f0_clean)),
        "pitch_std": float(np.std(f0_clean)),
        "pitch_min": float(np.min(f0_clean)),
        "pitch_max": float(np.max(f0_clean)),
        "pitch_range": float(np.ptp(f0_clean)),
        "pitch_median": float(np.median(f0_clean)),
        "pitch_skew": _safe_skew(f0_clean),
        "pitch_kurtosis": _safe_kurtosis(f0_clean),
        "pitch_slope": _compute_slope(f0_clean),
        "voiced_ratio": float(voiced_ratio),
    }


def _empty_pitch_stats() -> Dict[str, float]:
    return {
        "pitch_mean": 0.0, "pitch_std": 0.0, "pitch_min": 0.0,
        "pitch_max": 0.0, "pitch_range": 0.0, "pitch_median": 0.0,
        "pitch_skew": 0.0, "pitch_kurtosis": 0.0, "pitch_slope": 0.0,
        "voiced_ratio": 0.0,
    }


# =============================================================
#  3. Energy
# =============================================================
def _safe_energy(features, y, hop_length, frame_length) -> np.ndarray:
    rms = librosa.feature.rms(
        y=y, frame_length=frame_length, hop_length=hop_length,
    )[0]
    features.update({
        "energy_mean": float(np.mean(rms)),
        "energy_std": float(np.std(rms)),
        "energy_min": float(np.min(rms)),
        "energy_max": float(np.max(rms)),
        "energy_range": float(np.ptp(rms)),
        "energy_skew": _safe_skew(rms),
        "energy_kurtosis": _safe_kurtosis(rms),
        "energy_slope": _compute_slope(rms),
    })
    return rms


# =============================================================
#  4. Spectral Features
# =============================================================
def _extract_spectral(
    y: np.ndarray, sr: int, S_power: np.ndarray, hop_length: int,
) -> Dict[str, Union[float, np.ndarray]]:
    # Use precomputed power spectrogram for efficiency
    S_mag = np.sqrt(S_power)

    spec_cent = librosa.feature.spectral_centroid(S=S_mag, sr=sr)[0]
    spec_bw = librosa.feature.spectral_bandwidth(S=S_mag, sr=sr)[0]
    spec_rolloff = librosa.feature.spectral_rolloff(S=S_mag, sr=sr)[0]
    spec_flat = librosa.feature.spectral_flatness(S=S_mag)[0]
    spec_contrast = librosa.feature.spectral_contrast(S=S_mag, sr=sr)

    return {
        "spectral_centroid_mean": float(np.mean(spec_cent)),
        "spectral_centroid_std": float(np.std(spec_cent)),
        "spectral_bandwidth_mean": float(np.mean(spec_bw)),
        "spectral_bandwidth_std": float(np.std(spec_bw)),
        "spectral_rolloff_mean": float(np.mean(spec_rolloff)),
        "spectral_rolloff_std": float(np.std(spec_rolloff)),
        "spectral_flatness_mean": float(np.mean(spec_flat)),
        "spectral_flatness_std": float(np.std(spec_flat)),
        "spectral_contrast_mean": np.mean(spec_contrast, axis=1),
        "spectral_contrast_std": np.std(spec_contrast, axis=1),
    }


# =============================================================
#  5. ZCR
# =============================================================
def _extract_zcr(
    y: np.ndarray, hop_length: int, frame_length: int,
) -> Dict[str, float]:
    zcr = librosa.feature.zero_crossing_rate(
        y, frame_length=frame_length, hop_length=hop_length,
    )[0]
    return {
        "zcr_mean": float(np.mean(zcr)),
        "zcr_std": float(np.std(zcr)),
    }


# =============================================================
#  6. Chroma
# =============================================================
def _safe_chroma(features, S_power, sr) -> Optional[np.ndarray]:
    try:
        chroma = librosa.feature.chroma_stft(S=S_power, sr=sr)
        features["chroma_mean"] = np.mean(chroma, axis=1)
        features["chroma_std"] = np.std(chroma, axis=1)
        return chroma
    except Exception as e:
        warnings.warn(f"chroma failed: {e}")
        features["chroma_mean"] = np.zeros(12, dtype=np.float32)
        features["chroma_std"] = np.zeros(12, dtype=np.float32)
        return None


# =============================================================
#  7. Tonnetz (uses chroma directly — no expensive HPSS)
# =============================================================
def _extract_tonnetz(chroma: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
    if chroma is None:
        return {
            "tonnetz_mean": np.zeros(6, dtype=np.float32),
            "tonnetz_std": np.zeros(6, dtype=np.float32),
        }
    tonnetz = librosa.feature.tonnetz(chroma=chroma)
    return {
        "tonnetz_mean": np.mean(tonnetz, axis=1),
        "tonnetz_std": np.std(tonnetz, axis=1),
    }


# =============================================================
#  8. Tempo / Speech Rate
# =============================================================
def _extract_tempo(
    y: np.ndarray, sr: int, hop_length: int,
) -> Dict[str, float]:
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    # Handle librosa version differences for tempo
    try:
        tempo = librosa.feature.rhythm.tempo(
            onset_envelope=onset_env, sr=sr, hop_length=hop_length,
        )
    except AttributeError:
        tempo = librosa.beat.tempo(
            onset_envelope=onset_env, sr=sr, hop_length=hop_length,
        )

    onsets = librosa.onset.onset_detect(
        y=y, sr=sr, onset_envelope=onset_env, hop_length=hop_length,
    )
    duration = len(y) / sr

    return {
        "tempo": float(np.atleast_1d(tempo)[0]),
        "onset_rate": float(len(onsets) / duration) if duration > 0 else 0.0,
    }


# =============================================================
#  9. Voice Quality: Jitter, Shimmer, HNR
# =============================================================
def _extract_voice_quality(
    y: np.ndarray, sr: int, f0_clean: np.ndarray,
    frame_length: int, hop_length: int,
) -> Dict[str, float]:
    return {
        "jitter": _compute_jitter(f0_clean),
        "shimmer": _compute_shimmer(y, sr, frame_length, hop_length),
        "hnr": _compute_hnr(y, sr),
    }


def _compute_jitter(f0_clean: np.ndarray) -> float:
    """
    Frame-level pitch period variability (proxy for true Praat jitter).
    Higher = unstable voice (stress / fear).
    """
    if len(f0_clean) < 2:
        return 0.0
    valid = f0_clean[f0_clean > 0]
    if len(valid) < 2:
        return 0.0
    periods = 1.0 / valid
    diffs = np.abs(np.diff(periods))
    mean_period = np.mean(periods)
    if mean_period == 0:
        return 0.0
    return float(np.mean(diffs) / mean_period)


def _compute_shimmer(
    y: np.ndarray, sr: int, frame_length: int, hop_length: int,
) -> float:
    """Amplitude variability between consecutive voiced frames."""
    rms = librosa.feature.rms(
        y=y, frame_length=frame_length, hop_length=hop_length,
    )[0]
    if len(rms) < 2:
        return 0.0

    # Adaptive voiced threshold: use percentile-based silence cut
    threshold = max(np.mean(rms) * 0.1, np.percentile(rms, SILENCE_PERCENTILE))
    voiced_rms = rms[rms > threshold]
    if len(voiced_rms) < 2:
        return 0.0

    diffs = np.abs(np.diff(voiced_rms))
    mean_amp = np.mean(voiced_rms)
    if mean_amp == 0:
        return 0.0
    return float(np.mean(diffs) / mean_amp)


def _compute_hnr(y: np.ndarray, sr: int) -> float:
    """
    Harmonics-to-Noise Ratio via FFT-based autocorrelation.
    High (>20 dB) = clean voice. Low (<10 dB) = noisy/strained.
    """
    frame_length = int(0.04 * sr)   # 40 ms
    hop_length = int(0.01 * sr)     # 10 ms

    if len(y) < frame_length:
        return 0.0

    min_lag = max(int(sr / 500), 1)
    max_lag = int(sr / 50)

    hnr_values: List[float] = []

    for start in range(0, len(y) - frame_length, hop_length):
        frame = y[start:start + frame_length]

        # FFT-based autocorrelation (much faster than np.correlate)
        n = 2 * frame_length
        fft = np.fft.rfft(frame, n=n)
        corr = np.fft.irfft(fft * np.conj(fft), n=n)[:frame_length]

        if corr[0] <= 0:
            continue
        corr = corr / corr[0]

        if max_lag >= len(corr):
            continue
        if min_lag >= max_lag:
            continue

        peak_val = float(np.max(corr[min_lag:max_lag]))
        # Clip to a safe (0, 1) range to avoid log(0) or log of negative
        peak_val = float(np.clip(peak_val, 1e-6, 1.0 - 1e-6))
        hnr_values.append(10.0 * np.log10(peak_val / (1.0 - peak_val)))

    return float(np.mean(hnr_values)) if hnr_values else 0.0


# =============================================================
#  10. Pause Features
# =============================================================
def _extract_pause_features(
    rms: np.ndarray, sr: int, hop_length: int,
) -> Dict[str, float]:
    if len(rms) == 0:
        return _empty_pause_stats()

    # Robust adaptive threshold
    threshold = max(np.mean(rms) * 0.1, np.percentile(rms, SILENCE_PERCENTILE))
    is_silence = rms < threshold

    pauses: List[float] = []
    current = 0
    for silent in is_silence:
        if silent:
            current += 1
        else:
            if current > 0:
                dur = current * hop_length / sr
                if dur > MIN_PAUSE_DURATION:
                    pauses.append(dur)
            current = 0
    # Trailing pause
    if current > 0:
        dur = current * hop_length / sr
        if dur > MIN_PAUSE_DURATION:
            pauses.append(dur)

    total_duration = len(rms) * hop_length / sr
    if not pauses or total_duration <= 0:
        return _empty_pause_stats()

    return {
        "pause_count": float(len(pauses)),
        "pause_rate": float(len(pauses) / total_duration),
        "pause_mean_duration": float(np.mean(pauses)),
        "pause_max_duration": float(np.max(pauses)),
        "pause_total_ratio": float(np.sum(pauses) / total_duration),
    }


def _empty_pause_stats() -> Dict[str, float]:
    return {
        "pause_count": 0.0, "pause_rate": 0.0,
        "pause_mean_duration": 0.0, "pause_max_duration": 0.0,
        "pause_total_ratio": 0.0,
    }


# =============================================================
#  Statistical Helpers
# =============================================================
def _safe_skew(arr: np.ndarray) -> float:
    if len(arr) < 3:
        return 0.0
    mean, std = np.mean(arr), np.std(arr)
    if std == 0:
        return 0.0
    return float(np.mean(((arr - mean) / std) ** 3))


def _safe_kurtosis(arr: np.ndarray) -> float:
    if len(arr) < 4:
        return 0.0
    mean, std = np.mean(arr), np.std(arr)
    if std == 0:
        return 0.0
    return float(np.mean(((arr - mean) / std) ** 4) - 3.0)  # excess kurtosis


def _compute_slope(arr: np.ndarray) -> float:
    if len(arr) < 2:
        return 0.0
    x = np.arange(len(arr), dtype=np.float64)
    coeffs = np.polyfit(x, arr.astype(np.float64), 1)
    return float(coeffs[0])


# =============================================================
#  Flatten / Names — guaranteed aligned
# =============================================================
def flatten_features(features: Dict[str, Union[float, np.ndarray]]) -> np.ndarray:
    """Flatten feature dict to 1D float32 vector (sorted key order)."""
    parts = []
    for key in sorted(features.keys()):
        val = features[key]
        if isinstance(val, np.ndarray):
            parts.append(val.astype(np.float32).ravel())
        else:
            parts.append(np.array([val], dtype=np.float32))
    flat = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    return np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)


def get_feature_names(
    features: Dict[str, Union[float, np.ndarray]],
) -> List[str]:
    """
    Generate names aligned with `flatten_features` output.
    Must be called with a real features dict (or a reference one) so that
    array dimensions are known.
    """
    names: List[str] = []
    for key in sorted(features.keys()):
        val = features[key]
        if isinstance(val, np.ndarray):
            for i in range(val.size):
                names.append(f"{key}_{i}")
        else:
            names.append(key)
    return names

def get_reference_features(sr: int = 16000, duration: float = 1.0) -> dict:
    """
    Generate reference features from a synthetic sine wave.
    Useful for metadata when no real samples have been processed yet.
    """
    import numpy as np
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Mix of frequencies to ensure all features are non-trivial
    y = (0.3 * np.sin(2 * np.pi * 200 * t) +
         0.2 * np.sin(2 * np.pi * 440 * t) +
         0.1 * np.random.randn(len(t))).astype(np.float32)
    return extract_classical_features(y, sr=sr)