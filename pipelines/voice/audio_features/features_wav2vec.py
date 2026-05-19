"""
Wav2Vec 2.0 Deep Feature Extraction
Returns 1536-dim features (768 mean + 768 std) from last hidden state.

Provides both single-sample and batched APIs:
  - extract_wav2vec_features(waveform)          -> single sample
  - extract_wav2vec_features_batch(waveforms)   -> list of samples (faster)
"""

from typing import List, Union

import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model


_PROCESSOR = None
_MODEL = None
_DEVICE = None
_LOADED_NAME = None

MAX_SECONDS = 20.0  # cap to prevent OOM on long clips


# ----------------------------------------------------------------------
# Model loading (singleton)
# ----------------------------------------------------------------------
def _get_wav2vec(model_name="facebook/wav2vec2-base"):
    global _PROCESSOR, _MODEL, _DEVICE, _LOADED_NAME
    if _MODEL is None:
        _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        _PROCESSOR = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        _MODEL = Wav2Vec2Model.from_pretrained(model_name).to(_DEVICE)
        _MODEL.eval()
        _LOADED_NAME = model_name
    elif _LOADED_NAME != model_name:
        raise RuntimeError(
            f"Wav2Vec already loaded as '{_LOADED_NAME}', "
            f"requested '{model_name}'"
        )
    return _PROCESSOR, _MODEL, _DEVICE


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _to_numpy_1d(waveform) -> np.ndarray:
    """Convert tensor/array to 1D float32 numpy array."""
    if isinstance(waveform, torch.Tensor):
        y = waveform.detach().cpu().float().numpy()
    else:
        y = np.asarray(waveform, dtype=np.float32)
    y = np.squeeze(y)
    if y.ndim != 1:
        raise ValueError(f"Expected 1D waveform, got shape {y.shape}")
    return y.astype(np.float32, copy=False)


def _truncate(y: np.ndarray, sr: int) -> np.ndarray:
    """Center-crop to MAX_SECONDS to bound memory."""
    max_samples = int(MAX_SECONDS * sr)
    if y.shape[0] > max_samples:
        start = (y.shape[0] - max_samples) // 2
        y = y[start:start + max_samples]
    return y


def _masked_mean_std(hidden: torch.Tensor, feat_mask: torch.Tensor):
    """
    Compute masked mean and std along the time axis.

    Args:
        hidden:    (B, T', D) — last hidden states
        feat_mask: (B, T')    — 1.0 for valid frames, 0.0 for padding

    Returns:
        mean: (B, D)
        std:  (B, D)
    """
    mask = feat_mask.unsqueeze(-1).float()           # (B, T', 1)
    count = mask.sum(dim=1).clamp(min=1.0)           # (B, 1)

    summed = (hidden * mask).sum(dim=1)              # (B, D)
    mean = summed / count                            # (B, D)

    diff = (hidden - mean.unsqueeze(1)) * mask       # zero-out padded frames
    var = (diff * diff).sum(dim=1) / count           # (B, D)
    std = var.clamp(min=1e-8).sqrt()                 # (B, D)

    return mean, std


# ----------------------------------------------------------------------
# Single-sample API (kept for backward compatibility)
# ----------------------------------------------------------------------
@torch.no_grad()
def extract_wav2vec_features(
    waveform,
    sr=16000,
    model_name="facebook/wav2vec2-base",
):
    """
    Extract 1536-dim Wav2Vec 2.0 features (mean+std pooled) from one waveform.

    Returns:
        dict with 'w2v_meanstd' -> np.ndarray shape (1536,) float32
    """
    assert sr == 16000, f"Wav2Vec2 expects 16kHz, got {sr}"
    processor, model, device = _get_wav2vec(model_name)

    y = _truncate(_to_numpy_1d(waveform), sr)

    inputs = processor(y, sampling_rate=sr, return_tensors="pt")
    input_values = inputs["input_values"].to(device)

    out = model(input_values)
    hidden = out.last_hidden_state                    # (1, T', 768)

    mean_vec = hidden.mean(dim=1)
    std_vec = hidden.std(dim=1, unbiased=False)

    feat = torch.cat([mean_vec, std_vec], dim=1).squeeze(0)
    feat = feat.detach().cpu().numpy().astype(np.float32)

    return {"w2v_meanstd": feat}


# ----------------------------------------------------------------------
# Batched API
# ----------------------------------------------------------------------
@torch.no_grad()
def extract_wav2vec_features_batch(
    waveforms: List[Union[np.ndarray, torch.Tensor]],
    sr: int = 16000,
    model_name: str = "facebook/wav2vec2-base",
    batch_size: int = 8,
) -> List[dict]:
    """
    Extract 1536-dim Wav2Vec 2.0 features for a list of waveforms.

    Pads to longest in each minibatch and uses attention-mask-aware pooling
    so padded frames don't pollute the mean/std.

    Args:
        waveforms:  list of 1D waveforms (np.ndarray or torch.Tensor) at `sr` Hz
        sr:         sampling rate (must be 16000)
        model_name: HF model id
        batch_size: minibatch size (4-8 fits easily on RTX 3060 for ~20s clips)

    Returns:
        list of dicts, each {'w2v_meanstd': np.ndarray (1536,) float32},
        in the same order as the input list.
    """
    assert sr == 16000, f"Wav2Vec2 expects 16kHz, got {sr}"
    processor, model, device = _get_wav2vec(model_name)

    # Pre-process all waveforms (1D float32, truncated)
    prepared = [_truncate(_to_numpy_1d(w), sr) for w in waveforms]

    results: List[dict] = [None] * len(prepared)

    for start in range(0, len(prepared), batch_size):
        chunk = prepared[start:start + batch_size]

        # Processor handles padding to longest in chunk + attention mask
        inputs = processor(
            chunk,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        input_values = inputs["input_values"].to(device)        # (B, T)
        attention_mask = inputs["attention_mask"].to(device)    # (B, T)

        out = model(input_values, attention_mask=attention_mask)
        hidden = out.last_hidden_state                          # (B, T', 768)

        # Convert sample-level mask -> frame-level mask (Wav2Vec2 ~320x downsample)
        feat_mask = model._get_feature_vector_attention_mask(
            hidden.shape[1], attention_mask
        )                                                       # (B, T')

        mean_vec, std_vec = _masked_mean_std(hidden, feat_mask) # (B, 768) each
        feats = torch.cat([mean_vec, std_vec], dim=1)           # (B, 1536)
        feats = feats.detach().cpu().numpy().astype(np.float32)

        for i in range(feats.shape[0]):
            results[start + i] = {"w2v_meanstd": feats[i]}

    return results