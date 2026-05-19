"""
تحميل ومعالجة الملفات الصوتية
"""

import torch
import torchaudio


# cache الـ resampler عشان مش نعمله كل مرة
_RESAMPLERS = {}


def get_resampler(orig_sr: int, target_sr: int):
    """بيرجع resampler محفوظ - أسرع من إنه يعمل واحد جديد كل مرة"""
    key = (orig_sr, target_sr)
    if key not in _RESAMPLERS:
        _RESAMPLERS[key] = torchaudio.transforms.Resample(
            orig_freq=orig_sr,
            new_freq=target_sr,
        )
    return _RESAMPLERS[key]


def load_and_preprocess(wav_path: str, target_sr: int = 16000):
    """
    بيحمل ملف صوتي ويجهزه للـ feature extraction

    Args:
        wav_path: مسار الملف
        target_sr: الـ sample rate المطلوب (Wav2Vec محتاج 16000)

    Returns:
        waveform: torch.Tensor shape (T,)
        sr: int = target_sr
    """
    waveform, sr = torchaudio.load(wav_path)
    waveform = waveform.float()

    # check الملف مش فاضي
    if waveform.numel() == 0:
        raise ValueError(f"Empty audio file: {wav_path}")

    # convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # resample if needed
    if sr != target_sr:
        resampler = get_resampler(sr, target_sr)
        waveform = resampler(waveform)

    # normalize to [-1, 1]
    max_val = waveform.abs().max()
    if max_val > 0:
        waveform = waveform / max_val

    waveform = torch.clamp(waveform, -1.0, 1.0)

    # squeeze الـ channel dimension بس
    return waveform.squeeze(0), target_sr