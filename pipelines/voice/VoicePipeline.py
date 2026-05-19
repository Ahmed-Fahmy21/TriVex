import torch
import numpy as np
from typing import Union
from config import TARGET_SR, TOTAL_FEATURE_DIM
from Voice.audio_features.preprocessing import load_and_preprocess
from Voice.audio_features.features_classical import extract_classical_features,flatten_features
from Voice.audio_features.features_wav2vec import extract_wav2vec_features


class VoicePipeline:
    def __init__(self, target_sr: int = TARGET_SR, device: Union[str, torch.device] = "cpu"):
        self.target_sr = target_sr
        self.device = torch.device(device)

    def process_waveform(self, waveform, sr: int = 16000) -> torch.Tensor:
        """
        Process a single waveform into feature tensor.
        Returns: torch.Tensor shape (TOTAL_FEATURE_DIM,)
        """
        if sr != self.target_sr:
            raise ValueError(f"Expected sample rate {self.target_sr}, got {sr}")

        # Classical features
        classical = flatten_features(
            extract_classical_features(waveform, sr)
        ).astype(np.float32)

        # Wav2Vec features
        wav2vec = extract_wav2vec_features(waveform, sr)["w2v_meanstd"].astype(np.float32)

        combined = np.concatenate([classical, wav2vec], axis=0).astype(np.float32)

        if combined.shape[0] != TOTAL_FEATURE_DIM:
            raise ValueError(
                f"Total feature dim mismatch: got {combined.shape[0]}, expected {TOTAL_FEATURE_DIM}"
            )

        return torch.from_numpy(combined).to(self.device)

    def process_file(self, wav_path: str) -> torch.Tensor:
        """
        Load audio file and process into feature tensor.
        Returns: torch.Tensor shape (TOTAL_FEATURE_DIM,)
        """
        waveform, sr = load_and_preprocess(wav_path, target_sr=self.target_sr)
        return self.process_waveform(waveform, sr)
