"""
Precompute IEMOCAP Text Features (Whisper + BERT) locally.

Usage:
    python precompute_iemocap_text.py
"""

import os
import re
import json
import shutil
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import whisper
from transformers import BertTokenizer, BertModel
from datasets import load_dataset, Audio
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# ============================================
#  Local Paths & Configs
# ============================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent if Path(__file__).parent.name == "data" else Path.cwd()
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache" / "text_features"
LOGS_DIR = DATA_DIR / "logs"

IEMOCAP_CACHE_DIR = CACHE_DIR / "iemocap"
IEMOCAP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SR = 16000
EXPECTED_TEXT_DIM = 768
MIN_DURATION_SEC = 0.1

EMOTION_NAMES = ["neutral_calm", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]
IEMOCAP_EMOTION_MAP = {
    "neutral": 0, "happy": 1, "excited": 1,
    "sad": 2, "angry": 3, "frustrated": 3,
    "fear": 4, "surprise": 6,
}

IEMOCAP_TRAIN_SPEAKERS = ["Ses01_F", "Ses01_M", "Ses02_F", "Ses02_M", "Ses03_F", "Ses03_M"]
IEMOCAP_VAL_SPEAKERS = ["Ses04_F", "Ses04_M"]
IEMOCAP_TEST_SPEAKERS = ["Ses05_F", "Ses05_M"]


# ============================================
#  Extractor Class
# ============================================
class TextFeatureExtractor:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        print(f"Loading Whisper 'base' and BERT on {self.device}...")
        self.whisper_model = whisper.load_model("base").to(self.device)
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased').to(self.device)
        self.bert_model.eval()

    @torch.no_grad()
    def process_waveform(self, waveform: np.ndarray):
        result = self.whisper_model.transcribe(waveform, fp16=torch.cuda.is_available())
        text = result["text"].strip()
        if not text: text = "[SILENCE]"

        inputs = self.bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(
            self.device)
        outputs = self.bert_model(**inputs)
        return outputs.last_hidden_state[:, 0, :].squeeze().cpu().float(), text


# ============================================
#  Helpers
# ============================================
def get_speaker_id(filename: str) -> Optional[str]:
    prefix = filename.split("_")[0]
    if len(prefix) < 6 or prefix[5] not in ("F", "M"): return None
    return f"{prefix[:5]}_{prefix[5]}"


def build_iemocap_meta(idx: int, filename: str, major_emotion: str) -> Optional[dict]:
    if major_emotion not in IEMOCAP_EMOTION_MAP: return None
    speaker_id = get_speaker_id(filename)
    if not speaker_id: return None

    if speaker_id in IEMOCAP_TRAIN_SPEAKERS:
        split = "train"
    elif speaker_id in IEMOCAP_VAL_SPEAKERS:
        split = "val"
    elif speaker_id in IEMOCAP_TEST_SPEAKERS:
        split = "test"
    else:
        return None

    emotion_id = IEMOCAP_EMOTION_MAP[major_emotion]
    safe_name = re.sub(r'[/\\:*?"<>|\s]', '_', filename.rsplit(".", 1)[0])
    cache_path = IEMOCAP_CACHE_DIR / f"{safe_name}_text.pt"

    return {
        "ds_index": idx, "filename": filename, "cache_path": cache_path,
        "emotion_id": emotion_id, "emotion_name": EMOTION_NAMES[emotion_id],
        "original_emotion": major_emotion, "speaker_id": speaker_id, "split": split,
    }


# ============================================
#  Main
# ============================================
if __name__ == "__main__":
    print(f"📥 Local IEMOCAP Text Extraction -> {IEMOCAP_CACHE_DIR}")
    ds = load_dataset("AbstractTTS/IEMOCAP")

    all_files, all_emotions = ds["train"]["file"], ds["train"]["major_emotion"]

    todo, cached, skipped = [], 0, 0
    for i in range(len(all_files)):
        meta = build_iemocap_meta(i, all_files[i], all_emotions[i])
        if not meta:
            skipped += 1
        elif meta["cache_path"].exists():
            cached += 1
        else:
            todo.append(meta)

    print(f"Rows: {len(all_files)} | To process: {len(todo)} | Cached: {cached} | Skipped: {skipped}")

    if todo:
        ds_audio = ds["train"].cast_column("audio", Audio(sampling_rate=TARGET_SR, decode=True))
        extractor = TextFeatureExtractor()
        saved, failed = 0, []

        for s in tqdm(todo, desc="IEMOCAP Processing"):
            try:
                audio = ds_audio[s["ds_index"]]["audio"]
                waveform = np.asarray(audio["array"], dtype=np.float32)
                if waveform.ndim > 1: waveform = waveform.mean(axis=0)
                if len(waveform) < TARGET_SR * MIN_DURATION_SEC: continue

                text_tensor, text = extractor.process_waveform(waveform)

                save_dict = {
                    "text_feature": text_tensor, "text": text,
                    "emotion_id": s["emotion_id"], "emotion_name": s["emotion_name"],
                    "speaker_id": s["speaker_id"], "source": "iemocap",
                    "split": s["split"], "original_filename": s["filename"],
                }
                torch.save(save_dict, s["cache_path"])
                saved += 1
            except Exception as e:
                failed.append({"file": s["filename"], "error": str(e)})

        print(f"✅ Saved: {saved} | Failed: {len(failed)}")