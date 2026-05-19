"""
Precompute RAVDESS Text Features (Whisper + BERT) locally.

Usage:
    python precompute_ravdess_text.py
"""

import json
import shutil
import librosa
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import whisper
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import kagglehub
import warnings

warnings.filterwarnings("ignore")

# ============================================
#  Local Paths & Configs
# ============================================
# Automatically detects your project root based on where this file is run
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent if Path(__file__).parent.name == "data" else Path.cwd()
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache" / "text_features"
LOGS_DIR = DATA_DIR / "logs"

RAVDESS_CACHE_DIR = CACHE_DIR / "ravdess"
RAVDESS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SR = 16000
EXPECTED_TEXT_DIM = 768
MIN_DURATION_SEC = 0.1

EMOTION_NAMES = ["neutral_calm", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]
RAVDESS_EMOTION_MAP = {"01": 0, "02": 0, "03": 1, "04": 2, "05": 3, "06": 4, "07": 5, "08": 6}
RAVDESS_TRAIN_ACTORS = list(range(1, 17))
RAVDESS_VAL_ACTORS = list(range(17, 21))
RAVDESS_TEST_ACTORS = list(range(21, 25))


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
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().float()
        return cls_embedding, text


# ============================================
#  Helpers
# ============================================
def get_ravdess_split(actor_id: int) -> Optional[str]:
    if actor_id in RAVDESS_TRAIN_ACTORS: return "train"
    if actor_id in RAVDESS_VAL_ACTORS:   return "val"
    if actor_id in RAVDESS_TEST_ACTORS:  return "test"
    return None


def build_ravdess_meta(wav_path: Path) -> Optional[dict]:
    codes = wav_path.stem.split("-")
    if len(codes) != 7: return None
    emotion_code, intensity, actor_id = codes[2], int(codes[3]), int(codes[6])

    if emotion_code not in RAVDESS_EMOTION_MAP: return None
    split = get_ravdess_split(actor_id)
    if split is None: return None

    emotion_id = RAVDESS_EMOTION_MAP[emotion_code]
    cache_path = RAVDESS_CACHE_DIR / f"{wav_path.stem}_text.pt"

    return {
        "wav_path": wav_path, "cache_path": cache_path,
        "emotion_id": emotion_id, "emotion_name": EMOTION_NAMES[emotion_id],
        "actor_id": actor_id, "speaker_id": f"ravdess_actor_{actor_id:02d}",
        "split": split, "intensity": intensity, "emotion_code": emotion_code,
        "original_filename": wav_path.name,
    }


# ============================================
#  Main
# ============================================
if __name__ == "__main__":
    print(f"📥 Local RAVDESS Text Extraction -> {RAVDESS_CACHE_DIR}")
    root_dir = Path(kagglehub.dataset_download("uwrfkaggler/ravdess-emotional-speech-audio"))
    wav_files = list(root_dir.rglob("*.wav"))

    todo, cached, skipped = [], 0, 0
    for w in wav_files:
        meta = build_ravdess_meta(w)
        if not meta:
            skipped += 1
        elif meta["cache_path"].exists():
            cached += 1
        else:
            todo.append(meta)

    print(f"Files: {len(wav_files)} | To process: {len(todo)} | Cached: {cached} | Skipped: {skipped}")

    if todo:
        extractor = TextFeatureExtractor()
        saved, failed = 0, []
        for s in tqdm(todo, desc="RAVDESS Processing"):
            try:
                waveform, sr = librosa.load(s["wav_path"], sr=TARGET_SR)
                if waveform.ndim > 1: waveform = waveform.mean(axis=0)
                if len(waveform) < sr * MIN_DURATION_SEC: continue

                text_tensor, text = extractor.process_waveform(waveform)

                save_dict = {
                    "text_feature": text_tensor, "text": text,
                    "emotion_id": s["emotion_id"], "emotion_name": s["emotion_name"],
                    "speaker_id": s["speaker_id"], "source": "ravdess",
                    "split": s["split"], "original_filename": s["original_filename"]
                }
                torch.save(save_dict, s["cache_path"])
                saved += 1
            except Exception as e:
                failed.append({"file": s["original_filename"], "error": str(e)})

        print(f"✅ Saved: {saved} | Failed: {len(failed)}")