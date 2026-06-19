import torch
import torch.nn as nn

from pipelines.bodyEncoder import BodyEncoder
from pipelines.faceEncoder import FaceEncoder
from pipelines.voiceEncoder import VoiceEncoder
from pipelines.textEncoder import TextEncoder
from FusionLayer import FusionLayer

class EmotionModel(nn.Module):
    def __init__(self):
        super(EmotionModel, self).__init__()
        
        # 1. Encoders
        self.body_encoder = BodyEncoder()
        self.face_encoder = FaceEncoder()
        self.voice_encoder = VoiceEncoder()
        self.text_encoder = TextEncoder()
        
        # 2. Fusion Layer
        self.fusion = FusionLayer()
        
        # 3. Head 3 — Emotional State (7 Classes)
        self.head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 7)
        )

    def forward(self, body, face, voice, text, silent_mask=None):
        b_vec = self.body_encoder(body)
        f_vec = self.face_encoder(face)
        v_vec = self.voice_encoder(voice)
        t_vec = self.text_encoder(text, silent_mask)
        
        fused_vec = self.fusion(b_vec, f_vec, v_vec, t_vec)
        logits = self.head(fused_vec)
        
        return logits