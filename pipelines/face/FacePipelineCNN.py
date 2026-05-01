import cv2
import numpy as np
import mediapipe as mp
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms


class FaceEncoderCNNModel(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()

        backbone = models.resnet18(weights=None)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone

        self.encoder = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128)
        )

        self.head = nn.Linear(128, num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        z = self.encoder(feats)
        logits = self.head(z)
        return z, logits


class FacePipelineCNN:
    def __init__(
        self,
        model_weights_path="face_encoder_classifier_affectnet_cnn.pth",
        max_num_faces=1,
        min_detection_confidence=0.5,
        device=None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=min_detection_confidence
        )

        self.model = FaceEncoderCNNModel(num_classes=7).to(self.device)
        self.model.load_state_dict(torch.load(model_weights_path, map_location=self.device))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def detect_face_bbox(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb)

        if not results.detections:
            return None, 0.0

        det = results.detections[0]
        bbox = det.location_data.relative_bounding_box

        x1 = max(int(bbox.xmin * w), 0)
        y1 = max(int(bbox.ymin * h), 0)
        bw = int(bbox.width * w)
        bh = int(bbox.height * h)
        x2 = min(x1 + bw, w - 1)
        y2 = min(y1 + bh, h - 1)

        conf = det.score[0] if det.score else 0.0
        return (x1, y1, x2, y2), float(conf)

    def crop_face(self, frame_bgr, bbox, margin=0.15):
        if bbox is None:
            return None

        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = bbox

        bw = x2 - x1
        bh = y2 - y1

        mx = int(bw * margin)
        my = int(bh * margin)

        nx1 = max(0, x1 - mx)
        ny1 = max(0, y1 - my)
        nx2 = min(w, x2 + mx)
        ny2 = min(h, y2 + my)

        crop = frame_bgr[ny1:ny2, nx1:nx2]
        if crop.size == 0:
            return None

        return crop

    def encode_face_crop(self, face_crop_bgr):
        rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        x = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            z, logits = self.model(x)

        z = z.squeeze(0).cpu().numpy().astype(np.float32)
        logits = logits.squeeze(0).cpu().numpy().astype(np.float32)

        probs = np.exp(logits - np.max(logits))
        probs = probs / (np.sum(probs) + 1e-8)

        return z, probs

    def sample_frames_from_video(self, video_path, start_sec=None, end_sec=None, sample_fps=6):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0.0

        if start_sec is None:
            start_sec = 0.0
        if end_sec is None:
            end_sec = duration

        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)
        frame_interval = max(int(fps / sample_fps), 1)

        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        idx = start_frame
        while cap.isOpened() and idx <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            if (idx - start_frame) % frame_interval == 0:
                frames.append(frame)

            idx += 1

        cap.release()
        return frames

    def process_frames(self, frames):
        embeddings = []
        probs_all = []
        det_scores = []

        for frame in frames:
            bbox, conf = self.detect_face_bbox(frame)
            if bbox is None:
                continue

            face_crop = self.crop_face(frame, bbox)
            if face_crop is None:
                continue

            try:
                z, probs = self.encode_face_crop(face_crop)
                embeddings.append(z)
                probs_all.append(probs)
                det_scores.append(conf)
            except Exception:
                continue

        if len(embeddings) == 0:
            face_vector_128 = np.zeros(128, dtype=np.float32)
            emotion_probs_mean = np.zeros(7, dtype=np.float32)
            quality_score = 0.0
        else:
            face_vector_128 = np.mean(np.stack(embeddings, axis=0), axis=0).astype(np.float32)
            emotion_probs_mean = np.mean(np.stack(probs_all, axis=0), axis=0).astype(np.float32)
            quality_score = float(np.mean(det_scores)) if len(det_scores) > 0 else 0.0

        return {
            "face_vector_128": face_vector_128,
            "emotion_probs_mean": emotion_probs_mean,
            "num_sampled_frames": len(frames),
            "num_valid_face_frames": len(embeddings),
            "face_present_ratio": float(len(embeddings) / max(len(frames), 1)),
            "quality_score": quality_score
        }

    def process_video(self, video_path, start_sec=None, end_sec=None, sample_fps=6):
        frames = self.sample_frames_from_video(video_path, start_sec, end_sec, sample_fps)
        return self.process_frames(frames)