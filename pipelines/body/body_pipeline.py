import numpy as np
import torch


class BodyPipelineExtractor:
    NUM_KEYPOINTS = 33
    FEATURES_PER_KP = 4
    FEATURE_DIM = 33 * 4  # 132

    def __init__(
        self,
        static_image_mode: bool = False,
        min_detection_confidence: float = 0.5,
        model_path: str = None,
    ):
        import mediapipe as mp
        self._mp = mp
        self._version = tuple(int(x) for x in mp.__version__.split(".")[:2])
        self._use_new_api = self._version >= (0, 10)

        if self._use_new_api:
            if model_path is None:
                raise RuntimeError(
                    "\n❌ MediaPipe >= 0.10 Needs model file.\n\n"
                    "\n Download it With :"
                    "  wget -O pose_landmarker.task \\\n"
                    "    https://storage.googleapis.com/mediapipe-models/"
                    "pose_landmarker/pose_landmarker_lite/float16/latest/"
                    "pose_landmarker_lite.task\n\n"
                    "ثم:\n"
                    "  BodyPipelineExtractor(model_path='pose_landmarker.task')\n"
                )
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.core.base_options import BaseOptions
            options = vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=min_detection_confidence,
            )
            self._landmarker = vision.PoseLandmarker.create_from_options(options)
        else:
            mp_pose = mp.solutions.pose
            self._pose = mp_pose.Pose(
                static_image_mode=static_image_mode,
                model_complexity=1,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=0.5,
            )

    def _process_frame(self, frame_rgb: np.ndarray) -> np.ndarray:
        if self._use_new_api:
            img = self._mp.Image(
                image_format=self._mp.ImageFormat.SRGB,
                data=frame_rgb.astype(np.uint8),
            )
            result = self._landmarker.detect(img)
            if not result.pose_landmarks:
                return np.zeros(self.FEATURE_DIM, dtype=np.float32)
            keypoints = []
            for lm in result.pose_landmarks[0]:
                keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])
        else:
            result = self._pose.process(frame_rgb)
            if result.pose_landmarks is None:
                return np.zeros(self.FEATURE_DIM, dtype=np.float32)
            keypoints = []
            for lm in result.pose_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])

        return np.array(keypoints, dtype=np.float32)

    def extract_from_frames(self, frames: list) -> torch.Tensor:

        features = [self._process_frame(f) for f in frames]
        return torch.from_numpy(np.stack(features, axis=0))

    def extract_from_video(self, video_path: str, max_frames: int = 150) -> torch.Tensor:

        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Can't open : {video_path}")
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        if not frames:
            raise ValueError(f"None Video : {video_path}")
        if len(frames) > max_frames:
            idx = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
            frames = [frames[i] for i in idx]
        return self.extract_from_frames(frames)

    def close(self):
        if self._use_new_api:
            self._landmarker.close()
        else:
            self._pose.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
