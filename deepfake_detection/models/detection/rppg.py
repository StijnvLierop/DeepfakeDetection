from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union, Literal, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

from deepfake_detection.data.dataset import Dataset
from deepfake_detection.data.instance import (
    FileVideoInstance,
    Instance,
    VideoFeatureInstance,
)
from deepfake_detection.models import Model, Prediction, TrainableMixin
from deepfake_detection.models.custom_networks.rppg_methods.green import green
from deepfake_detection.models.custom_networks.rppg_methods.pos import pos

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MediaPipe facial landmark regions
# Each key maps to a list of landmark indices from the 468-point face mesh.
# ---------------------------------------------------------------------------

LANDMARK_REGIONS: dict[str, list[int]] = {
    # High Priority Zones (Best for rPPG)
    "high_prio_forehead": [10, 67, 69, 104, 108, 109, 151, 299, 337, 338],
    "high_prio_nose": [3, 4, 5, 6, 45, 51, 115, 122, 131, 134, 142, 174, 195, 196, 197, 198, 209,
                       217, 220, 236, 248, 275, 277, 281, 360, 363, 399, 419, 420, 429, 437, 440,
                       456],
    "high_prio_left_cheek": [36, 47, 50, 100, 101, 116, 117, 118, 119, 123, 126, 147, 187, 203, 205,
                             206, 207, 216],
    "high_prio_right_cheek": [266, 280, 329, 330, 346, 347, 348, 355, 371, 411, 423, 425, 426, 427,
                              436],

    # Mid Priority Zones
    "mid_prio_forehead": [8, 9, 21, 68, 103, 251, 284, 297, 298, 301, 332, 333, 372, 383],
    "mid_prio_nose": [1, 44, 49, 114, 120, 121, 128, 168, 188, 351, 358, 412],
    "mid_prio_left_cheek": [34, 111, 137, 156, 177, 192, 213, 227, 234],
    "mid_prio_right_cheek": [340, 345, 352, 361, 454],
    "mid_prio_chin": [135, 138, 169, 170, 199, 208, 210, 211, 214, 262, 288, 416, 428, 430, 431,
                      432, 433, 434],
    "mid_prio_mouth": [92, 164, 165, 167, 186, 212, 322, 391, 393, 410],

    # Specific Anatomical Segments
    "forehead_left": [21, 71, 68, 54, 103, 104, 63, 70, 53, 52, 65, 107, 66, 108, 69, 67, 109, 105],
    "forehead_center": [10, 151, 9, 8, 107, 336, 285, 55],
    "forehead_right": [338, 337, 336, 296, 285, 295, 282, 334, 293, 301, 251, 298, 333, 299, 297,
                       332, 284],
    "cheek_left_top": [116, 111, 117, 118, 119, 100, 47, 126, 101, 123, 137, 177, 50, 36, 209, 129,
                       205, 147, 187, 215, 206, 203],
    "cheek_right_top": [349, 348, 347, 346, 345, 447, 323, 280, 352, 330, 371, 358, 423, 426, 425,
                        427, 411, 376],
    "cheek_left_bottom": [215, 138, 135, 210, 212, 57, 216, 207, 192],
    "cheek_right_bottom": [435, 427, 416, 364, 394, 422, 287, 410, 434, 436],
    "nose_full": [193, 417, 168, 188, 6, 412, 197, 174, 399, 456, 195, 236, 131, 51, 281, 360, 440,
                  4, 220, 219, 305],
    "chin_full": [204, 170, 140, 194, 201, 171, 175, 200, 418, 396, 369, 421, 431, 379, 424],

    # Exclusion Zones
    "left_eye": [157, 144, 145, 22, 23, 25, 154, 31, 160, 33, 46, 52, 53, 55, 56, 189, 190, 63, 65,
                 66, 70, 221, 222, 223, 225, 226, 228, 229, 230, 231, 232, 105, 233, 107, 243, 124],
    "right_eye": [384, 385, 386, 259, 388, 261, 265, 398, 276, 282, 283, 285, 413, 293, 296, 300,
                  441, 442, 445, 446, 449, 451, 334, 463, 336, 464, 467, 339, 341, 342, 353, 381,
                  373, 249, 253, 255],
    "mouth_full": [391, 393, 11, 269, 270, 271, 287, 164, 165, 37, 167, 40, 43, 181, 313, 314, 186,
                   57, 315, 61, 321, 73, 76, 335, 83, 85, 90, 106],

    # Global Coverage
    "equispaced_facial_points": [2, 3, 4, 5, 6, 8, 9, 10, 18, 21, 32, 35, 36, 43, 46, 47, 48, 50,
                                 54, 58, 67, 68, 69, 71, 92, 93, 101, 103, 104, 108, 109, 116, 117,
                                 118, 123, 132, 134, 135, 138, 139, 142, 148, 149, 150, 151, 152,
                                 182, 187, 188, 193, 197, 201, 205, 206, 207, 210, 211, 212, 216,
                                 234, 248, 251, 262, 265, 266, 273, 277, 278, 280, 284, 288, 297,
                                 299, 322, 323, 330, 332, 333, 337, 338, 345, 346, 361, 363, 364,
                                 367, 368, 371, 377, 379, 411, 412, 417, 421, 425, 426, 427, 430,
                                 432, 436],
}

# Default 77-landmark selection (one green-channel value per landmark = 77 features/frame).
SELECTED_LANDMARKS: list[int] = [
    10, 34, 35, 36, 47, 50, 53, 67, 69, 70, 100, 101, 104, 108, 109, 111, 116,
    117, 118, 119, 121, 123, 124, 126, 127, 139, 143, 147, 151, 187, 189, 203,
    205, 206, 207, 216, 222, 228, 230, 234, 244, 264, 266, 276, 280, 282, 283,
    299, 300, 329, 330, 337, 338, 340, 346, 347, 348, 353, 355, 368, 371, 372,
    411, 417, 423, 425, 426, 427, 436, 441, 444, 446, 448, 450, 452, 454, 464,
]


# ---------------------------------------------------------------------------
# Output dataclass (HuggingFace Trainer expects .loss and .logits)
# ---------------------------------------------------------------------------


@dataclass
class BVPLSTMOutput:
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class rPPGLSTM(TrainableMixin, Model):
    """
    Multi-layer LSTM classifier for rPPG-based deepfake detection.

    Accepts two input types:
    - FileVideoInstance     rPPG features are extracted on the fly from the video.
    - VideoFeatureInstance  precomputed features loaded from disk or memory.

    Feature shape fed to the LSTM: (batch, num_frames, num_features).

    Feature extraction uses MediaPipe FaceLandmarker to locate facial landmarks
    and samples the mean green-channel value in a 5×5 patch around each of the
    ``num_features`` selected landmark points per frame.  This produces one
    scalar per landmark per frame, giving a (num_frames, num_features) array
    that captures the temporal rPPG signal at anatomically meaningful skin sites.

    Set ``feature_cache_dir`` to a directory path to enable transparent feature
    caching: on the first call for a given video, features are extracted and saved
    as a ``.npy`` file; on subsequent calls (e.g. every training epoch after the
    first) the cached file is loaded instead, avoiding redundant extraction.
    """

    def __init__(
        self,
        name: str = "rPPGLSTM",
        load_model: bool = False,
        ckpt: Optional[str] = None,
        extraction_method: Literal["GREEN", "POS"] = "GREEN",
        num_frames: int = 120,
        num_features: int = 77,
        num_layers: int = 5,
        hidden_size: int = 128,
        num_classes: int = 2,
        feature_cache_dir: Optional[str] = None,
        landmark_model_path: str = "face_landmarker.task",
        selected_landmark_regions: Optional[List[str]] = None,
    ):
        # torch.nn.Module must be initialised before registering parameters.
        super().__init__()

        self.ckpt = ckpt
        self.num_frames = num_frames
        self.num_features = num_features
        self.feature_cache_dir = Path(feature_cache_dir) if feature_cache_dir else None
        self.landmark_model_path = landmark_model_path
        self.extraction_method = extraction_method

        # Build landmark index list from regions (deduplicated, order-preserving),
        # or fall back to the default 77-point selection.
        if selected_landmark_regions:
            self._selected_landmarks: list[int] = list(
                dict.fromkeys(
                    itertools.chain.from_iterable(
                        LANDMARK_REGIONS[r]
                        for r in selected_landmark_regions
                        if r in LANDMARK_REGIONS
                    )
                )
            )
        else:
            self._selected_landmarks = SELECTED_LANDMARKS

        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

        Model.__init__(self, name=name, load_model=load_model)

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        if self.ckpt is None:
            return
        path = Path(self.ckpt)
        if not path.exists():
            raise FileNotFoundError(f"rPPGLSTM checkpoint not found: {self.ckpt}")
        if path.suffix == ".safetensors":
            try:
                from safetensors.torch import load_file
            except ImportError as exc:
                raise ImportError(
                    "Loading .safetensors checkpoints requires the safetensors package."
                ) from exc
            self.load_state_dict(load_file(str(path)))
        else:
            payload = torch.load(path, map_location="cpu")
            state = payload.get("state_dict", payload)
            self.load_state_dict(state)

    # ------------------------------------------------------------------
    # rPPG feature extraction using MediaPipe library
    # ------------------------------------------------------------------

    def _extract_landmarks(self, frames_rgb: list[np.ndarray], fps: float) -> Tuple[np.ndarray, List]:
        """
        Extracts facial landmarks from a list of frames and returns them as a numpy array.

        :param frames_rgb: RGB frames to extract landmarks from.
        :param fps: The frame rate of the video.
        :return: - A numpy array containing the mean RGB value
                   of the selected landmarks with shape (num_frames, 3).
                 - A list of FaceLandmarker detection results with length num_frames.
        """
        import mediapipe as mp

        # Get landmarks
        base_options = mp.tasks.BaseOptions(
            model_asset_path=self.landmark_model_path
        )
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
        )
        landmark_detector = mp.tasks.vision.FaceLandmarker.create_from_options(options)

        # Get selected landmarks
        selected_landmarks = [v for l, v in LANDMARK_REGIONS.items() if l in self._selected_landmarks]
        selected_landmarks = list(itertools.chain.from_iterable(selected_landmarks))

        # Get face landmarks for all frames
        landmarks_list = []
        landmark_detector_results = []
        for frame_idx, frame in enumerate(list(frames_rgb)):

            # Get timestamp
            timestamp_ms = int((frame_idx / fps) * 1000)

            # Detect landmarks for frame
            landmark_detector_result = landmark_detector.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=frame),
                timestamp_ms
            )

            # If face landmarks were found
            if landmark_detector_result.face_landmarks:

                image_height, image_width, _ = frame.shape

                # Get landmarks
                landmarks = landmark_detector_result.face_landmarks[0]

                # Extract pixel values at the specific locations
                pixel_colors = []
                roi_size = 5  # We will take a 5x5 square around each landmark (25 pixels total)
                offset = roi_size // 2
                for index in selected_landmarks:
                    landmark = landmarks[index]

                    # Get landmark center coordinates
                    x_px = int(landmark.x * image_width)
                    y_px = int(landmark.y * image_height)

                    # Define ROI boundaries and clip to image dimensions
                    y1 = max(0, y_px - offset)
                    y2 = min(image_height, y_px + offset + 1)
                    x1 = max(0, x_px - offset)
                    x2 = min(image_width, x_px + offset + 1)

                    # Extract the square patch and average its pixels
                    roi_patch = frame[y1:y2, x1:x2]
                    if roi_patch.size > 0:
                        # Average over the patch (spatial filtering)
                        mean_color = np.mean(roi_patch, axis=(0, 1))
                        pixel_colors.append(mean_color)

                # Add mean RGB value of landmark locations to landmarks list
                landmarks_list.append(np.mean(pixel_colors, axis=0))

            else:
                pass

            # Add predicted landmark locations result
            landmark_detector_results.append(landmark_detector_result)

        return np.array(landmarks_list), landmark_detector_results

    def _extract_signal_from_rgb_landmarks(self, landmarks: np.ndarray, fps: float) -> np.ndarray:
        """
        Extract BVP signal from RGB landmarks using the specified method.

        :param landmarks: Detected landmarks of shape (num_frames, num_landmarks).
        :param fps: Frames per second of the video.
        :return: Extracted BVP signal.
        """
        print(landmarks.shape)
        if self.extraction_method == 'GREEN':
            bvp_signal = green(landmarks)
        elif self.extraction_method == 'POS':
            bvp_signal = pos(landmarks, fps)

        return bvp_signal

    def _resample(self, signal: np.ndarray) -> np.ndarray:
        """
        Uniformly resample ``signal`` along the time axis to ``num_frames`` rows.

        If the signal has more or fewer rows than ``num_frames``, linear interpolation is used to produce exactly
        ``num_frames`` rows. The feature dimension is trimmed or zero-padded to ``num_features``.
        """
        T, F = signal.shape

        # Resample time axis
        if T != self.num_frames:
            if T == 0:
                signal = np.zeros((self.num_frames, F), dtype=np.float32)
            else:
                old_t = np.linspace(0, 1, T)
                new_t = np.linspace(0, 1, self.num_frames)
                resampled = np.empty((self.num_frames, F), dtype=np.float32)
                for j in range(F):
                    resampled[:, j] = np.interp(new_t, old_t, signal[:, j])
                signal = resampled

        # Align feature dimension
        if F >= self.num_features:
            return signal[:, : self.num_features]
        return np.pad(signal, ((0, 0), (0, self.num_features - F)))

    def extract_features(self, cap: cv2.VideoCapture) -> np.ndarray:
        """
        Extract rPPG features from an open VideoCapture.

        Returns an array of shape (num_frames, num_features). The capture
        position is reset to frame 0 before reading so the same instance can be
        reused across multiple calls (e.g. different training epochs without caching).
        """
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        # Read videoframes in BGR format
        frames_bgr: list[np.ndarray] = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames_bgr.append(frame)

        if not frames_bgr:
            return np.zeros((self.num_frames, self.num_features), dtype=np.float32)

        try:
            # Convert frames to RGB format
            frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]

            # Extract landmarks
            landmarks, _ = self._extract_landmarks(frames_rgb, fps)

            # Extract rPPG features
            signal = self._extract_signal_from_rgb_landmarks(landmarks, fps)
        except ImportError:
            raise ImportError()

        return self._resample(signal)

    # ------------------------------------------------------------------
    # TrainableMixin interface
    # ------------------------------------------------------------------

    def _cache_path(self, instance: FileVideoInstance) -> Optional[Path]:
        """Return the .npy cache path for a video instance, or None if caching is disabled."""
        if self.feature_cache_dir is None:
            return None
        self.feature_cache_dir.mkdir(parents=True, exist_ok=True)
        return self.feature_cache_dir / f"{hash(instance.path)}.npy"

    def transform_input(self, instance: Instance) -> torch.Tensor:
        """
        Convert an Instance into a (num_frames, num_features) float32 tensor.

        Accepts FileVideoInstance or VideoFeatureInstance.  For FileVideoInstance,
        if ``feature_cache_dir`` is set the extracted features are written to a
        ``.npy`` file on the first call and loaded from it on subsequent calls,
        so extraction only happens once per video across all training epochs.
        """
        # If features instance load these
        if isinstance(instance, VideoFeatureInstance):
            return torch.from_numpy(np.asarray(instance.data, dtype=np.float32))

        # Otherwise load features from file video instance if cached, otherwise calculate
        if isinstance(instance, FileVideoInstance):
            cache = self._cache_path(instance)
            if cache is not None and cache.exists():
                return torch.from_numpy(np.load(cache))
            features = self.extract_features(instance.data)
            if cache is not None:
                np.save(cache, features)
            return torch.from_numpy(features)

        raise TypeError(
            f"rPPGLSTM expects FileVideoInstance or VideoFeatureInstance, "
            f"got {type(instance).__name__}"
        )

    @property
    def data_collator(self):
        return self._collate_fn

    @staticmethod
    def _collate_fn(batch: list) -> dict:
        """
        Collate a list of {"inputs": tensor, "labels": tensor} dicts (as
        produced by TorchDataset) into a single batched dict for the HF Trainer.
        """
        return {
            "pixel_values": torch.stack([item["inputs"] for item in batch]),
            "labels": torch.stack([item["labels"] for item in batch]),
        }

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> BVPLSTMOutput:
        lstm_out, _ = self.lstm(pixel_values)
        last_hidden = lstm_out[:, -1, :]
        logits = self.classifier(last_hidden)
        loss = self.loss_fn(logits, labels) if labels is not None else None
        return BVPLSTMOutput(loss=loss, logits=logits)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_batch(self, instances: Union[List[Instance], Dataset]) -> List[Prediction]:
        device = next(self.lstm.parameters()).device
        inputs = torch.stack(
            [self.transform_input(i) for i in instances]
        ).to(device)
        with torch.no_grad():
            out = self.forward(inputs)
        probs = torch.softmax(out.logits, dim=-1)
        return [
            Prediction(classification={"real": float(p[0]), "fake": float(p[1])})
            for p in probs
        ]
