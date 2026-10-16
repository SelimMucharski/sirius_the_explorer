"""ProbeKeypointDetector on the Ultralytics pose3d model (oak_probe_vision train_pose3d.py)."""

from typing import List

import numpy as np

from probe_detector.domain.ports import ProbeKeypointDetector
from probe_detector.domain.probe import ProbeKeypoints


class YoloPose3dDetector(ProbeKeypointDetector):

    def __init__(self, weights: str, imgsz: int, conf: float = 0.25, device: str = ''):
        from probe_detector.infrastructure import pose3d_patch  # noqa: F401  (before YOLO)
        from ultralytics import YOLO
        self._model = YOLO(weights)
        self._imgsz, self._conf, self._device = imgsz, conf, device or None
        yaml = getattr(self._model.model, 'yaml', None)
        kpt = yaml.get('kpt_shape') if yaml else None
        if kpt is not None and list(kpt) != [4, 4]:
            raise ValueError(f'{weights}: kpt_shape {kpt}, expected [4, 4] (pose3d model)')

    def warmup(self, width: int, height: int) -> None:
        self.detect(np.zeros((height, width, 3), np.uint8))

    def detect(self, image_bgr: np.ndarray) -> List[ProbeKeypoints]:
        r = self._model.predict(image_bgr, imgsz=self._imgsz, conf=self._conf, device=self._device,
                                verbose=False)[0]
        if not len(r.boxes):
            return []
        kps = r.keypoints.data.cpu().numpy()
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        return [ProbeKeypoints(keypoints=k, box_xyxy=tuple(float(x) for x in b), score=float(s))
                for k, b, s in zip(kps, boxes, scores)]
