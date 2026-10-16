"""Ports of the probe detector. Pure Python, no ROS dependencies."""

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from probe_detector.domain.probe import ProbeKeypoints


class ProbeKeypointDetector(ABC):
    """Finds probes in an image and returns their keypoints with predicted depth."""

    @abstractmethod
    def detect(self, image_bgr: np.ndarray) -> List[ProbeKeypoints]:
        ...
