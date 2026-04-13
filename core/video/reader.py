"""
Video reader supporting both video files and frame-sequence directories.

Provides a unified iterator interface for reading frames from video files
(via cv2.VideoCapture) or directories of image frames (sorted by filename).
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple

from .utils import IMAGE_EXTENSIONS


class VideoReader:
    """
    Unified video reader for video files and frame directories.

    Usage:
        with VideoReader('/path/to/video.mp4') as reader:
            for frame in reader:
                process(frame)

        with VideoReader('/path/to/frames_dir') as reader:
            for frame in reader:
                process(frame)
    """

    def __init__(self, path: str, max_frames: Optional[int] = None):
        """
        Args:
            path: Path to a video file or directory of image frames.
            max_frames: Maximum number of frames to read (None = all).
        """
        self._path = Path(path)
        self._max_frames = max_frames
        self._cap = None
        self._frame_paths: List[Path] = []
        self._frame_idx = 0
        self._is_video = False

        if self._path.is_file():
            self._is_video = True
            self._cap = cv2.VideoCapture(str(self._path))
            if not self._cap.isOpened():
                raise IOError(f"Cannot open video file: {self._path}")
        elif self._path.is_dir():
            self._is_video = False
            self._frame_paths = sorted([
                f for f in self._path.iterdir()
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
            ])
            if not self._frame_paths:
                raise IOError(f"No image files found in: {self._path}")
        else:
            raise IOError(f"Path does not exist: {self._path}")

    @property
    def is_video_file(self) -> bool:
        return self._is_video

    @property
    def fps(self) -> float:
        if self._is_video and self._cap is not None:
            return self._cap.get(cv2.CAP_PROP_FPS)
        return 30.0  # Default for frame sequences

    @property
    def frame_count(self) -> int:
        if self._is_video and self._cap is not None:
            count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            count = len(self._frame_paths)
        if self._max_frames is not None:
            count = min(count, self._max_frames)
        return count

    @property
    def frame_size(self) -> Tuple[int, int]:
        """Returns (width, height)."""
        if self._is_video and self._cap is not None:
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (w, h)
        elif self._frame_paths:
            img = cv2.imread(str(self._frame_paths[0]))
            if img is not None:
                h, w = img.shape[:2]
                return (w, h)
        return (0, 0)

    @property
    def frame_names(self) -> List[str]:
        """Original filenames for frame-sequence directories."""
        if not self._is_video:
            return [f.name for f in self._frame_paths]
        return []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def release(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __iter__(self):
        self._frame_idx = 0
        if self._is_video and self._cap is not None:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return self

    def __next__(self) -> np.ndarray:
        if self._max_frames is not None and self._frame_idx >= self._max_frames:
            raise StopIteration

        if self._is_video:
            if self._cap is None or not self._cap.isOpened():
                raise StopIteration
            ret, frame = self._cap.read()
            if not ret:
                raise StopIteration
            self._frame_idx += 1
            return frame
        else:
            while self._frame_idx < len(self._frame_paths):
                frame = cv2.imread(str(self._frame_paths[self._frame_idx]))
                self._frame_idx += 1
                if frame is not None:
                    return frame
            raise StopIteration
