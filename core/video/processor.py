"""
Video processing orchestrator for face de-identification.

Coordinates VideoReader, FaceDetector, and de-identifier methods to
process video frames with optional detection skipping for real-time
performance.
"""

import time
import numpy as np
from pathlib import Path
from typing import Optional

from .reader import VideoReader
from .writer import VideoWriter


class VideoProcessor:
    """
    Orchestrates video de-identification pipeline.

    Reads frames, detects faces (with optional skip for speed),
    applies de-identification, and writes output frames.

    Args:
        detector: FaceDetector instance for face detection.
        deidentifier: De-identifier with process_frame(frame, face_bbox) method.
        detect_every_n: Run detection every N frames, reuse last bboxes in between.
        verbose: Print progress messages.
    """

    def __init__(self, detector, deidentifier, detect_every_n: int = 1, verbose: bool = True):
        self.detector = detector
        self.deidentifier = deidentifier
        self.detect_every_n = max(1, detect_every_n)
        self.verbose = verbose

    def process_video(self,
                      input_path: str,
                      output_path: str,
                      max_frames: Optional[int] = None,
                      fps: Optional[float] = None,
                      codec: str = 'mp4v') -> dict:
        """
        Process a single video or frame-sequence directory.

        Args:
            input_path: Path to input video file or frame directory.
            output_path: Path to output video file or frame directory.
            max_frames: Maximum frames to process (None = all).
            fps: Output FPS (None = match input).
            codec: FourCC codec for video output.

        Returns:
            Dictionary with processing statistics.
        """
        with VideoReader(input_path, max_frames=max_frames) as reader:
            source_fps = fps if fps is not None else reader.fps
            total_frames = reader.frame_count

            # Determine writer mode
            if reader.is_video_file:
                frame_names = None
            else:
                frame_names = reader.frame_names
                if max_frames is not None:
                    frame_names = frame_names[:max_frames]

            with VideoWriter(
                output_path,
                fps=source_fps,
                frame_size=reader.frame_size,
                codec=codec,
                frame_names=frame_names
            ) as writer:
                last_detections = []
                frames_processed = 0
                faces_detected = 0
                start_time = time.time()

                for frame_idx, frame in enumerate(reader):
                    # Detect faces (or reuse previous detections)
                    if frame_idx % self.detect_every_n == 0:
                        try:
                            last_detections = self.detector.detect(frame)
                            faces_detected += len(last_detections)
                        except Exception as e:
                            if self.verbose:
                                print(f"  Warning: Detection failed at frame {frame_idx}: {e}")
                            last_detections = []

                    # Apply de-identification to each detected face
                    result = frame.copy()
                    for det in last_detections:
                        bbox = det.bbox.astype(int)
                        result = self.deidentifier.process_frame(result, face_bbox=bbox)

                    writer.write(result)
                    frames_processed += 1

                    # Progress reporting
                    if self.verbose and frames_processed % 100 == 0:
                        elapsed = time.time() - start_time
                        proc_fps = frames_processed / elapsed if elapsed > 0 else 0
                        print(f"  Frame {frames_processed}/{total_frames} "
                              f"({proc_fps:.1f} fps)")

                elapsed = time.time() - start_time
                proc_fps = frames_processed / elapsed if elapsed > 0 else 0
                realtime = proc_fps >= source_fps

                stats = {
                    'input_path': str(input_path),
                    'output_path': str(output_path),
                    'frames_processed': frames_processed,
                    'total_frames': total_frames,
                    'faces_detected': faces_detected,
                    'source_fps': source_fps,
                    'processing_fps': round(proc_fps, 2),
                    'realtime': realtime,
                    'elapsed_seconds': round(elapsed, 2),
                    'detect_every_n': self.detect_every_n,
                }

                if self.verbose:
                    print(f"  Completed: {frames_processed} frames in {elapsed:.1f}s "
                          f"({proc_fps:.1f} fps, {'real-time' if realtime else 'non-real-time'})")

                return stats
