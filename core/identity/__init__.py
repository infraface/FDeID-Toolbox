"""
Identity Recognition Module

This package provides:
- AdaFace: Face recognition and verification
- ArcFace: Face recognition and verification (alternative to AdaFace)
- CosFace: Face recognition and verification (alternative to AdaFace/ArcFace)
- FaceDetector: RetinaFace-based face detection
"""

from .adaface import AdaFace, RecognitionResult
from .arcface import ArcFace, ArcFaceRecognitionResult
from .cosface import CosFace, CosFaceRecognitionResult
from .retinaface import FaceDetector, DetectionResult

__all__ = [
    'AdaFace',
    'ArcFace',
    'CosFace',
    'RecognitionResult',
    'ArcFaceRecognitionResult',
    'CosFaceRecognitionResult',
    'FaceDetector',
    'DetectionResult',
]
