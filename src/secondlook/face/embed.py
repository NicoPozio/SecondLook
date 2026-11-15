"""Face detection and ArcFace embedding via insightface"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class FaceSample:
    """One detected face: L2-normalized embedding plus detection metadata"""

    embedding: np.ndarray #512 floats normalizzati (|embedding|=1)
    bbox: np.ndarray #vettore di 4 int (x e y dell'angolo alto a sinistra e dell'angolo basso a destra)
    det_score: float


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance in [0, 2]; 0 means identical direction"""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(1.0 - np.dot(a, b))


def load_image(path: str | Path) -> np.ndarray:
    """Read an image as BGR, which is what insightface expects"""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


class FaceEmbedder:
    """RetinaFace detection, 5-point alignment and ArcFace embedding

    buffalo_l is a packet of pretrained models, from which we take
    SCFRD (faces detection) and ArcFace (Face embedding).
    It's downloaded to ~/.insightface on first use
    """

    def __init__(self, model_pack: str = "buffalo_l", det_size: int = 640):
        from insightface.app import FaceAnalysis

        self._app = FaceAnalysis(
            name=model_pack,
            allowed_modules=["detection", "recognition"], #SCFRD e ArcFace
            providers=["CPUExecutionProvider"], #Esegui su CPU
        )
        self._app.prepare(ctx_id=-1, det_size=(det_size, det_size)) #Le foto sono messe a risoluzione det x det

    def faces(self, img: np.ndarray) -> list[FaceSample]:
        """All faces in the image, sorted by detector confidence"""
        found = self._app.get(img)
        samples = [
            FaceSample(
                embedding=f.normed_embedding.astype(np.float32),
                bbox=f.bbox.astype(int),
                det_score=float(f.det_score),
            )
            for f in found
        ]
        return sorted(samples, key=lambda s: s.det_score, reverse=True)

    def embed(self, img: np.ndarray) -> FaceSample:
        """The largest face in the image; raises if none is found"""
        samples = self.faces(img)
        if not samples:
            raise ValueError("No face detected")
        return max(samples, key=lambda s: _area(s.bbox))


def _area(bbox: np.ndarray) -> int:
    return int((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))