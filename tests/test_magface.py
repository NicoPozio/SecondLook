"""Test di face/magface.py: incertezza appresa con la lunghezza del vettore MagFace."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from secondlook.face.magface import MAGFACE_FILE, MagFace


def test_prepare_by_hand():
    crop = np.zeros((112, 112, 3), dtype=np.uint8)
    crop[0, 0] = (255, 128, 0)                        # un pixel BGR
    x = MagFace.prepare(crop)
    assert x.shape == (1, 3, 112, 112)                # (foto, colore, alto, largo)
    assert x[0, :, 0, 0] == pytest.approx([1.0, 128 / 255, 0.0])   # BGR restano BGR, diviso 255
    with pytest.raises(ValueError):
        MagFace.prepare(np.zeros((100, 100, 3), dtype=np.uint8))


def test_missing_model_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="magface_to_onnx"):
        MagFace(tmp_path)


@pytest.fixture(scope="module")
def models():
    if not (Path("data/models") / MAGFACE_FILE).exists():
        pytest.skip("modello MagFace non presente in data/models (vedi tools/magface_to_onnx.py)")
    from secondlook.face import FaceEmbedder

    return MagFace(), FaceEmbedder()


def test_sharp_face_longer_than_blurred(models):
    import insightface

    magface, embedder = models
    img = cv2.imread(str(Path(insightface.__file__).parent / "data" / "images" / "t1.jpg"))
    crop = embedder.crop(img, select="center")
    sharp = magface.magnitude(crop)
    blurred = magface.magnitude(cv2.GaussianBlur(crop, (0, 0), 3))
    assert 10 < sharp < 110                           # limiti fissati nell'addestramento di MagFace
    assert blurred < sharp                            # foto rovinata -> vettore piu' corto
