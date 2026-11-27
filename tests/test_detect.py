"""Test di face/detect.py e scripts/eval_detectors.py: Viola-Jones contro SCRFD."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from secondlook.face.detect import iou, square_crop

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from eval_detectors import detection_stats  # noqa: E402


def test_iou_by_hand():
    assert iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(50 / 150)      # 0.33
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_square_crop_size_and_center():
    img = np.zeros((250, 250, 3), dtype=np.uint8)
    img[120:130, 120:130] = 255                       # un quadratino bianco al centro del riquadro
    crop = square_crop(img, [100, 100, 150, 150])     # lato 1.2 * 50 = 60, centro (125, 125)
    assert crop.shape == (112, 112, 3)
    assert crop[56, 56].mean() > 200                  # il centro del ritaglio e' il quadratino
    edge = square_crop(img, [0, 0, 40, 40])           # riquadro sul bordo: niente errori
    assert edge.shape == (112, 112, 3)


def test_detection_stats_by_hand():
    nan = np.nan
    v = {
        "a": np.r_[[0, 0, 10, 10], [0, 0, 10, 10], np.zeros(1024)],    # entrambi, IoU 1
        "b": np.r_[[0, 0, 10, 10], [nan] * 4, np.zeros(1024)],         # solo SCRFD
        "c": np.r_[[0, 0, 10, 10], [20, 20, 30, 30], np.zeros(1024)],  # entrambi, volti diversi
    }
    s = detection_stats(v)
    assert (s["scrfd_found"], s["vj_found"], s["vj_correct"], s["vj_only"]) == (3, 2, 1, 0)
    assert s["vj_correct_rate"] == pytest.approx(1 / 3)


@pytest.fixture(scope="module")
def models():
    from secondlook.face import FaceEmbedder
    from secondlook.face.detect import haar_cascade

    if not hasattr(cv2, "CascadeClassifier"):
        pytest.skip("OpenCV senza Viola-Jones (OpenCV 5 senza contrib)")
    try:
        return FaceEmbedder(), haar_cascade()
    except Exception as exc:
        pytest.skip(f"modelli non disponibili: {exc}")


@pytest.fixture(scope="module")
def group_photo():
    import insightface

    return cv2.imread(str(Path(insightface.__file__).parent / "data" / "images" / "t1.jpg"))


def test_viola_jones_finds_faces_where_scrfd_does(models, group_photo):
    from secondlook.face.detect import viola_jones

    embedder, cascade = models
    vj = viola_jones(group_photo, cascade)
    scrfd = [f.bbox for f in embedder.faces(group_photo)]
    assert len(vj) >= 1
    # Sulla foto di gruppo SCRFD trova 6 volti, Viola-Jones solo quelli piu' frontali
    # (3 su 6 con IoU ~0.7-0.75; i volti girati li perde: e' il limite che vogliamo misurare).
    # Si controlla che quelli che trova siano davvero volti di SCRFD, non che li trovi tutti.
    matched = sum(max(iou(s, v) for v in vj) >= 0.5 for s in scrfd)
    assert matched >= 3


def test_embed_crop_matches_embed(models, group_photo):
    embedder, _ = models
    emb = embedder.embed_crop(embedder.crop(group_photo, select="center"))
    assert np.allclose(emb, embedder.embed(group_photo, select="center").embedding, atol=1e-5)
