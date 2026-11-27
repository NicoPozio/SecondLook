"""Test di face/quality.py: indici di qualita' SP, SI, SY."""

import numpy as np
import pytest

from secondlook.face.quality import (
    AXIS_X,
    SI_REGIONS,
    SYMMETRIC_PAIRS,
    pitch_norm,
    pose_measures,
    score_illumination,
    score_pose,
    score_symmetry,
)


def _lm106_with_chin(x, y):
    lm = np.zeros((106, 2))
    lm[0] = (x, y)                                          # il mento e' il punto 0
    return lm


def test_pose_frontal_by_hand():
    # occhi (40, 50) e (80, 50), naso (60, 70), mento (60, 120)
    kps = np.array([[40, 50], [80, 50], [60, 70], [45, 90], [75, 90]])
    roll, yaw, r = pose_measures(kps, _lm106_with_chin(60, 120))
    assert roll == 0 and yaw == pytest.approx(0)
    assert r == pytest.approx(20 / 50)                       # 0.4


def test_roll_and_yaw():
    kps = np.array([[40, 50], [80, 90], [60, 70], [45, 90], [75, 90]])   # occhi inclinati di 45 gradi
    assert pose_measures(kps, _lm106_with_chin(60, 120))[0] == pytest.approx(1.0)
    kps = np.array([[40, 50], [80, 50], [50, 70], [45, 90], [75, 90]])   # naso spostato: yaw
    assert pose_measures(kps, _lm106_with_chin(60, 120))[1] > 0


def test_pitch_and_sp_by_hand():
    assert np.allclose(pitch_norm([0.40, 0.50, 1.0], 0.40), [0, 0.25, 1.0])
    assert score_pose(0, 0.1, 0.25) == pytest.approx((1 + 0.9 + 0.75) / 3)    # 0.883


def test_illumination():
    flat = np.full((112, 112), 150, dtype=np.uint8)
    assert score_illumination(flat) == pytest.approx(1.0)
    shadow = flat.copy()
    x1, y1, x2, y2 = SI_REGIONS["guancia_sinistra"]
    shadow[y1:y2, x1:x2] = 60                                # guancia in ombra
    x1, y1, x2, y2 = SI_REGIONS["guancia_destra"]
    shadow[y1:y2, x1:x2] = 180
    # medie [150, 60, 180, 150, 150] -> std 40.7 -> SI = 0.68
    assert score_illumination(shadow) == pytest.approx(1 - np.std([150, 60, 180, 150, 150]) / 127.5)


def test_symmetry():
    rng = np.random.default_rng(0)
    p = rng.random((106, 2)) * 100
    for i, j in SYMMETRIC_PAIRS:                             # rende ogni coppia perfettamente speculare
        p[j] = (2 * AXIS_X - p[i, 0], p[i, 1])
    assert score_symmetry(p) == pytest.approx(1.0)
    p[[j for _, j in SYMMETRIC_PAIRS], 0] += 5               # sposta un lato di 5 pixel
    assert score_symmetry(p) == pytest.approx(1 - 5 / 35.2359, abs=1e-3)


@pytest.fixture(scope="module")
def embedder():
    from secondlook.face import FaceEmbedder

    try:
        return FaceEmbedder(landmarks=True)
    except Exception as exc:
        pytest.skip(f"modello non disponibile: {exc}")


def test_quality_inputs_use_the_same_crop(embedder):
    from pathlib import Path

    import cv2
    import insightface

    from secondlook.face.quality import quality_measures

    img = cv2.imread(str(Path(insightface.__file__).parent / "data" / "images" / "t1.jpg"))
    crop, kps5, lm106 = embedder.quality_inputs(img, select="center")
    assert np.array_equal(crop, embedder.crop(img, select="center"))   # stesso preprocessing
    assert kps5.shape == (5, 2) and lm106.shape == (106, 2)
    q = quality_measures(crop, kps5, lm106)
    assert q.shape == (5,) and np.all(q[[0, 1, 3, 4]] >= 0) and np.all(q[[0, 1, 3, 4]] <= 1)
