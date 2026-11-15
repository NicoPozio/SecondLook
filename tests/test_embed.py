from pathlib import Path

import cv2
import numpy as np
import pytest

from secondlook.face import cosine_distance

SAMPLE_GROUP_PHOTO = "t1.jpg"


def test_cosine_identical():
    v = np.array([1.0, 2.0, 3.0])
    assert cosine_distance(v, v) == pytest.approx(0.0, abs=1e-6)


def test_cosine_opposite():
    v = np.array([1.0, 0.0])
    assert cosine_distance(v, -v) == pytest.approx(2.0)


def test_cosine_scale_invariant():
    a = np.array([1.0, 2.0])
    b = np.array([2.0, 1.0])
    assert cosine_distance(a, b) == pytest.approx(cosine_distance(10 * a, b))


@pytest.fixture(scope="module")
def embedder():
    from secondlook.face import FaceEmbedder

    try:
        return FaceEmbedder()
    except Exception as exc:
        pytest.skip(f"ArcFace model unavailable: {exc}")


@pytest.fixture(scope="module")
def group_photo():
    import insightface

    path = Path(insightface.__file__).parent / "data" / "images" / SAMPLE_GROUP_PHOTO
    img = cv2.imread(str(path))
    assert img is not None, f"sample photo missing: {path}"
    return img


def _crop(img, bbox, margin=0.6):
    x1, y1, x2, y2 = bbox
    mx, my = int((x2 - x1) * margin), int((y2 - y1) * margin)
    h, w = img.shape[:2]
    return img[max(0, y1 - my):min(h, y2 + my), max(0, x1 - mx):min(w, x2 + mx)].copy()


def _perturb(img):
    """Same face, different 'acquisition': brighter, blurred, rescaled."""
    out = cv2.convertScaleAbs(img, alpha=1.15, beta=20)
    out = cv2.GaussianBlur(out, (3, 3), 0)
    return cv2.resize(out, None, fx=1.4, fy=1.4)


def test_detects_faces(embedder, group_photo):
    faces = embedder.faces(group_photo)
    assert len(faces) >= 3
    for f in faces:
        assert f.embedding.shape == (512,)
        assert np.linalg.norm(f.embedding) == pytest.approx(1.0, abs=1e-3)


def test_same_closer_than_others(embedder, group_photo):
    faces = embedder.faces(group_photo)
    anchor = faces[0]
    probe = embedder.embed(_perturb(_crop(group_photo, anchor.bbox)))

    genuine = cosine_distance(anchor.embedding, probe.embedding)
    impostors = [cosine_distance(f.embedding, probe.embedding) for f in faces[1:]]

    assert genuine < min(impostors)
    assert genuine < 0.4


def test_no_face_raises(embedder):
    blank = np.full((256, 256, 3), 127, dtype=np.uint8)
    with pytest.raises(ValueError):
        embedder.embed(blank)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))