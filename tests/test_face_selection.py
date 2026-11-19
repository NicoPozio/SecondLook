"""Test della scelta del volto quando in una foto ce n'e' piu' di uno.

Non serve il modello: si costruiscono a mano dei FaceSample finti con
riquadri (bbox) scelti apposta. Il caso riproduce quello trovato su LFW:
la persona giusta al centro, un'altra persona piu' grande sul bordo.
"""

import numpy as np
import pytest

from secondlook.face.embed import FaceSample, select_face

IMAGE_SHAPE = (250, 250, 3)        # come le foto di LFW: 250 x 250 pixel


def _face(x1, y1, x2, y2):
    return FaceSample(embedding=np.zeros(512, np.float32),
                      bbox=np.array([x1, y1, x2, y2]), det_score=0.8)


# al centro: riquadro 100 x 120 = 12.000 pixel, centro (125, 130) -> 5 pixel dal centro
CENTRAL = _face(75, 70, 175, 190)
# sul bordo destro: riquadro 110 x 130 = 14.300 pixel, piu' grande, ma lontano dal centro
BORDER = _face(180, 60, 290, 190)


def test_largest_takes_the_biggest_box():
    assert select_face([CENTRAL, BORDER], IMAGE_SHAPE, "largest") is BORDER


def test_center_takes_the_central_box():
    assert select_face([CENTRAL, BORDER], IMAGE_SHAPE, "center") is CENTRAL


def test_single_face_is_the_same_for_both():
    assert select_face([CENTRAL], IMAGE_SHAPE, "largest") is CENTRAL
    assert select_face([CENTRAL], IMAGE_SHAPE, "center") is CENTRAL


def test_unknown_selection_raises():
    with pytest.raises(ValueError):
        select_face([CENTRAL], IMAGE_SHAPE, "random")


# --- con il modello vero: embed() deve dare lo stesso risultato di faces() -------------

@pytest.fixture(scope="module")
def embedder():
    from secondlook.face import FaceEmbedder

    try:
        return FaceEmbedder()
    except Exception as exc:
        pytest.skip(f"ArcFace model unavailable: {exc}")


@pytest.fixture(scope="module")
def group_photo():
    from pathlib import Path

    import cv2
    import insightface

    img = cv2.imread(str(Path(insightface.__file__).parent / "data" / "images" / "t1.jpg"))
    assert img is not None
    return img


@pytest.mark.parametrize("select", ["largest", "center"])
def test_embed_matches_faces(embedder, group_photo, select):
    """embed() calcola l'embedding solo del volto scelto; deve coincidere con
    quello che faces() calcola per lo stesso volto insieme a tutti gli altri."""
    from secondlook.face import cosine_distance

    expected = select_face(embedder.faces(group_photo), group_photo.shape, select)
    got = embedder.embed(group_photo, select=select)
    assert np.array_equal(got.bbox, expected.bbox)
    assert cosine_distance(got.embedding, expected.embedding) == pytest.approx(0.0, abs=1e-5)
