"""Test di face/lbp.py: LBP a griglia e distanza chi-quadro.

Non serve il modello: si usano piccole immagini scritte a mano, con il
risultato calcolato a mano nei commenti (gli stessi esempi delle docstring).
"""

import numpy as np
import pytest

from secondlook.face.lbp import (
    UNIFORM_LOOKUP,
    chi2_distance,
    chi2_distance_matrix,
    lbp_codes,
    lbp_descriptor,
    lbp_histogram,
)

# il pixel centrale vale 50; vicini in senso orario da in alto a sinistra:
# 60 20 70 40 50 80 10 90 -> bit 1 0 1 0 1 1 0 1 -> 1 + 4 + 16 + 32 + 128 = 181
PATCH = np.array([[60, 20, 70],
                  [90, 50, 40],
                  [10, 80, 50]], dtype=np.uint8)


def test_code_of_one_pixel_by_hand():
    codes = lbp_codes(PATCH)
    assert codes.shape == (1, 1)            # un'immagine 3 x 3 ha un solo pixel interno
    assert codes[0, 0] == 181


def test_flat_image_gives_all_ones():
    # zona piatta: ogni vicino e' >= del centro -> 11111111 = 255
    codes = lbp_codes(np.full((5, 6), 50, dtype=np.uint8))
    assert codes.shape == (3, 4)            # (5-2) x (6-2)
    assert np.all(codes == 255)


def test_uniform_lookup():
    assert len(set(UNIFORM_LOOKUP.tolist())) == 59      # 58 uniformi + 1 "non uniforme"
    assert UNIFORM_LOOKUP[0] == 0                       # 00000000: 0 cambi
    assert UNIFORM_LOOKUP[255] == 57                    # 11111111: l'ultimo uniforme
    assert UNIFORM_LOOKUP[5] == 58                      # 00000101: 4 cambi -> non uniforme
    assert UNIFORM_LOOKUP[181] == 58                    # il codice di PATCH: 6 cambi


def test_histogram_length_and_cells_sum_to_one():
    rng = np.random.default_rng(0)
    codes = lbp_codes(rng.integers(0, 256, size=(112, 112), dtype=np.uint8))   # 110 x 110
    h = lbp_histogram(codes, grid=7, uniform=True)
    assert h.shape == (7 * 7 * 59,)                     # 2891
    assert np.allclose(h.reshape(49, 59).sum(axis=1), 1.0)
    h256 = lbp_histogram(codes, grid=7, uniform=False)
    assert h256.shape == (7 * 7 * 256,)


def test_descriptor_same_for_bgr_and_gray():
    rng = np.random.default_rng(1)
    gray = rng.integers(0, 256, size=(112, 112), dtype=np.uint8)
    bgr = np.stack([gray, gray, gray], axis=2)          # colore "grigio": B = G = R
    assert np.allclose(lbp_descriptor(bgr), lbp_descriptor(gray))


def test_chi2_by_hand():
    a = [0.5, 0.5, 0.0]
    b = [1.0, 0.0, 0.0]
    # 0.25 / 1.5 + 0.25 / 0.5 + 0 = 0.1667 + 0.5 = 0.6667
    assert chi2_distance(a, b) == pytest.approx(2 / 3)
    assert chi2_distance(b, a) == pytest.approx(2 / 3)  # simmetrica
    assert chi2_distance(a, a) == 0.0


def test_chi2_matrix_matches_pairs_and_blocks():
    rng = np.random.default_rng(2)
    P = rng.random((5, 20)).astype(np.float32)
    G = rng.random((4, 20)).astype(np.float32)
    expected = np.array([[chi2_distance(p, g) for g in G] for p in P])
    for block in (1, 2, 16):                            # il blocco non cambia il risultato
        assert np.allclose(chi2_distance_matrix(P, G, block=block), expected, atol=1e-5)
