"""Test di reliability/margin.py: margine top-1/top-2 e distanza dalla soglia."""

import numpy as np
import pytest

from secondlook.reliability.margin import margin_top2, threshold_distance, top_two


def test_cases_a_and_b_from_guide():
    D = [[0.30, 0.80, 0.85, 0.90, 0.95],          # caso A: netta
         [0.95, 0.30, 0.90, 0.35, 0.32]]          # caso B: ambigua, colonne in disordine
    d1, d2 = top_two(D, ["A", "B", "C", "D", "E"])
    assert np.allclose(d1, [0.30, 0.30]) and np.allclose(d2, [0.80, 0.32])
    assert np.allclose(margin_top2(d1, d2), [0.50, 0.02])


def test_one_distance_per_person():
    # la seconda foto di Mario non deve diventare il "secondo candidato"
    d1, d2 = top_two([[0.30, 0.33, 0.80, 0.85]], ["Mario", "Mario", "Anna", "Luca"])
    assert (d1[0], d2[0]) == (0.30, 0.80)
    assert margin_top2(d1, d2)[0] == pytest.approx(0.50)       # sulle foto sarebbe 0.03


def test_margin_ignores_crowding_after_second_place():
    # stesso margine, ma nel secondo caso c'e' una folla dal terzo posto: il margine non la vede
    D = [[0.30, 0.50, 0.90, 0.95], [0.30, 0.50, 0.52, 0.55]]
    d1, d2 = top_two(D, ["A", "B", "C", "D"])
    assert np.allclose(margin_top2(d1, d2), [0.20, 0.20])


def test_threshold_distance():
    assert np.allclose(threshold_distance([0.30, 0.48, 0.70], 0.5), [0.20, 0.02, -0.20])
