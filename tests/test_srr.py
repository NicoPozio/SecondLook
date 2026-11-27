"""Test di reliability/srr.py: devono riprodurre gli esempi a mano degli appunti e della guida."""

import numpy as np
import pytest

from secondlook.reliability.srr import critical_phi, phi_density, phi_matrix, phi_relative, srr

# caso A e caso B della guida del progetto (gallery di 5 persone, phi_bar = 0.6)
CASE_A = [0.30, 0.80, 0.85, 0.90, 0.95]       # risposta netta
CASE_B = [0.30, 0.32, 0.35, 0.90, 0.95]       # risposta ambigua


def test_relative_distance_example_from_notes():
    # (0.25 - 0.15) / 0.45 = 0.10 / 0.45 = 0.22
    assert phi_relative([0.15, 0.25, 0.28, 0.45]) == pytest.approx(0.10 / 0.45)


def test_density_ratio_example_from_notes():
    # nuvola < 2 * 0.15 = 0.30: dentro solo 0.25 -> 1 - 1/3 = 0.67
    assert phi_density([0.15, 0.25, 0.31, 0.45]) == pytest.approx(2 / 3)


def test_case_a_and_b_from_guide():
    assert phi_relative(CASE_A) == pytest.approx(0.50 / 0.95)     # 0.53
    assert phi_relative(CASE_B) == pytest.approx(0.02 / 0.95)     # 0.02
    assert phi_density(CASE_A) == 1.0                              # nessuno sotto 0.60
    assert phi_density(CASE_B) == 0.5                              # 0.32 e 0.35 -> 1 - 2/4
    assert srr(1.0, 0.6) == pytest.approx(1.0)                     # caso A: +1.00
    assert srr(0.5, 0.6) == pytest.approx(-0.1 / 0.6)              # caso B: -0.17


def test_first_candidate_is_excluded_from_the_cloud():
    # se il primo fosse contato, sarebbe sempre nella nuvola e phi non arriverebbe mai a 1
    assert phi_density([0.2, 0.9, 0.95]) == 1.0


def test_srr_range_and_fixed_points():
    phi = np.array([0.0, 0.6, 1.0])
    assert np.allclose(srr(phi, 0.6), [-1.0, 0.0, 1.0])
    assert srr(0.0, 0.0) == 0.0                                     # S = 0: niente divisione per zero


def test_distances_must_be_sorted_and_at_least_two():
    with pytest.raises(ValueError):
        phi_density([0.5, 0.2, 0.9])
    with pytest.raises(ValueError):
        phi_relative([0.3])


def test_phi_matrix_matches_single_functions():
    # tre probe, colonne in ordine qualsiasi: phi_matrix ordina da solo
    rng = np.random.default_rng(0)
    D = rng.random((3, 6))
    ids = ["A", "B", "C", "D", "E", "F"]
    for kind, f in (("relative", phi_relative), ("density", phi_density)):
        expected = [f(np.sort(row)) for row in D]
        assert np.allclose(phi_matrix(D, ids, kind), expected)


def test_phi_matrix_uses_one_distance_per_person():
    # Mario ha 2 template: sulle foto la seconda di Mario (0.33) riempirebbe la nuvola
    D = [[0.30, 0.33, 0.80, 0.85, 0.90]]
    ids = ["Mario", "Mario", "Anna", "Luca", "Sara"]
    assert phi_density(sorted(D[0])) == 0.75                       # sulle foto: falsa ambiguita'
    assert phi_matrix(D, ids, "density")[0] == 1.0                  # per persona: risposta netta


def test_critical_phi_by_hand():
    # soglia 0.5: e_wrong = 1/4 (0.75), e_correct = 1/4 (0.4); le altre soglie fanno peggio
    phi_bar, e_wrong, e_correct = critical_phi([0.9, 0.8, 0.7, 0.4], [0.2, 0.3, 0.5, 0.75])
    assert (phi_bar, e_wrong, e_correct) == (0.5, 0.25, 0.25)


def test_critical_phi_perfect_separation():
    # giuste tutte sopra, sbagliate tutte sotto: nessun errore, phi_bar = la sbagliata piu' alta
    assert critical_phi([0.8, 0.9], [0.1, 0.3]) == (0.3, 0.0, 0.0)
