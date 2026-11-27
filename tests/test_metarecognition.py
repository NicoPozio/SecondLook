"""Test di reliability/metarecognition.py: meta-recognition di Weibull."""

import numpy as np
import pytest

from secondlook.reliability.metarecognition import weibull_score, weibull_scores

CASE_A = [0.30, 0.80, 0.85, 0.90, 0.95]       # primo lontanissimo dalla coda
CASE_B = [0.30, 0.32, 0.35, 0.90, 0.95]       # la coda arriva fino al primo


def test_cases_a_and_b():
    a, b = weibull_score(CASE_A), weibull_score(CASE_B)
    assert a == pytest.approx(8.51, abs=0.05)              # p circa 3e-09
    assert b == pytest.approx(0.92, abs=0.05)              # p circa 0.12
    assert a > b


def test_closer_first_is_more_reliable():
    tail = [0.80, 0.85, 0.90, 0.95, 0.97]
    scores = [weibull_score([d1] + tail) for d1 in (0.1, 0.4, 0.7)]
    assert scores[0] > scores[1] > scores[2]


def test_only_the_tail_is_used():
    # con tail_size=3 la quarta e la quinta distanza non contano
    base = weibull_score([0.3, 0.8, 0.85, 0.9, 0.95], tail_size=3)
    assert weibull_score([0.3, 0.8, 0.85, 0.9, 5.0], tail_size=3) == pytest.approx(base)


def test_matrix_uses_one_distance_per_person():
    # Mario ha 2 template: la sua seconda foto non deve finire nella coda
    D = [[0.30, 0.31, 0.80, 0.85, 0.90, 0.95]]
    ids = ["Mario", "Mario", "Anna", "Luca", "Sara", "Piero"]
    assert weibull_scores(D, ids)[0] == pytest.approx(weibull_score(CASE_A))


def test_needs_enough_people_and_sorted_input():
    with pytest.raises(ValueError):
        weibull_score([0.3, 0.8, 0.9])                     # solo 2 di coda
    with pytest.raises(ValueError):
        weibull_score([0.9, 0.3, 0.8, 0.85])
    assert weibull_score([0.3, 0.8, 0.8, 0.8]) == 0.0       # coda tutta uguale: niente fit
