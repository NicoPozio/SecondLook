"""Test delle metriche per i criteri di affidabilita' (eval/metrics.py): curva errore-rifiuto, PRR, AUROC."""

import numpy as np
import pytest

from secondlook.eval.metrics import auroc, error_reject_curve, prr

SCORES = [0.9, 0.8, 0.6, 0.4, 0.2]
CORRECT = [True, True, False, True, False]


def test_error_reject_curve_by_hand():
    r, err = error_reject_curve(SCORES, CORRECT)
    assert np.allclose(r, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    assert np.allclose(err, [0.4, 0.2, 0.2, 0, 0, 0])


def test_prr_by_hand():
    # (0.20 - 0.12) / (0.20 - 0.08) = 0.67
    assert prr(SCORES, CORRECT) == pytest.approx(2 / 3)


def test_prr_perfect_random_and_reversed():
    correct = [True, True, False, False]
    assert prr([0.9, 0.8, 0.2, 0.1], correct) == pytest.approx(1.0)     # oracolo
    assert prr([0.5, 0.5, 0.5, 0.5], correct) == pytest.approx(0.0)     # tutti uguali = a caso
    assert prr([0.1, 0.2, 0.8, 0.9], correct) < 0                        # al contrario: peggio del caso
    assert np.isnan(prr([0.1, 0.2], [True, True]))                       # niente errori da trovare


def test_ties_are_rejected_together():
    # tre risposte con lo stesso valore: la curva passa dritta da (0, 2/3) a (1, 0)
    r, err = error_reject_curve([0.5, 0.5, 0.5], [True, False, False])
    assert np.allclose(r, [0, 1]) and np.allclose(err, [2 / 3, 0])


def test_auroc_by_hand_and_ties():
    assert auroc([0.9, 0.8, 0.4], [0.6, 0.2]) == pytest.approx(5 / 6)
    assert auroc([0.5], [0.5]) == 0.5                                     # pareggio = 1/2
    assert auroc([1.0, 0.9], [0.1, 0.0]) == 1.0
