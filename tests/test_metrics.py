"""Test delle metriche: ogni valore atteso e' calcolato a mano (vedi la spiegazione nel .md)."""

import numpy as np
import pytest

from secondlook.eval.metrics import (
    cms_curve,
    decidability,
    dir_at_false_alarm,
    eer,
    far_frr,
    open_set_rates,
    roc_curve,
)

# Esempio di verifica usato in tutti i test: distanze genuine e impostore
GENUINE = [0.1, 0.2, 0.3, 0.6]
IMPOSTOR = [0.4, 0.7, 0.8, 0.9]


def test_far_frr_by_hand():
    _, far, frr = far_frr(GENUINE, IMPOSTOR, thresholds=[0.05, 0.3, 0.4, 0.65, 1.0])
    #             t:   0.05  0.3   0.4   0.65  1.0
    assert far.tolist() == pytest.approx([0.0, 0.0, 0.25, 0.25, 1.0])
    assert frr.tolist() == pytest.approx([1.0, 0.25, 0.25, 0.0, 0.0])


def test_roc_gar_is_one_minus_frr():
    _, far, gar = roc_curve(GENUINE, IMPOSTOR, thresholds=[0.4])
    assert far[0] == pytest.approx(0.25)
    assert gar[0] == pytest.approx(0.75)


def test_eer_exact_crossing():
    rate, threshold = eer(GENUINE, IMPOSTOR)
    assert rate == pytest.approx(0.25)
    assert threshold == pytest.approx(0.4)


def test_eer_perfect_separation():
    rate, threshold = eer([0.1, 0.2], [0.8, 0.9])
    assert rate == pytest.approx(0.0)
    assert 0.2 <= threshold < 0.8


def test_eer_interpolated():
    # t=0.2: FAR=0, FRR=1/2  -> FAR-FRR=-0.5 ; t=0.6: FAR=1/2, FRR=0 -> +0.5
    # a meta' strada tra le due soglie: soglia 0.4, EER 0.25
    rate, threshold = eer([0.2, 0.6], [0.6, 0.9])
    assert rate == pytest.approx(0.25)
    assert threshold == pytest.approx(0.4)


def test_decidability_by_hand():
    # medie 0.2 e 0.8, varianze entrambe 0.02/3 -> d' = 0.6 / sqrt(0.02/3) = 7.348...
    assert decidability([0.1, 0.2, 0.3], [0.7, 0.8, 0.9]) == pytest.approx(7.3485, abs=1e-3)


def test_cms_by_hand():
    #             A     B     C        identita' giusta -> rango
    dist = [[0.2, 0.5, 0.7],   # P1 e' A: A e' il piu' vicino          -> 1
            [0.3, 0.4, 0.9],   # P2 e' B: A (0.3) batte B (0.4)        -> 2
            [0.6, 0.5, 0.8]]   # P3 e' C: B e A battono C              -> 3
    cms = cms_curve(dist, probe_ids=["A", "B", "C"], gallery_ids=["A", "B", "C"])
    assert cms.tolist() == pytest.approx([1 / 3, 2 / 3, 1.0])


def test_cms_multiple_templates_take_closest():
    # due template di A: conta il piu' vicino (0.1), quindi P1 torna al rango 1
    dist = [[0.9, 0.1, 0.5]]
    cms = cms_curve(dist, probe_ids=["A"], gallery_ids=["A", "A", "B"])
    assert cms[0] == pytest.approx(1.0)


def test_cms_rejects_unknown_probe():
    with pytest.raises(ValueError):
        cms_curve([[0.1, 0.2]], probe_ids=["Z"], gallery_ids=["A", "B"])


def test_open_set_by_hand():
    gallery = ["A", "B"]
    known = [[0.2, 0.6],     # P1 e' A: rango 1, distanza 0.2
             [0.3, 0.5]]     # P2 e' B: rango 2 (A e' piu' vicino), distanza 0.5
    unknown = [[0.45, 0.9]]  # P3 non registrato: distanza minima 0.45
    t, dir1, fa = open_set_rates(known, ["A", "B"], unknown, gallery, [0.25, 0.5], rank=1)
    assert dir1.tolist() == pytest.approx([0.5, 0.5])
    assert fa.tolist() == pytest.approx([0.0, 1.0])
    _, dir2, _ = open_set_rates(known, ["A", "B"], unknown, gallery, [0.25, 0.5], rank=2)
    assert dir2.tolist() == pytest.approx([0.5, 1.0])


def test_dir_at_false_alarm_zero():
    gallery = ["A", "B"]
    known = [[0.2, 0.6], [0.3, 0.5]]
    unknown = [[0.45, 0.9], [0.8, 0.7]]
    rate, threshold, fa = dir_at_false_alarm(known, ["A", "B"], unknown, gallery, target_fa=0.0)
    assert fa == 0.0 and threshold < 0.45
    assert rate == pytest.approx(0.5)       # solo P1 (0.2) passa
    rate, threshold, fa = dir_at_false_alarm(known, ["A", "B"], unknown, gallery, target_fa=0.5)
    assert threshold == pytest.approx(0.45) and fa == pytest.approx(0.5)


def test_large_random_is_consistent():
    rng = np.random.default_rng(0)
    g = rng.normal(0.3, 0.1, 2000)
    i = rng.normal(0.9, 0.1, 20000)
    rate, thr = eer(g, i)
    assert 0.0 <= rate < 0.01 and 0.5 < thr < 0.7
