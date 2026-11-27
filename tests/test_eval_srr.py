"""Test di scripts/eval_srr.py: divisione delle persone e conteggio degli errori dell'SRR."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from eval_srr import criteria_scores, evaluate_kind, quality_scores, split_subjects, threshold_free  # noqa: E402


def test_split_is_half_by_person_and_repeatable():
    ids = ["Anna", "Bruno", "Carla", "Dario", "Anna", "Elena", "Fabio"]   # Anna ripetuta
    tuning = split_subjects(ids)
    assert len(tuning) == 3                         # 6 persone diverse -> meta' = 3
    assert tuning <= set(ids)
    assert split_subjects(ids) == tuning            # stesso seme -> stessa divisione


def test_evaluate_kind_by_hand():
    # tuning: giuste [0.9, 0.6], sbagliate [0.2, 0.4] -> phi_bar = 0.4 (nessun errore)
    # test:   giuste [0.9, 0.8, 0.3], sbagliate [0.1 (registrato), 0.7 (sconosciuto)]
    phi_known = np.array([0.9, 0.6, 0.2, 0.9, 0.8, 0.3, 0.1])
    correct = np.array([True, True, False, True, True, True, False])
    known_tuning = np.array([True, True, True, False, False, False, False])
    phi_unknown = np.array([0.4, 0.7])
    unknown_tuning = np.array([True, False])
    r = evaluate_kind(phi_known, correct, known_tuning, phi_unknown, unknown_tuning)
    assert r["critical"] == 0.4
    assert r["errors_let_through"] == pytest.approx(1 / 2)   # 0.7 > 0.4
    assert r["useless_step_up"] == pytest.approx(1 / 3)      # 0.3 <= 0.4
    assert r["step_up_rate"] == pytest.approx(2 / 5)         # 0.3 e 0.1


def test_criteria_scores_case_b():
    # caso B della guida: 0.30, 0.32, 0.35, 0.90, 0.95
    sc = criteria_scores([[0.30, 0.32, 0.35, 0.90, 0.95]], ["A", "B", "C", "D", "E"])
    assert sc["relative"][0] == pytest.approx(0.02 / 0.95)
    assert sc["density"][0] == 0.5
    assert sc["margin"][0] == pytest.approx(0.02)
    assert sc["threshold"][0] == pytest.approx(-0.30)
    assert sc["weibull"][0] == pytest.approx(0.92, abs=0.05)


def test_margin_has_no_srr_medians():
    r = evaluate_kind(np.array([0.5, 0.1, 0.6]), np.array([True, False, True]),
                      np.array([True, True, False]), np.array([0.05, 0.2]), np.array([True, False]),
                      normalize=False)
    assert "srr_median_correct" not in r


def test_threshold_free_by_hand():
    # test: giuste [0.9, 0.8, 0.4], uno scambio [0.6], uno sconosciuto [0.2]; il tuning (0.5) e' escluso
    phi_known = np.array([0.5, 0.9, 0.8, 0.6, 0.4])
    correct = np.array([True, True, True, False, True])
    known_tuning = np.array([True, False, False, False, False])
    r = threshold_free(phi_known, correct, known_tuning, np.array([0.2]), np.array([False]))
    assert r["prr"] == pytest.approx(2 / 3)
    assert r["auroc_all"] == pytest.approx(5 / 6)
    assert r["auroc_swap"] == pytest.approx(2 / 3)       # 0.9, 0.8 > 0.6; 0.4 < 0.6
    assert r["auroc_unknown"] == 1.0
    assert (r["n_swap"], r["n_unknown"]) == (1, 1)
    assert r["curve"]["error"][0] == pytest.approx(0.4) and r["curve"]["error"][-1] == 0.0


def test_quality_scores_r0_from_tuning():
    # [roll, yaw, r, SI, SY]; r0 = mediana di r sul tuning = mediana(0.40, 0.44, 0.48) = 0.44
    q_known = np.array([[0, 0.05, 0.40, 0.9, 0.95], [0, 0.05, 0.44, 0.8, 0.90], [0, 0.05, 0.44, 0.7, 0.80]])
    q_unknown = np.array([[0, 0.05, 0.48, 0.6, 0.70]])
    known_tuning = np.array([True, True, False])
    qk, qu, r0 = quality_scores(q_known, q_unknown, known_tuning, np.array([True]))
    assert r0 == pytest.approx(0.44)
    assert qk["sp"][2] == pytest.approx((1 + 0.95 + 1) / 3)       # pitch 0 -> SP = 0.983
    assert np.allclose(qk["si"], [0.9, 0.8, 0.7]) and np.allclose(qu["sy"], [0.70])
