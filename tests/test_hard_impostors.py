"""Test di scripts/eval_hard_impostors.py: il caso "impostore molto simile, foto ottima"."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from eval_hard_impostors import excellent_photo, let_through  # noqa: E402


def test_excellent_photo_by_hand():
    # tuning: i percentili 75 sono SP 0.8, SI 0.8, SY 0.9
    q_tuning = {"sp": np.array([0.5, 0.6, 0.7, 0.8, 0.9]),
                "si": np.array([0.5, 0.6, 0.7, 0.8, 0.9]),
                "sy": np.array([0.6, 0.7, 0.8, 0.9, 1.0])}
    q_test = {"sp": np.array([0.85, 0.85, 0.70]),
              "si": np.array([0.90, 0.90, 0.95]),
              "sy": np.array([0.95, 0.85, 0.99])}
    mask, cut = excellent_photo(q_tuning, q_test)
    assert cut == pytest.approx({"sp": 0.8, "si": 0.8, "sy": 0.9})
    # foto 1: tutti sopra; foto 2: SY sotto; foto 3: SP sotto
    assert mask.tolist() == [True, False, False]


def test_let_through_by_hand():
    assert let_through(np.array([0.1, 0.7, 0.9]), 0.5) == pytest.approx(2 / 3)
    assert let_through(np.array([0.5]), 0.5) == 0.0           # uguale al critico: step-up, come in eval_srr
    assert np.isnan(let_through(np.array([]), 0.5))           # nessuna risposta: non definito
