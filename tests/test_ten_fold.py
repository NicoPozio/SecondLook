"""Test di ten_fold_accuracy (scripts/eval_lfw.py): l'accuratezza con il protocollo di LFW.

Per ogni gruppo la soglia si sceglie sugli ALTRI gruppi e si misura sul gruppo
escluso. Non serve il modello ne' LFW: bastano piccoli array scritti a mano.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# eval_lfw.py e' uno script, non fa parte del pacchetto: lo si rende importabile
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from eval_lfw import ten_fold_accuracy  # noqa: E402

G, I = True, False          # G = coppia genuina, I = coppia impostore

# l'esempio della docstring: 2 gruppi da 2 coppie
DIST = np.array([0.1, 0.9, 0.3, 0.6])
SAME = np.array([G, I, G, I])
FOLDS = np.array([0, 0, 1, 1])


def test_example_by_hand():
    # gruppo 0 escluso: soglia scelta sul gruppo 1 -> 0.3 -> sul gruppo 0 tutto giusto  -> 1.0
    # gruppo 1 escluso: soglia scelta sul gruppo 0 -> 0.1 -> 0.3 (genuina) rifiutata   -> 0.5
    # media (1.0 + 0.5) / 2 = 0.75; deviazione standard 0.25
    mean, std = ten_fold_accuracy(DIST, SAME, FOLDS)
    assert mean == pytest.approx(0.75)
    assert std == pytest.approx(0.25)


def test_threshold_is_not_chosen_on_the_test_fold():
    # Ogni gruppo, preso da solo, si separa perfettamente:
    #   gruppo 0: genuina 0.1 < impostore 0.9  -> una soglia in mezzo da' 2/2
    #   gruppo 1: genuina 0.3 < impostore 0.6  -> una soglia in mezzo da' 2/2
    # Se la soglia fosse scelta guardando il gruppo d'esame ("sbirciando"), il
    # risultato sarebbe 1.0. Invece viene scelta sull'ALTRO gruppo, e per il
    # gruppo 1 la soglia 0.1 (scelta sul gruppo 0) rifiuta la genuina 0.3:
    # il risultato deve essere minore di 1.
    for f in (0, 1):
        d, s = DIST[FOLDS == f], SAME[FOLDS == f]
        assert d[s].max() < d[~s].min()             # il gruppo da solo e' separabile
    mean, _ = ten_fold_accuracy(DIST, SAME, FOLDS)
    assert mean < 1.0


def test_perfect_separation_gives_one():
    # 3 gruppi; in ognuno i genuini (0.1, 0.2) sono tutti sotto gli impostori (0.8, 0.9)
    dist = np.array([0.1, 0.2, 0.8, 0.9] * 3)
    same = np.array([G, G, I, I] * 3)
    folds = np.repeat([0, 1, 2], 4)
    mean, std = ten_fold_accuracy(dist, same, folds)
    assert mean == 1.0 and std == 0.0


def test_order_of_pairs_does_not_matter():
    # mescolare le coppie (tenendo insieme dist, same e fold di ognuna) non cambia niente
    rng = np.random.default_rng(0)
    dist = rng.random(60)
    same = rng.random(60) < 0.5
    folds = np.repeat(np.arange(10), 6)
    perm = rng.permutation(60)
    assert ten_fold_accuracy(dist, same, folds) == pytest.approx(
        ten_fold_accuracy(dist[perm], same[perm], folds[perm]))
