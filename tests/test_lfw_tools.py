"""Test di datasets.py e matrices.py su una "mini-LFW" finta creata al volo.

Non serve scaricare LFW: si crea in una cartella temporanea la stessa
struttura (una cartella per persona, immagini numerate, pairs.txt).
"""

import numpy as np
import pytest

from secondlook.eval.datasets import (
    MANIFEST_COLUMNS,
    lfw_manifest,
    read_lfw_pairs,
    read_manifest,
    split_gallery_probes,
    write_manifest,
)
from secondlook.eval.matrices import EmbeddingCache, cosine_distance_matrix

PEOPLE = {"Anna_Rossi": 3, "Bruno_Verdi": 2, "Carla_Bianchi": 1}


@pytest.fixture()
def mini_lfw(tmp_path):
    for name, n in PEOPLE.items():
        d = tmp_path / "lfw" / name
        d.mkdir(parents=True)
        for k in range(1, n + 1):
            (d / f"{name}_{k:04d}.jpg").write_bytes(b"")      # file vuoti: bastano per la struttura
    # 1 gruppo, 1 coppia per tipo
    (tmp_path / "pairs.txt").write_text(
        "1\t1\n"
        "Anna_Rossi\t1\t3\n"
        "Bruno_Verdi\t2\tCarla_Bianchi\t1\n"
    )
    return tmp_path


def test_manifest_has_common_columns(mini_lfw):
    rows = lfw_manifest(mini_lfw)
    assert len(rows) == 6
    assert list(rows[0]) == MANIFEST_COLUMNS
    assert rows[0]["subject_id"] == "Anna_Rossi"
    assert rows[0]["sample_id"] == "Anna_Rossi_0001"
    assert {r["source"] for r in rows} == {"lfw"}


def test_manifest_roundtrip(mini_lfw, tmp_path):
    rows = lfw_manifest(mini_lfw)
    write_manifest(rows, tmp_path / "out" / "manifest.csv")
    assert read_manifest(tmp_path / "out" / "manifest.csv") == rows


def test_pairs(mini_lfw):
    pairs = read_lfw_pairs(mini_lfw)
    assert len(pairs) == 2
    assert pairs[0].same and pairs[0].sample_a == "Anna_Rossi_0001" and pairs[0].sample_b == "Anna_Rossi_0003"
    assert not pairs[1].same and pairs[1].sample_b == "Carla_Bianchi_0001"
    assert {p.fold for p in pairs} == {0}


def test_split_gallery_probes(mini_lfw):
    gallery, known, unknown = split_gallery_probes(lfw_manifest(mini_lfw))
    assert [r["sample_id"] for r in gallery] == ["Anna_Rossi_0001", "Bruno_Verdi_0001"]
    assert [r["sample_id"] for r in known] == ["Anna_Rossi_0002", "Anna_Rossi_0003", "Bruno_Verdi_0002"]
    assert [r["sample_id"] for r in unknown] == ["Carla_Bianchi_0001"]


def test_cosine_distance_matrix_by_hand():
    probes = np.array([[3.0, 4.0], [1.0, 0.0]])
    gallery = np.array([[4.0, 3.0], [0.0, 2.0]])
    d = cosine_distance_matrix(probes, gallery)
    # (3,4)/5 . (4,3)/5 = 0.96 -> 0.04 ;  (3,4)/5 . (0,1) = 0.8 -> 0.2
    # (1,0)   . (0.8,0.6) = 0.8 -> 0.2 ;  (1,0)   . (0,1) = 0   -> 1.0
    assert d == pytest.approx(np.array([[0.04, 0.2], [0.2, 1.0]]), abs=1e-6)


def test_embedding_cache_roundtrip(tmp_path):
    cache = EmbeddingCache(tmp_path / "emb.npz")
    cache.vectors["a"] = np.ones(512, np.float32)
    cache.failed.add("b")
    cache.save()
    again = EmbeddingCache(tmp_path / "emb.npz")
    assert again.get(["a"]).shape == (1, 512)
    assert again.failed == {"b"}
    rows = [{"sample_id": s} for s in ["a", "b", "c"]]
    assert [r["sample_id"] for r in again.missing(rows)] == ["c"]
