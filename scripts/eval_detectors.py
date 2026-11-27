"""Viola-Jones contro SCRFD su LFW (Fase 5, passo 5.7).

Domanda: la qualita' del detector influenza il riconoscimento e l'affidabilita' finale?

Cosa fa:
  1. per ogni foto di LFW (una volta sola, poi in cache in data/lfw/detectors_center.npz):
       - riquadro del volto centrale trovato da SCRFD e da Viola-Jones
       - ArcFace sul ritaglio NON allineato di ognuno dei due riquadri
  2. DETECTION RATE: su quante foto ogni detector trova il volto centrale. LFW non
     ha i riquadri "veri": come riferimento si usa il volto di SCRFD, e un volto di
     Viola-Jones conta come "giusto" se si sovrappone ad esso con IoU >= 0.5
  3. salva gli embedding in due cache, una per variante, nello stesso formato di
     eval_lfw.py, cosi' si possono valutare con gli script che esistono gia':
         data/lfw/embeddings_arcface_noalign_center.npz   riquadro SCRFD, senza allineamento
         data/lfw/embeddings_arcface_vj_center.npz        riquadro Viola-Jones, senza allineamento
  4. stampa e salva data/lfw/results_detectors.json

Dopo, per le due varianti:
    uv run python scripts/eval_lfw.py --method arcface_noalign      (e arcface_vj)
    uv run python scripts/eval_srr.py --method arcface_noalign      (e arcface_vj)

Uso (dalla cartella SecondLook):
    uv run python scripts/eval_detectors.py
Serve Viola-Jones in OpenCV: con OpenCV 5 serve opencv-contrib-python, oppure opencv-python<5.
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from secondlook.eval.datasets import lfw_manifest
from secondlook.eval.matrices import EmbeddingCache, embed_rows
from secondlook.face.detect import iou

IOU_OK = 0.5


def detection_stats(vectors: dict) -> dict:
    """Detection rate dei due detector dai vettori di detector_vector.

    COME LO CALCOLA IL CODICE
        per ogni foto: SCRFD ha trovato un volto se le posizioni 0..3 non sono nan,
        Viola-Jones se non lo sono le posizioni 4..7. Se li hanno trovati entrambi
        si calcola l'IoU tra i due riquadri.
        vj_correct = Viola-Jones trova un volto con IoU >= 0.5 rispetto a quello di SCRFD

    ESEMPIO (3 foto)
        foto 1: SCRFD si, VJ si, IoU 0.72   foto 2: SCRFD si, VJ no   foto 3: SCRFD si, VJ si, IoU 0.10
        scrfd_rate = 3/3   vj_rate = 2/3   vj_correct = 1/3   (la foto 3 e' un altro volto)

    INPUT       : vectors = dizionario sample_id -> vettore di 1032 numeri
    RESTITUISCE : dizionario con conteggi e percentuali
    CHIAMA      : iou
    CHIAMATA DA : main, tests/test_detect.py
    """
    v = np.stack(list(vectors.values()))
    has_s = ~np.isnan(v[:, 0])
    has_v = ~np.isnan(v[:, 4])
    both = has_s & has_v
    ious = np.array([iou(r[0:4], r[4:8]) for r in v[both]])
    n = len(v)
    return {
        "photos": int(n),
        "scrfd_found": int(has_s.sum()), "scrfd_rate": float(has_s.mean()),
        "vj_found": int(has_v.sum()), "vj_rate": float(has_v.mean()),
        "vj_correct": int((ious >= IOU_OK).sum()), "vj_correct_rate": float((ious >= IOU_OK).sum() / n),
        "iou_median": float(np.median(ious)) if ious.size else float("nan"),
        "vj_only": int((has_v & ~has_s).sum()),
    }


def split_caches(det: EmbeddingCache, root: Path) -> None:
    """Dalla cache dei detector, le due cache di embedding nel formato di eval_lfw.py.

    COME LO CALCOLA IL CODICE
        noalign: embedding = posizioni 8..519, foto senza volto SCRFD in failed
        vj:      embedding = posizioni 520..1031, foto senza volto Viola-Jones in failed

    CHIAMA      : EmbeddingCache.save
    CHIAMATA DA : main
    """
    for name, start, box in (("arcface_noalign", 8, 0), ("arcface_vj", 520, 4)):
        out = EmbeddingCache(root / f"embeddings_{name}_center.npz")
        out.vectors, out.failed = {}, set(det.failed)
        for sid, vec in det.vectors.items():
            if np.isnan(vec[box]):
                out.failed.add(sid)
            else:
                out.vectors[sid] = vec[start:start + 512]
        out.save()


def main() -> None:
    """Calcola (o riprende) la cache dei detector, la detection rate e le due cache di embedding.

    CHIAMATA DA : riga di comando
    """
    warnings.filterwarnings("ignore", message="`estimate` is deprecated", category=FutureWarning)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="data/lfw")
    args = parser.parse_args()
    root = Path(args.root)

    rows = lfw_manifest(root)
    det = EmbeddingCache(root / "detectors_center.npz")
    todo = det.missing(rows)
    print(f"confronto detector: {len(rows) - len(todo)} foto gia' in cache, {len(todo)} da calcolare")
    if todo:
        from secondlook.face import FaceEmbedder

        embed_rows(todo, FaceEmbedder(), det, select="center", method="detectors")

    stats = detection_stats(det.vectors)
    split_caches(det, root)
    print(f"\nfoto                     {stats['photos']}")
    print(f"SCRFD trova un volto     {stats['scrfd_found']}  ({stats['scrfd_rate'] * 100:.2f} %)")
    print(f"Viola-Jones trova un volto {stats['vj_found']}  ({stats['vj_rate'] * 100:.2f} %)")
    print(f"  ... sullo stesso volto di SCRFD (IoU >= {IOU_OK})  {stats['vj_correct']}  "
          f"({stats['vj_correct_rate'] * 100:.2f} %)   IoU mediano {stats['iou_median']:.2f}")
    print(f"  ... dove SCRFD non trova niente  {stats['vj_only']}")
    out = root / "results_detectors.json"
    out.write_text(json.dumps(stats, indent=2))
    print(f"\nrisultati salvati in {out}; cache pronte per eval_lfw.py ed eval_srr.py "
          "con --method arcface_noalign e --method arcface_vj")


if __name__ == "__main__":
    main()
