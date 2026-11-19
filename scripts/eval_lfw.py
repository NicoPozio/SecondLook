"""Valutazione offline di ArcFace su LFW (Fase 2 della pipeline).

Fa tre cose:
  1. VERIFICA 1:1 sul protocollo ufficiale di LFW (6000 coppie in 10 gruppi):
     EER, soglia dell'EER, d' e accuratezza con la validazione a 10 gruppi.
  2. IDENTIFICAZIONE CLOSED-SET: gallery = prima immagine di ogni persona con
     almeno 2 immagini, probe = le altre. Recognition Rate, CMS(5), CMS(10).
  3. IDENTIFICAZIONE OPEN-SET: come sopra, piu' i probe delle persone con una
     sola immagine come sconosciuti. DIR al rango 1 con false alarm all'1% e al 10%.

Uso (dalla cartella SecondLook):
    uv run python scripts/eval_lfw.py --quick     # prova veloce, pochi minuti
    uv run python scripts/eval_lfw.py             # tutto LFW, circa mezz'ora o piu'

LFW viene scaricato in data/lfw (esclusa da git). Gli embedding vengono salvati
in data/lfw/embeddings_arcface_center.npz: se si interrompe con Ctrl+C, al
riavvio riparte da dove era arrivato. Se in una foto ci sono piu' volti si usa
quello piu' vicino al centro, perche' in LFW la persona etichettata e' centrata.
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from secondlook.eval.datasets import (
    download_lfw,
    lfw_manifest,
    read_lfw_pairs,
    split_gallery_probes,
    write_manifest,
)
from secondlook.eval.matrices import EmbeddingCache, cosine_distance_matrix, embed_rows
from secondlook.eval.metrics import cms_curve, decidability, dir_at_false_alarm, eer


def ten_fold_accuracy(dist: np.ndarray, same: np.ndarray, folds: np.ndarray) -> tuple[float, float]:
    """Accuratezza con il protocollo ufficiale di LFW (validazione a 10 gruppi).

    TEORIA
        Accuratezza = quota di coppie decise giuste:
            genuina con distanza <= t (accettata)  oppure
            impostore con distanza  > t (rifiutato).
        Il problema: se la soglia t si sceglie guardando le stesse coppie
        su cui poi si misura, il risultato e' troppo ottimista (la soglia
        "conosce" gia' le risposte). Allora, per ogni gruppo f:
            1. si sceglie t sugli ALTRI 9 gruppi (quella con accuratezza massima)
            2. si misura l'accuratezza sul gruppo f, che t non ha mai visto
        Il risultato e' la media (e la deviazione standard) dei 10 valori.
        E' il numero che si confronta con gli articoli (ArcFace: > 99.5%).

    COME LO CALCOLA IL CODICE
        candidates = tutte le distanze osservate (come in metrics.default_thresholds)
        per ogni gruppo f:
            train = maschera delle coppie NON in f, test = quelle in f
            (dist[train] <= t) == same[train]  -> True dove la decisione e' giusta
            .mean() -> quota di giuste; argmax -> la soglia migliore
            poi la stessa formula sul test con quella soglia

    ESEMPIO (2 gruppi da 2 coppie invece di 10 da 600)
        dist  = [0.1, 0.9, 0.3, 0.6]
        same  = [ G,   I,   G,   I ]       G = genuina (True), I = impostore
        folds = [ 0,   0,   1,   1 ]
        gruppo 0 escluso: soglia scelta sul gruppo 1 ([0.3 G, 0.6 I]) -> t = 0.3
                          sul gruppo 0: 0.1 <= 0.3 accetta G giusto,
                          0.9 > 0.3 rifiuta I giusto -> 2/2 = 1.0
        gruppo 1 escluso: soglia scelta sul gruppo 0 ([0.1 G, 0.9 I]) -> t = 0.1
                          sul gruppo 1: 0.3 > 0.1 rifiuta G SBAGLIATO,
                          0.6 > 0.1 rifiuta I giusto -> 1/2 = 0.5
        -> (media 0.75, deviazione standard 0.25)

    INPUT (tre array allineati: posizione k = coppia k di pairs.txt)
        dist  : distanza coseno di ogni coppia
        same  : True se la coppia e' della stessa persona (Pair.same)
        folds : gruppo 0..9 di ogni coppia (Pair.fold)
    RESTITUISCE : (accuratezza media, deviazione standard), due float in [0, 1]
    CHIAMA      : np.unique, np.argmax, np.mean, np.std
    CHIAMATA DA : main (solo se ci sono piu' gruppi, cioe' non con --quick)
    """
    accs = []
    candidates = np.unique(dist)                    # soglie da provare
    for f in np.unique(folds):                      # f = 0, 1, ..., 9
        train, test = folds != f, folds == f        # maschere True/False
        train_acc = [((dist[train] <= t) == same[train]).mean() for t in candidates]
        t_best = candidates[int(np.argmax(train_acc))]   # a parita' vince la soglia piu' bassa
        accs.append(((dist[test] <= t_best) == same[test]).mean())
    return float(np.mean(accs)), float(np.std(accs))


def main() -> None:
    """Esegue tutta la valutazione e stampa/salva i risultati.

    PASSI (e funzioni usate per ognuno)
        0. dati:          download_lfw, lfw_manifest, write_manifest,
                          read_lfw_pairs, split_gallery_probes
        1. embedding:     EmbeddingCache, embed_rows (-> FaceEmbedder.embed)
        2. verifica:      distanza coppia per coppia, eer, decidability,
                          ten_fold_accuracy
        3. closed-set:    cosine_distance_matrix, cms_curve
        4. open-set:      cosine_distance_matrix, dir_at_false_alarm
        5. salvataggio:   results_arcface(_quick).json

    DOVE SI SEPARANO GENUINI/IMPOSTORI E DOVE ENTRANO I NOMI
        - verifica: la lista `same` (da pairs.txt) divide dist in
          dist[same] (genuini) e dist[~same] (impostori), poi passati a eer
        - identificazione: le matrici D_known/D_unknown sono solo numeri; i nomi
          delle righe (k_ids) e delle colonne (g_ids) vengono dal manifest, nello
          stesso ordine con cui si sono costruite le matrici

    CHIAMATA DA : riga di comando (uv run python scripts/eval_lfw.py [--quick])
    """
    # avviso interno di insightface (usa una funzione di scikit-image dichiarata superata):
    # non riguarda il nostro codice, quindi lo si nasconde come fa gia' pytest
    warnings.filterwarnings("ignore", message="`estimate` is deprecated", category=FutureWarning)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="data/lfw", help="cartella di LFW (default: data/lfw)")
    parser.add_argument("--quick", action="store_true",
                        help="prova veloce: un solo gruppo di coppie e 100 persone per l'identificazione")
    args = parser.parse_args()

    # --- 0. dati ------------------------------------------------------------
    root = download_lfw(args.root)                        # scarica solo la prima volta
    rows = lfw_manifest(root)                             # 13.233 righe, una per foto
    write_manifest(rows, Path(root) / "manifest.csv")     # copia leggibile su disco
    by_id = {r["sample_id"]: r for r in rows}             # sample_id -> riga, es. by_id["Kate_Winslet_0002"]
    print(f"LFW: {len(rows)} immagini di {len({r['subject_id'] for r in rows})} persone")

    pairs = read_lfw_pairs(root)                          # 6000 Pair
    if args.quick:
        pairs = [p for p in pairs if p.fold == 0]         # solo le prime 600

    gallery, known, unknown = split_gallery_probes(rows)  # 1680 / 7484 / 4069 righe
    if args.quick:
        # prime 100 persone della gallery con i loro probe, e 100 sconosciuti
        keep = {r["subject_id"] for r in gallery[:100]}
        gallery = [r for r in gallery if r["subject_id"] in keep]
        known = [r for r in known if r["subject_id"] in keep]
        unknown = unknown[:100]

    # --- 1. embedding (solo delle immagini che servono) --------------------
    # insieme (set) dei sample_id usati da coppie e identificazione; | = unione
    needed = {p.sample_a for p in pairs} | {p.sample_b for p in pairs}
    needed |= {r["sample_id"] for r in gallery + known + unknown}
    from secondlook.face import FaceEmbedder

    # In LFW la persona etichettata e' sempre al centro della foto, ma ai bordi
    # possono comparire volti di altre persone anche piu' grandi: si prende quindi
    # il volto piu' vicino al centro. La cache ha un nome diverso da quella fatta
    # con il volto piu' grande, cosi' i due tipi di embedding non si mescolano.
    cache = EmbeddingCache(Path(root) / "embeddings_arcface_center.npz")
    todo = cache.missing([by_id[s] for s in sorted(needed)])   # righe ancora da calcolare
    print(f"embedding: {len(needed) - len(todo)} gia' in cache, {len(todo)} da calcolare")
    if todo:
        embed_rows(todo, FaceEmbedder(), cache, select="center")
    results = {"quick": args.quick, "face_selection": "center",
               "faces_not_detected": len(cache.failed & needed)}

    # --- 2. verifica sulle coppie ufficiali --------------------------------
    # coppie in cui entrambe le foto hanno un volto rilevato
    usable = [p for p in pairs if p.sample_a in cache.vectors and p.sample_b in cache.vectors]
    a = cache.get([p.sample_a for p in usable])      # riga k = prima foto della coppia k   (N x 512)
    b = cache.get([p.sample_b for p in usable])      # riga k = seconda foto della coppia k (N x 512)
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    # qui NON serve la matrice: si confronta solo la riga k di a con la riga k di b.
    # (a * b).sum(axis=1) = prodotto scalare riga per riga -> N numeri
    dist = 1.0 - (a * b).sum(axis=1)                 # distanza coseno coppia per coppia
    same = np.array([p.same for p in usable])        # es. [True, True, ..., False, False]
    folds = np.array([p.fold for p in usable])
    # dist[same] = solo le distanze delle coppie genuine, dist[~same] (~ = "non") le impostore
    rate, thr = eer(dist[same], dist[~same])
    # l'accuratezza a 10 gruppi ha senso solo se ci sono piu' gruppi (non in --quick)
    acc_mean, acc_std = ten_fold_accuracy(dist, same, folds) if len(set(folds)) > 1 else (None, None)
    results["verification"] = {
        "pairs_used": len(usable), "pairs_skipped": len(pairs) - len(usable),
        "eer": rate, "eer_threshold": thr,
        "d_prime": decidability(dist[same], dist[~same]),
        "genuine_mean": float(dist[same].mean()), "impostor_mean": float(dist[~same].mean()),
        "accuracy_mean": acc_mean, "accuracy_std": acc_std,
    }

    # --- 3. e 4. identificazione --------------------------------------------
    # tiene solo le righe con un volto rilevato
    ok = lambda rs: [r for r in rs if r["sample_id"] in cache.vectors]
    gallery, known, unknown = ok(gallery), ok(known), ok(unknown)
    g_ids = [r["subject_id"] for r in gallery]       # nome di ogni COLONNA delle matrici
    # se la foto di gallery di una persona non aveva volto, i suoi probe non sono
    # piu' "registrati": si tolgono (in closed-set ogni probe deve avere un template)
    known = [r for r in known if r["subject_id"] in set(g_ids)]
    if not gallery or not known or not unknown:
        raise SystemExit("identificazione impossibile: servono gallery, probe registrati e sconosciuti "
                         f"(trovati {len(gallery)}, {len(known)}, {len(unknown)} con volto rilevato)")
    G = cache.get([r["sample_id"] for r in gallery])                                    # template (n_gallery x 512)
    D_known = cosine_distance_matrix(cache.get([r["sample_id"] for r in known]), G)     # (n_registrati x n_gallery)
    D_unknown = cosine_distance_matrix(cache.get([r["sample_id"] for r in unknown]), G)  # (n_sconosciuti x n_gallery)
    k_ids = [r["subject_id"] for r in known]         # nome VERO di ogni riga di D_known
    cms = cms_curve(D_known, k_ids, g_ids)           # cms[0] = RR, cms[4] = CMS(5), ...
    results["closed_set"] = {
        "gallery_identities": len(g_ids), "probes": len(known),
        # con meno di 5 o 10 persone in gallery il rango massimo e' il numero di persone
        "rr": float(cms[0]), "cms5": float(cms[min(4, len(cms) - 1)]),
        "cms10": float(cms[min(9, len(cms) - 1)]),
    }
    open_set = {"unknown_probes": len(unknown)}
    for target in (0.01, 0.10):                      # false alarm tollerato: 1% e 10%
        d, t, fa = dir_at_false_alarm(D_known, k_ids, D_unknown, g_ids, target)
        open_set[f"dir_at_fa_{int(target * 100)}pct"] = {"dir": d, "threshold": t, "false_alarm": fa}
    results["open_set"] = open_set

    # --- 5. stampa e salvataggio -------------------------------------------
    v, c = results["verification"], results["closed_set"]
    print("\n=== VERIFICA 1:1 (coppie ufficiali LFW) ===")
    print(f"coppie usate        {v['pairs_used']}  (escluse per volto non trovato: {v['pairs_skipped']})")
    print(f"EER                 {v['eer'] * 100:.2f} %   alla soglia {v['eer_threshold']:.3f}")
    print(f"distanza media      genuini {v['genuine_mean']:.3f}   impostori {v['impostor_mean']:.3f}")
    print(f"d'                  {v['d_prime']:.2f}")
    if v["accuracy_mean"] is not None:
        print(f"accuratezza 10-fold {v['accuracy_mean'] * 100:.2f} % (+/- {v['accuracy_std'] * 100:.2f})")
    print("\n=== IDENTIFICAZIONE CLOSED-SET ===")
    print(f"gallery {c['gallery_identities']} persone, {c['probes']} probe")
    print(f"RR = CMS(1) {c['rr'] * 100:.2f} %   CMS(5) {c['cms5'] * 100:.2f} %   CMS(10) {c['cms10'] * 100:.2f} %")
    print("\n=== IDENTIFICAZIONE OPEN-SET (rango 1) ===")
    print(f"sconosciuti {open_set['unknown_probes']}")
    for key in ("dir_at_fa_1pct", "dir_at_fa_10pct"):
        o = open_set[key]
        print(f"false alarm {o['false_alarm'] * 100:5.2f} %  ->  DIR {o['dir'] * 100:.2f} %  (soglia {o['threshold']:.3f})")
    out = Path(root) / ("results_arcface_quick.json" if args.quick else "results_arcface.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"\nrisultati salvati in {out}")


if __name__ == "__main__":
    main()
