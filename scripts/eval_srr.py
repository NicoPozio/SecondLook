"""Valutazione dell'SRR su LFW (Fase 5, passo 5.2).

Domanda: l'SRR riesce a capire quando la risposta del primo stadio (identificazione
open-set contro tutta la gallery) e' sbagliata?

Cosa fa:
  1. ricostruisce le matrici dell'identificazione di eval_lfw.py (registrati e
     sconosciuti contro tutta la gallery), usando i vettori GIA' in cache
  2. etichetta ogni risposta: GIUSTA se il probe e' registrato e il primo candidato
     e' la sua persona; SBAGLIATA se il primo candidato e' un altro, oppure se il
     probe e' di uno sconosciuto (qualunque nome gli si dia e' sbagliato)
  3. calcola phi di ogni probe (relative distance e density ratio), senza usare
     le etichette, come farebbe il login dal vivo
  4. divide le PERSONE a meta': sulla meta' "tuning" stima phi_bar, sulla meta'
     "test" misura gli errori con phi_bar fisso
  5. stampa e salva i risultati in data/lfw/results_srr_<metodo>.json

Uso (dalla cartella SecondLook), dopo aver gia' eseguito eval_lfw.py con lo stesso metodo:
    uv run python scripts/eval_srr.py                 # ArcFace
    uv run python scripts/eval_srr.py --method lbp    # LBP
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from eval_lfw import distance_matrix      # stesso script-folder: coseno (arcface) o chi-quadro (lbp)
from secondlook.eval.datasets import lfw_manifest, split_gallery_probes
from secondlook.eval.matrices import EmbeddingCache
from secondlook.eval.metrics import _per_identity, _true_rank_and_distance
from secondlook.reliability.srr import KINDS, critical_phi, phi_matrix, srr


def split_subjects(subject_ids, seed: int = 0) -> set:
    """Sceglie a caso META' delle persone come gruppo di tuning; le altre sono il test.

    TEORIA
        Le soglie (qui phi_bar) si scelgono su un gruppo di persone e i risultati
        si riportano su ALTRE persone, per non "tarare sul test". Si divide per
        persona, non per foto: tutte le foto di una persona stanno dalla stessa parte.

    COME LO CALCOLA IL CODICE
        ordina i nomi (cosi' il risultato non dipende dall'ordine di arrivo),
        li mescola con un generatore a seme fisso (stessa divisione a ogni
        esecuzione, anche su un altro PC) e prende la prima meta'.

    ESEMPIO : ["Anna", "Bruno", "Carla", "Dario"] -> per esempio {"Carla", "Anna"}
    INPUT   : subject_ids = nomi (anche ripetuti); seed = seme del generatore
    RESTITUISCE : insieme dei nomi del gruppo di tuning
    CHIAMATA DA : main
    """
    names = np.array(sorted(set(subject_ids)))
    rng = np.random.default_rng(seed)
    return set(names[rng.permutation(len(names))][: len(names) // 2].tolist())


def evaluate_kind(phi_known, correct, known_tuning, phi_unknown, unknown_tuning) -> dict:
    """phi_bar sul tuning, errori sul test, per una versione di phi.

    COME LO CALCOLA IL CODICE
        tuning: giuste = registrati del tuning con rango 1
                sbagliate = registrati del tuning con rango > 1 + sconosciuti del tuning
                phi_bar = critical_phi(giuste, sbagliate)
        test:   stessa divisione sulle persone del test, e con la regola
                "affidabile se phi > phi_bar" si contano:
                  errori_lasciati_passare = sbagliate con phi > phi_bar  / sbagliate
                  step_up_inutili         = giuste    con phi <= phi_bar / giuste
                  tasso_step_up           = probe con phi <= phi_bar / tutti i probe

    ESEMPIO (numeri piccoli)
        test: giuste phi [0.9, 0.8, 0.3], sbagliate phi [0.1, 0.7], phi_bar = 0.5
        errori lasciati passare = 1/2 (0.7)     step-up inutili = 1/3 (0.3)
        tasso di step-up = 2/5 (0.3 e 0.1)

    INPUT
        phi_known      : phi di ogni probe registrato
        correct        : True dove il primo candidato e' la persona giusta (rango 1)
        known_tuning   : True dove il probe registrato appartiene al gruppo di tuning
        phi_unknown    : phi di ogni probe sconosciuto
        unknown_tuning : True dove lo sconosciuto appartiene al gruppo di tuning
    RESTITUISCE : dizionario con phi_bar, errori sul tuning e sul test, conteggi, mediane dell'SRR
    CHIAMA      : critical_phi, srr
    CHIAMATA DA : main
    """
    phi_bar, _, _ = critical_phi(
        phi_known[known_tuning & correct],
        np.concatenate([phi_known[known_tuning & ~correct], phi_unknown[unknown_tuning]]),
    )
    test_correct = phi_known[~known_tuning & correct]
    test_wrong = np.concatenate([phi_known[~known_tuning & ~correct], phi_unknown[~unknown_tuning]])
    test_all = np.concatenate([test_correct, test_wrong])
    return {
        "phi_bar": phi_bar,
        "test_correct": int(test_correct.size),
        "test_wrong": int(test_wrong.size),
        "errors_let_through": float((test_wrong > phi_bar).mean()),        # sbagliate prese per affidabili
        "useless_step_up": float((test_correct <= phi_bar).mean()),       # giuste prese per inaffidabili
        "step_up_rate": float((test_all <= phi_bar).mean()),
        "srr_median_correct": float(np.median(srr(test_correct, phi_bar))),
        "srr_median_wrong": float(np.median(srr(test_wrong, phi_bar))),
    }


def main() -> None:
    """Esegue la valutazione dell'SRR per un metodo (arcface o lbp).

    PASSI (e funzioni usate)
        1. dati:      lfw_manifest, split_gallery_probes, EmbeddingCache (solo lettura)
        2. matrici:   distance_matrix (da eval_lfw.py) per registrati e sconosciuti
        3. etichette: _per_identity, _true_rank_and_distance -> giusta se rango 1
        4. divisione: split_subjects (meta' tuning, meta' test, per persona)
        5. per ogni versione di phi: phi_matrix, evaluate_kind
        6. stampa e salva results_srr_<metodo>.json

    CHIAMATA DA : riga di comando
    """
    warnings.filterwarnings("ignore", message="`estimate` is deprecated", category=FutureWarning)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="data/lfw", help="cartella di LFW (default: data/lfw)")
    parser.add_argument("--method", choices=["arcface", "lbp"], default="arcface")
    args = parser.parse_args()
    root, method = Path(args.root), args.method

    # --- 1. dati: le stesse liste di eval_lfw.py -----------------------------
    cache_path = root / f"embeddings_{method}_center.npz"
    if not cache_path.exists():
        raise SystemExit(f"manca {cache_path}: esegui prima eval_lfw.py --method {method}")
    cache = EmbeddingCache(cache_path)
    gallery, known, unknown = split_gallery_probes(lfw_manifest(root))
    ok = lambda rs: [r for r in rs if r["sample_id"] in cache.vectors]
    gallery, known, unknown = ok(gallery), ok(known), ok(unknown)
    g_ids = [r["subject_id"] for r in gallery]
    known = [r for r in known if r["subject_id"] in set(g_ids)]
    k_ids = [r["subject_id"] for r in known]
    u_ids = [r["subject_id"] for r in unknown]

    # --- 2. matrici: ogni probe contro TUTTA la gallery ----------------------
    print(f"metodo {method}: {len(known)} registrati, {len(unknown)} sconosciuti, {len(g_ids)} persone in gallery")
    G = cache.get([r["sample_id"] for r in gallery])
    D_known = distance_matrix(cache.get([r["sample_id"] for r in known]), G, method)
    D_unknown = distance_matrix(cache.get([r["sample_id"] for r in unknown]), G, method)

    # --- 3. etichette: la risposta del primo stadio e' giusta? ---------------
    ids, reduced = _per_identity(D_known, g_ids)
    rank, _ = _true_rank_and_distance(reduced, ids, k_ids)
    correct = rank == 1                                   # per gli sconosciuti: sempre sbagliata

    # --- 4. divisione per persona: meta' tuning, meta' test ------------------
    tuning = split_subjects(k_ids + u_ids)
    known_tuning = np.array([s in tuning for s in k_ids])
    unknown_tuning = np.array([s in tuning for s in u_ids])

    # --- 5. una valutazione per ogni versione di phi -------------------------
    results = {"method": method, "known_probes": len(known), "unknown_probes": len(unknown),
               "rank1_correct": float(correct.mean()), "tuning_subjects": len(tuning)}
    for kind in KINDS:
        results[kind] = evaluate_kind(phi_matrix(D_known, g_ids, kind), correct, known_tuning,
                                      phi_matrix(D_unknown, g_ids, kind), unknown_tuning)

    # --- 6. stampa e salvataggio ---------------------------------------------
    print(f"risposte giuste (rango 1) tra i registrati: {results['rank1_correct'] * 100:.2f} %")
    for kind in KINDS:
        r = results[kind]
        print(f"\n=== {kind} === phi_bar = {r['phi_bar']:.4f} (stimato sul tuning)")
        print(f"test: {r['test_correct']} risposte giuste, {r['test_wrong']} sbagliate")
        print(f"errori lasciati passare  {r['errors_let_through'] * 100:6.2f} %   (sbagliate con phi > phi_bar)")
        print(f"step-up inutili          {r['useless_step_up'] * 100:6.2f} %   (giuste con phi <= phi_bar)")
        print(f"tasso di step-up         {r['step_up_rate'] * 100:6.2f} %   (su tutti i login del test)")
        print(f"SRR mediano: giuste {r['srr_median_correct']:+.2f}   sbagliate {r['srr_median_wrong']:+.2f}")
    out = root / f"results_srr_{method}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nrisultati salvati in {out}")


if __name__ == "__main__":
    main()
