"""Valutazione dell'SRR e dei suoi concorrenti su LFW (Fase 5, passi 5.2-5.6).

Domanda: l'SRR riesce a capire quando la risposta del primo stadio (identificazione
open-set contro tutta la gallery) e' sbagliata? E come se la cavano i criteri piu'
semplici, che guardano solo i primi posti (margine top-1/top-2, distanza dalla soglia)?

Cosa fa:
  1. ricostruisce le matrici dell'identificazione di eval_lfw.py (registrati e
     sconosciuti contro tutta la gallery), usando i vettori GIA' in cache
  2. etichetta ogni risposta: GIUSTA se il probe e' registrato e il primo candidato
     e' la sua persona; SBAGLIATA se il primo candidato e' un altro, oppure se il
     probe e' di uno sconosciuto (qualunque nome gli si dia e' sbagliato)
  3. calcola per ogni probe i cinque criteri, senza usare le etichette, come
     farebbe il login dal vivo (per tutti: piu' alto = piu' affidabile):
        relative    phi della relative distance  (SRR)
        density     phi del density ratio        (SRR)
        margin      margine top-1/top-2, d2 - d1
        threshold   distanza dalla soglia, t - d1 (si usa -d1: stesso ordine dei probe)
        weibull     meta-recognition: -log10 P(d1) secondo una Weibull adattata
                    alla coda delle distanze dei candidati sbagliati
     e, se esiste la cache data/lfw/quality_center.npz, i tre indici di QUALITA'
     della foto (calcolati prima del confronto, uguali per ArcFace e LBP):
        sp  posa,  si  illuminazione,  sy  simmetria   (face/quality.py)
  4. divide le PERSONE a meta': sulla meta' "tuning" stima il valore critico di
     ogni criterio, sulla meta' "test" misura gli errori con quel valore fisso
     ATTENZIONE: un solo punto per criterio, utile ma non per dire chi vince.
  5. confronta i criteri a TUTTE le soglie sulla meta' test (passo 5.4):
     PRR (dalla curva errore contro rifiuto) e AUROC, sia su tutte le risposte
     sbagliate sia per tipo di errore: scambio di persona (registrato
     riconosciuto come un altro) e sconosciuto (non registrato a cui si da' un nome)
  6. stampa e salva i risultati (con i punti delle curve, per i grafici)
     in data/lfw/results_srr_<metodo>.json

Uso (dalla cartella SecondLook), dopo aver gia' eseguito eval_lfw.py con lo stesso metodo:
    uv run python scripts/eval_srr.py                 # ArcFace
    uv run python scripts/eval_srr.py --method lbp    # LBP
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from eval_lfw import METHODS, distance_matrix   # stessa cartella: coseno (arcface e varianti) o chi-quadro (lbp)
from secondlook.eval.datasets import lfw_manifest, split_gallery_probes
from secondlook.eval.matrices import EmbeddingCache
from secondlook.eval.metrics import _per_identity, _true_rank_and_distance, auroc, error_reject_curve, prr
from secondlook.reliability.margin import margin_top2, top_two
from secondlook.reliability.metarecognition import weibull_scores
from secondlook.face.quality import pitch_norm, score_pose
from secondlook.reliability.srr import KINDS, critical_phi, phi_matrix, srr

# i criteri valutati; per relative e density si riporta anche l'SRR
CRITERIA = ("relative", "density", "margin", "threshold", "weibull")
QUALITY = ("sp", "si", "sy")       # indici di qualita' della foto (se c'e' la loro cache)


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


def evaluate_kind(phi_known, correct, known_tuning, phi_unknown, unknown_tuning,
                  normalize: bool = True) -> dict:
    """Valore critico sul tuning, errori sul test, per UN criterio di affidabilita'.

    Vale per qualunque criterio in cui piu' alto = piu' affidabile: phi
    dell'SRR, margine top-1/top-2, -d1. Qui "phi" indica il valore del criterio.

    COME LO CALCOLA IL CODICE
        tuning: giuste = registrati del tuning con rango 1
                sbagliate = registrati del tuning con rango > 1 + sconosciuti del tuning
                critical = critical_phi(giuste, sbagliate)   (phi_bar per l'SRR)
        test:   stessa divisione sulle persone del test, e con la regola
                "affidabile se phi > critical" si contano:
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
        normalize      : True solo per i phi dell'SRR (in [0, 1]): aggiunge le mediane
                         dell'SRR; per margine e -d1 la normalizzazione non ha senso
    RESTITUISCE : dizionario con il valore critico, errori sul test, conteggi
                  (e mediane dell'SRR se normalize)
    CHIAMA      : critical_phi, srr
    CHIAMATA DA : main
    """
    critical, _, _ = critical_phi(
        phi_known[known_tuning & correct],
        np.concatenate([phi_known[known_tuning & ~correct], phi_unknown[unknown_tuning]]),
    )
    test_correct = phi_known[~known_tuning & correct]
    test_wrong = np.concatenate([phi_known[~known_tuning & ~correct], phi_unknown[~unknown_tuning]])
    test_all = np.concatenate([test_correct, test_wrong])
    out = {
        "critical": critical,
        "test_correct": int(test_correct.size),
        "test_wrong": int(test_wrong.size),
        "errors_let_through": float((test_wrong > critical).mean()),       # sbagliate prese per affidabili
        "useless_step_up": float((test_correct <= critical).mean()),      # giuste prese per inaffidabili
        "step_up_rate": float((test_all <= critical).mean()),
    }
    if normalize:
        out["srr_median_correct"] = float(np.median(srr(test_correct, critical)))
        out["srr_median_wrong"] = float(np.median(srr(test_wrong, critical)))
    return out


def threshold_free(phi_known, correct, known_tuning, phi_unknown, unknown_tuning, points: int = 101) -> dict:
    """PRR, AUROC e curva errore-rifiuto di UN criterio, sulla meta' test, senza soglie.

    TEORIA
        evaluate_kind guarda il criterio a un solo punto (il suo valore critico).
        Qui lo si guarda a TUTTE le soglie, cosi' i criteri si confrontano alla
        pari. Gli errori della meta' test sono di due tipi:
            scambio      = registrato, ma il primo candidato e' un altro
            sconosciuto  = non registrato, qualunque nome gli si dia e' sbagliato
        Si calcola l'AUROC giuste contro sbagliate di ogni tipo e di tutte insieme.

    COME LO CALCOLA IL CODICE
        test = probe delle persone NON di tuning; risposte = registrati + sconosciuti
        prr(valori, giusta?) e auroc(giuste, sbagliate) da eval/metrics.py
        la curva si salva su `points` quote di rifiuto equidistanti (0, 0.01, ..., 1),
        con np.interp, per poterla disegnare senza salvare migliaia di punti

    ESEMPIO (numeri piccoli)
        test: giuste [0.9, 0.8, 0.4], scambi [0.6], sconosciuti [0.2]
        prr = 0.67   auroc_all = 5/6   auroc_swap = 2/3   auroc_unknown = 1.0

    INPUT       : come evaluate_kind; points = punti della curva salvata
    RESTITUISCE : dizionario con prr, auroc_all, auroc_swap, auroc_unknown, n_swap,
                  n_unknown e la curva (reject, error)
    CHIAMA      : prr, auroc, error_reject_curve
    CHIAMATA DA : main, tests/test_eval_srr.py
    """
    t_known, t_correct = phi_known[~known_tuning], correct[~known_tuning]
    t_unknown = phi_unknown[~unknown_tuning]
    good, swap = t_known[t_correct], t_known[~t_correct]
    scores = np.concatenate([t_known, t_unknown])
    is_correct = np.concatenate([t_correct, np.zeros(t_unknown.size, dtype=bool)])
    r, err = error_reject_curve(scores, is_correct)
    grid = np.linspace(0, 1, points)
    return {
        "prr": prr(scores, is_correct),
        "auroc_all": auroc(good, np.concatenate([swap, t_unknown])),
        "auroc_swap": auroc(good, swap) if swap.size else float("nan"),
        "auroc_unknown": auroc(good, t_unknown) if t_unknown.size else float("nan"),
        "n_swap": int(swap.size),
        "n_unknown": int(t_unknown.size),
        "curve": {"reject": grid.tolist(), "error": np.interp(grid, r, err).tolist()},
    }


def quality_scores(q_known, q_unknown, known_tuning, unknown_tuning):
    """SP, SI e SY di ogni probe dalle misure salvate in cache.

    COME LO CALCOLA IL CODICE
        la cache ha, per ogni foto, [roll, yaw, r, SI, SY] (face/quality.quality_measures).
        Per SP serve il valore "frontale" r0 del rapporto occhi-naso / naso-mento:
        si stima come MEDIANA di r sulle foto delle persone di tuning (registrati
        e sconosciuti), poi pitch = pitch_norm(r, r0) e SP = score_pose(roll, yaw, pitch).
        SI e SY sono gia' pronti.

    ESEMPIO : r0 = 0.44; foto con roll 0, yaw 0.05, r 0.44 -> pitch 0 -> SP = 0.983

    INPUT       : q_known, q_unknown = matrici (n x 5) dalla cache, nell'ordine dei probe;
                  known_tuning, unknown_tuning = True per le persone di tuning
    RESTITUISCE : (punteggi dei registrati, punteggi degli sconosciuti, r0);
                  i punteggi sono dizionari "sp"/"si"/"sy" -> array
    CHIAMA      : pitch_norm, score_pose
    CHIAMATA DA : main, tests/test_eval_srr.py
    """
    r0 = float(np.median(np.concatenate([q_known[known_tuning, 2], q_unknown[unknown_tuning, 2]])))

    def scores(q):
        return {"sp": score_pose(q[:, 0], q[:, 1], pitch_norm(q[:, 2], r0)), "si": q[:, 3], "sy": q[:, 4]}

    return scores(q_known), scores(q_unknown), r0


def criteria_scores(dist, gallery_ids) -> dict:
    """I cinque criteri di affidabilita' per ogni riga di una matrice probe x template.

    COME LO CALCOLA IL CODICE
        relative, density : phi_matrix (srr.py)
        margin            : top_two + margin_top2 (margin.py)
        threshold         : -d1. La distanza dalla soglia t - d1 ordina i probe
                            esattamente come -d1 per qualunque t, quindi il suo
                            valore critico c sul tuning equivale alla soglia t = -c
        weibull           : weibull_scores (metarecognition.py), coda di 10 distanze
    ESEMPIO (una riga, caso B della guida: 0.30, 0.32, 0.35, 0.90, 0.95)
        relative 0.02   density 0.50   margin 0.02   threshold -0.30   weibull 0.92
    INPUT       : dist = matrice (n_probe x n_template); gallery_ids = persona di ogni colonna
    RESTITUISCE : dizionario criterio -> array di n_probe valori (piu' alto = piu' affidabile)
    CHIAMA      : phi_matrix, top_two, margin_top2, weibull_scores
    CHIAMATA DA : main
    """
    d1, d2 = top_two(dist, gallery_ids)
    return {
        "relative": phi_matrix(dist, gallery_ids, "relative"),
        "density": phi_matrix(dist, gallery_ids, "density"),
        "margin": margin_top2(d1, d2),
        "threshold": -d1,
        "weibull": weibull_scores(dist, gallery_ids),
    }


def prepare(root: Path, method: str) -> dict:
    """Carica dati, distanze e punteggi di tutti i criteri: i passi 1-5 di main, senza valutare.

    Serve a main e agli script che analizzano gli stessi punteggi con la stessa
    divisione tuning/test (scripts/eval_hard_impostors.py).

    RESTITUISCE : dizionario con
        results          conteggi gia' pronti (probe, rango 1 giusto, persone di tuning, r0)
        criteria         nomi dei criteri disponibili (con la qualita' se c'e' la cache)
        s_known, s_unknown   punteggio di ogni criterio per ogni probe (piu' alto = piu' affidabile)
        correct          True dove il registrato ha la persona giusta al rango 1
        known_tuning, unknown_tuning   True dove il probe e' del gruppo di tuning
        known, unknown   righe del manifest dei probe
        ids, red_known, red_unknown    persone della gallery e distanza da ognuna (una per persona)
    CHIAMA      : le stesse funzioni di main (vedi PASSI)
    CHIAMATA DA : main, scripts/eval_hard_impostors.py
    """
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
    _, reduced_u = _per_identity(D_unknown, g_ids)
    rank, _ = _true_rank_and_distance(reduced, ids, k_ids)
    correct = rank == 1                                   # per gli sconosciuti: sempre sbagliata

    # --- 4. divisione per persona: meta' tuning, meta' test ------------------
    tuning = split_subjects(k_ids + u_ids)
    known_tuning = np.array([s in tuning for s in k_ids])
    unknown_tuning = np.array([s in tuning for s in u_ids])

    # --- 5. una valutazione per ogni criterio --------------------------------
    results = {"method": method, "known_probes": len(known), "unknown_probes": len(unknown),
               "rank1_correct": float(correct.mean()), "tuning_subjects": len(tuning)}
    s_known, s_unknown = criteria_scores(D_known, g_ids), criteria_scores(D_unknown, g_ids)
    criteria = list(CRITERIA)
    q_cache = EmbeddingCache(root / "quality_center.npz")
    k_samples, u_samples = [r["sample_id"] for r in known], [r["sample_id"] for r in unknown]
    if all(sid in q_cache.vectors for sid in k_samples + u_samples):
        qk, qu, r0 = quality_scores(q_cache.get(k_samples), q_cache.get(u_samples), known_tuning, unknown_tuning)
        s_known.update(qk)
        s_unknown.update(qu)
        criteria += list(QUALITY)
        results["quality_r0"] = r0
    else:
        print("cache della qualita' assente o incompleta: SP, SI, SY non valutati")
    return {"results": results, "criteria": criteria, "s_known": s_known, "s_unknown": s_unknown,
            "correct": correct, "known_tuning": known_tuning, "unknown_tuning": unknown_tuning,
            "known": known, "unknown": unknown, "ids": ids, "red_known": reduced, "red_unknown": reduced_u}


def main() -> None:
    """Esegue la valutazione dell'SRR per un metodo (arcface o lbp).

    PASSI (e funzioni usate; i passi 1-5 fino ai punteggi sono in prepare)
        1. dati:      lfw_manifest, split_gallery_probes, EmbeddingCache (solo lettura)
        2. matrici:   distance_matrix (da eval_lfw.py) per registrati e sconosciuti
        3. etichette: _per_identity, _true_rank_and_distance -> giusta se rango 1
        4. divisione: split_subjects (meta' tuning, meta' test, per persona)
        5. per ogni criterio: criteria_scores, evaluate_kind (un punto),
           threshold_free (PRR, AUROC, curva); se c'e' la cache della qualita',
           anche quality_scores -> SP, SI, SY
        6. stampa e salva results_srr_<metodo>.json

    CHIAMATA DA : riga di comando
    """
    warnings.filterwarnings("ignore", message="`estimate` is deprecated", category=FutureWarning)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="data/lfw", help="cartella di LFW (default: data/lfw)")
    parser.add_argument("--method", choices=METHODS, default="arcface",
                        help="arcface, lbp, arcface_noalign o arcface_vj (vedi eval_lfw.py)")
    args = parser.parse_args()
    root, method = Path(args.root), args.method

    p = prepare(root, method)
    results, criteria = p["results"], p["criteria"]
    s_known, s_unknown, correct = p["s_known"], p["s_unknown"], p["correct"]
    known_tuning, unknown_tuning = p["known_tuning"], p["unknown_tuning"]
    for c in criteria:
        results[c] = evaluate_kind(s_known[c], correct, known_tuning, s_unknown[c], unknown_tuning,
                                   normalize=c in KINDS)
        results[c].update(threshold_free(s_known[c], correct, known_tuning, s_unknown[c], unknown_tuning))
    # per la distanza dalla soglia si riporta la soglia vera t = -valore critico
    results["threshold"]["acceptance_threshold"] = -results["threshold"]["critical"]

    # --- 6. stampa e salvataggio ---------------------------------------------
    print(f"risposte giuste (rango 1) tra i registrati: {results['rank1_correct'] * 100:.2f} %")
    names = {"relative": "SRR relative distance (phi_bar)", "density": "SRR density ratio (phi_bar)",
             "margin": "margine top-1/top-2", "threshold": "distanza dalla soglia (soglia t)",
             "weibull": "meta-recognition di Weibull", "sp": "qualita': posa (SP)",
             "si": "qualita': illuminazione (SI)", "sy": "qualita': simmetria (SY)"}
    for c in criteria:
        r = results[c]
        crit = r.get("acceptance_threshold", r["critical"])
        print(f"\n=== {names[c]} === valore critico {crit:.4f} (stimato sul tuning)")
        print(f"test: {r['test_correct']} risposte giuste, {r['test_wrong']} sbagliate")
        print(f"errori lasciati passare  {r['errors_let_through'] * 100:6.2f} %   (sbagliate giudicate affidabili)")
        print(f"step-up inutili          {r['useless_step_up'] * 100:6.2f} %   (giuste giudicate inaffidabili)")
        print(f"tasso di step-up         {r['step_up_rate'] * 100:6.2f} %   (su tutti i login del test)")
        if "srr_median_correct" in r:
            print(f"SRR mediano: giuste {r['srr_median_correct']:+.2f}   sbagliate {r['srr_median_wrong']:+.2f}")
    print("\n=== a tutte le soglie (meta' test) ===")
    print(f"{'criterio':38s} {'PRR':>6s} {'AUROC':>7s} {'scambi':>8s} {'sconosc.':>9s}")
    for c in criteria:
        r = results[c]
        print(f"{names[c]:38s} {r['prr']:6.3f} {r['auroc_all']:7.4f} {r['auroc_swap']:8.4f} {r['auroc_unknown']:9.4f}")
    r0 = results[CRITERIA[0]]
    print(f"(nel test: {r0['n_swap']} scambi di persona, {r0['n_unknown']} sconosciuti)")
    out = root / f"results_srr_{method}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nrisultati salvati in {out}")


if __name__ == "__main__":
    main()
