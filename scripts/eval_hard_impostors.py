"""Il caso "impostore molto simile, foto ottima" (Fase 5, passo 5.9).

Domanda: una foto di ottima qualita' garantisce che la risposta del primo stadio sia giusta?
La guida dice di no, e chiede di mostrarlo con il caso piu' insidioso: una foto
perfetta (frontale, ben illuminata, simmetrica) di una persona che NON e' quella
proposta dal sistema, perche' non e' registrata oppure perche' assomiglia molto a
un'altra. Gli indici di qualita' guardano solo la foto e la giudicano ottima; l'SRR
guarda le distanze dalla gallery e dovrebbe accorgersi che il primo candidato non
si stacca dagli altri.

Cosa fa (stessi punteggi e stessa divisione tuning/test di eval_srr.py):
  1. FOTO OTTIMA: SP, SI e SY tutti nel quarto migliore. Le tre soglie (75esimo
     percentile) si stimano sulle persone di TUNING e si applicano al TEST.
  2. Tra le risposte SBAGLIATE del test (sconosciuti e scambi di persona) con foto
     ottima conta quante ne lascia passare ogni criterio, al suo valore critico
     stimato sul tuning. Per confronto, lo stesso conteggio su tutte le sbagliate
     e gli step-up inutili sulle risposte GIUSTE con foto ottima.
  3. Mostra gli esempi piu' insidiosi: le risposte sbagliate con foto ottima e la
     distanza d1 piu' PICCOLA (il volto piu' simile a qualcuno della gallery), con le
     loro distanze, l'SRR e il verdetto dei criteri.
  4. Salva data/lfw/results_hard_impostors_<metodo>.json

Uso (dalla cartella SecondLook; servono gli embedding e quality_center.npz):
    uv run python scripts/eval_hard_impostors.py                 (ArcFace)
    uv run python scripts/eval_hard_impostors.py --method lbp
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from eval_srr import QUALITY, evaluate_kind, prepare      # stessa cartella degli script
from secondlook.reliability.srr import srr

COMPARED = ("relative", "margin", "threshold", "weibull", "sp", "si", "sy")    # criteri messi a confronto
TOP_QUANTILE = 0.75                                                # "foto ottima" = quarto migliore


def excellent_photo(q_tuning: dict, q_test: dict, quantile: float = TOP_QUANTILE):
    """Quali foto del test sono "ottime": SP, SI e SY tutti sopra la soglia del tuning.

    COME LO CALCOLA IL CODICE
        per ogni indice (SP, SI, SY): soglia = percentile 75 dei valori del tuning
        foto ottima = indice >= soglia per tutti e tre

    ESEMPIO
        tuning SP [0.5, 0.6, 0.7, 0.8, 0.9] -> percentile 75 = 0.8
        (lo stesso per SI e SY); una foto del test con SP 0.85, SI 0.9, SY 0.95
        e soglie 0.8, 0.8, 0.9 e' ottima; con SY 0.85 no.
        Se i tre indici fossero indipendenti, le foto ottime sarebbero circa
        1/4 * 1/4 * 1/4 = 1.6 %; in pratica sono di piu', perche' una foto
        frontale tende ad avere anche simmetria alta.

    INPUT       : q_tuning, q_test = dizionari "sp"/"si"/"sy" -> array dei valori
    RESTITUISCE : (array booleano sulle foto del test, dizionario delle soglie)
    CHIAMATA DA : main, tests/test_hard_impostors.py
    """
    cut = {q: float(np.quantile(q_tuning[q], quantile)) for q in QUALITY}
    mask = np.all([q_test[q] >= cut[q] for q in QUALITY], axis=0)
    return mask, cut


def let_through(scores: np.ndarray, critical: float) -> float:
    """Quota giudicata affidabile ("affidabile se punteggio > valore critico", come in eval_srr).

    ESEMPIO : punteggi [0.1, 0.7, 0.9], critico 0.5 -> 2/3
    CHIAMATA DA : main
    """
    return float((scores > critical).mean()) if scores.size else float("nan")


def main() -> None:
    """Conta e mostra gli impostori con foto ottima per un metodo (arcface o lbp).

    PASSI (e funzioni usate)
        1. punteggi e divisione: eval_srr.prepare (le stesse di eval_srr.py)
        2. valori critici sul tuning: evaluate_kind, criterio per criterio
        3. foto ottime: excellent_photo (soglie sul tuning)
        4. conteggi sul test: let_through
        5. esempi: le sbagliate con foto ottima e d1 piu' piccolo (le piu' simili)
    CHIAMATA DA : riga di comando
    """
    warnings.filterwarnings("ignore", message="`estimate` is deprecated", category=FutureWarning)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="data/lfw")
    parser.add_argument("--method", choices=["arcface", "lbp"], default="arcface")
    parser.add_argument("--examples", type=int, default=5, help="quanti esempi mostrare")
    args = parser.parse_args()
    root, method = Path(args.root), args.method

    # --- 1. gli stessi punteggi di eval_srr.py -------------------------------
    p = prepare(root, method)
    if not set(QUALITY) <= set(p["criteria"]):
        raise SystemExit("serve data/lfw/quality_center.npz completa (vedi eval_srr.py)")
    sk, su, correct = p["s_known"], p["s_unknown"], p["correct"]
    kt, ut = p["known_tuning"], p["unknown_tuning"]

    # --- 2. valore critico di ogni criterio, stimato sul tuning --------------
    critical = {c: evaluate_kind(sk[c], correct, kt, su[c], ut, normalize=False)["critical"] for c in COMPARED}

    # le righe del test, in un unico elenco: prima i registrati, poi gli sconosciuti
    test_known, test_unknown = ~kt, ~ut
    score = {c: np.concatenate([sk[c][test_known], su[c][test_unknown]]) for c in sk}
    good = np.concatenate([correct[test_known], np.zeros(test_unknown.sum(), bool)])
    kind = np.array(["giusta" if g else "scambio" for g in correct[test_known]] + ["sconosciuto"] * int(test_unknown.sum()))
    rows = [r for r, t in zip(p["known"], test_known) if t] + [r for r, t in zip(p["unknown"], test_unknown) if t]
    red = np.vstack([p["red_known"][test_known], p["red_unknown"][test_unknown]])

    # --- 3. foto ottime: soglie stimate sul tuning ---------------------------
    q_tuning = {q: np.concatenate([sk[q][kt], su[q][ut]]) for q in QUALITY}
    top, cut = excellent_photo(q_tuning, {q: score[q] for q in QUALITY})
    hard = top & ~good                                 # sbagliate con foto ottima
    print(f"\nmetodo {method}: {good.size} risposte nel test, {int((~good).sum())} sbagliate")
    print("foto ottima = SP >= {sp:.3f}, SI >= {si:.3f}, SY >= {sy:.3f}  (75esimo percentile del tuning)".format(**cut))
    print(f"foto ottime nel test: {int(top.sum())} ({top.mean() * 100:.1f} %), di cui sbagliate {int(hard.sum())} "
          f"(sconosciuti {int((hard & (kind == 'sconosciuto')).sum())}, scambi {int((hard & (kind == 'scambio')).sum())})")

    # --- 4. chi le lascia passare? -------------------------------------------
    names = {"relative": "SRR relative distance", "margin": "margine top-1/top-2",
             "threshold": "solo soglia (d1 < t)", "weibull": "Weibull",
             "sp": "qualita' SP", "si": "qualita' SI", "sy": "qualita' SY"}
    table = {}
    print(f"\n{'criterio':24s} {'sbagliate con foto ottima':>27s} {'tutte le sbagliate':>20s} {'step-up inutili (giuste, foto ottima)':>40s}")
    print(f"{'':24s} {'lasciate passare':>27s} {'lasciate passare':>20s}")
    for c in COMPARED:
        table[c] = {"critical": critical[c],
                    "hard_let_through": let_through(score[c][hard], critical[c]),
                    "all_wrong_let_through": let_through(score[c][~good], critical[c]),
                    "useless_step_up_top_correct": 1 - let_through(score[c][top & good], critical[c])}
        t = table[c]
        print(f"{names[c]:24s} {t['hard_let_through'] * 100:26.1f} % {t['all_wrong_let_through'] * 100:19.1f} % "
              f"{t['useless_step_up_top_correct'] * 100:39.1f} %")

    # --- 5. gli esempi piu' insidiosi ----------------------------------------
    d1_all = red.min(axis=1)
    order = np.argsort(d1_all[hard])[: args.examples]          # i piu' simili per primi
    idx = np.flatnonzero(hard)[order]
    phi_bar = critical["relative"]
    examples = []
    print(f"\nle {len(idx)} risposte sbagliate con foto ottima e d1 piu' piccolo (le piu' simili):")
    for i in idx:
        d = red[i]
        first, second = np.argsort(d)[:2]
        ex = {"foto": rows[i]["sample_id"], "tipo": kind[i], "proposto": p["ids"][first],
              "d1": float(d[first]), "d2": float(d[second]),
              "srr": float(srr(np.array([score["relative"][i]]), phi_bar)[0]),
              **{q: float(score[q][i]) for q in QUALITY},
              "passa_qualita": [q for q in QUALITY if score[q][i] > critical[q]],
              "passa_srr": bool(score["relative"][i] > phi_bar),
              "passa_soglia": bool(score["threshold"][i] > critical["threshold"])}
        examples.append(ex)
        print(f"  {ex['foto']:32s} {ex['tipo']:11s} proposto {ex['proposto']:25s} d1 {ex['d1']:.3f} d2 {ex['d2']:.3f} "
              f"SRR {ex['srr']:+.2f}  SP {ex['sp']:.2f} SI {ex['si']:.2f} SY {ex['sy']:.2f}  "
              f"-> qualita' {'PASSA' if len(ex['passa_qualita']) == 3 else 'in parte'}, "
              f"soglia {'PASSA' if ex['passa_soglia'] else 'rifiuta'}, SRR {'PASSA' if ex['passa_srr'] else 'STEP-UP'}")

    out = root / f"results_hard_impostors_{method}.json"
    out.write_text(json.dumps({"method": method, "cut": cut, "acceptance_threshold": -critical["threshold"], "test_answers": int(good.size),
                               "excellent_photos": int(top.sum()), "hard": int(hard.sum()),
                               "criteria": table, "examples": examples}, indent=2))
    print(f"\nrisultati salvati in {out}")


if __name__ == "__main__":
    main()
