"""Margine top-1/top-2 e distanza dalla soglia: i concorrenti "semplici" dell'SRR.

TEORIA
    Come l'SRR, sono criteri calcolati DOPO il confronto con la gallery, sulla
    singola risposta, senza sapere chi e' il probe (si possono usare online).
    Servono alla domanda di ricerca D1: guardare la forma di TUTTE le distanze
    (SRR) e' meglio che guardare solo i primi posti?

    Si parte, come per l'SRR, dalle distanze del probe da ogni PERSONA della
    gallery, ordinate: d1 <= d2 <= ... <= dN.

    MARGINE TOP-1/TOP-2        m = d2 - d1
        Quanto il primo candidato stacca il secondo. Margine grande = primo
        ben isolato = risposta affidabile. Guarda SOLO i primi due posti:
        un terzo candidato vicinissimo non lo cambia (il density ratio si').
        Nome: si chiama sempre "margine top-1/top-2", perche' nelle note del
        corso "margine" indica un'altra cosa (la differenza tra FAR e FRR).

    DISTANZA DALLA SOGLIA      s = t - d1
        Quanto il primo candidato sta sotto la soglia di accettazione t.
        s > 0: il primo e' accettato, tanto piu' sicuramente quanto s e' grande;
        s < 0: rifiutato. Guarda SOLO il primo posto.
        Attenzione: s e' d1 spostato e girato di segno. Ordinare i probe per s
        o per -d1 da' lo stesso ordine, qualunque sia t: come criterio di
        affidabilita' equivale a "d1 piccolo = risposta sicura", e t conta solo
        come soglia di decisione (che si sceglie comunque sul tuning).

    Come per phi, valori PIU' ALTI = risposta piu' affidabile, quindi il valore
    critico si stima con la stessa funzione: srr.critical_phi.

COSA C'E' IN QUESTO FILE
    top_two             d1 e d2 di ogni probe (per persona)
    margin_top2         d2 - d1
    threshold_distance  t - d1

MAPPA DELLE CHIAMATE
    top_two -> eval.metrics._per_identity
    Chi usa questo file da fuori: tests/test_margin.py, scripts/eval_srr.py
"""

from __future__ import annotations

import numpy as np

from secondlook.eval.metrics import _per_identity


def top_two(dist, gallery_ids):
    """Le due distanze piu' piccole di ogni probe, contando UNA distanza per persona.

    COME LO CALCOLA IL CODICE
        _per_identity tiene per ogni persona il template piu' vicino; poi
        np.partition mette ai primi due posti di ogni riga i due valori piu'
        piccoli (piu' veloce di ordinare tutta la riga), e li si ordina.

    ESEMPIO (una riga, Mario con 2 template)
        colonne       Mario  Mario  Anna  Luca
        distanze      0.30   0.33   0.80  0.85
        per persona   Mario 0.30, Anna 0.80, Luca 0.85   -> d1 = 0.30, d2 = 0.80
        (contando le foto, d2 sarebbe 0.33, l'altra foto di Mario)

    INPUT       : dist = matrice (n_probe x n_template); gallery_ids = persona di
                  ogni colonna, almeno 2 persone diverse
    RESTITUISCE : (d1, d2), due array lunghi n_probe
    CHIAMA      : eval.metrics._per_identity, np.partition
    CHIAMATA DA : scripts/eval_srr.py, tests/test_margin.py
    """
    _, reduced = _per_identity(dist, gallery_ids)
    if reduced.shape[1] < 2:
        raise ValueError("servono almeno 2 persone in gallery")
    first_two = np.sort(np.partition(reduced, 1, axis=1)[:, :2], axis=1)
    return first_two[:, 0], first_two[:, 1]


def margin_top2(d1, d2):
    """Margine top-1/top-2: m = d2 - d1 (piu' alto = piu' affidabile).

    ESEMPIO (casi della guida del progetto)
        caso A, distanze 0.30, 0.80, ...  ->  m = 0.80 - 0.30 = 0.50   netta
        caso B, distanze 0.30, 0.32, ...  ->  m = 0.32 - 0.30 = 0.02   ambigua

    INPUT       : d1, d2 = numeri o array (da top_two)
    RESTITUISCE : stesso tipo dell'input, valori >= 0
    CHIAMATA DA : scripts/eval_srr.py, tests/test_margin.py
    """
    return np.asarray(d2, dtype=float) - np.asarray(d1, dtype=float)


def threshold_distance(d1, t: float):
    """Distanza dalla soglia di accettazione: s = t - d1 (piu' alto = piu' affidabile).

    ESEMPIO (t = 0.5)
        d1 = 0.30  ->  s = +0.20   accettato con margine
        d1 = 0.48  ->  s = +0.02   accettato per un soffio
        d1 = 0.70  ->  s = -0.20   rifiutato

    INPUT       : d1 = numero o array; t = soglia di accettazione (stessa scala delle distanze)
    RESTITUISCE : stesso tipo dell'input
    CHIAMATA DA : tests/test_margin.py (eval_srr.py usa -d1, che ordina i probe allo stesso modo)
    """
    return t - np.asarray(d1, dtype=float)
