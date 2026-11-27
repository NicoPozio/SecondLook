"""Meta-recognition di Weibull: il primo candidato "esce" dalla coda dei non-corrispondenti?

TEORIA (Scheirer et al., "Meta-Recognition", IEEE TPAMI 2011; guida del progetto, 3.4)
    Per un probe, quasi tutte le persone della gallery sono SBAGLIATE: le loro
    distanze descrivono come si comporta un "non-corrispondente" per quel probe.
    Se la risposta e' giusta, il primo candidato non e' un non-corrispondente
    qualunque: la sua distanza d1 e' un valore ANOMALO, molto piu' piccolo di
    quello che ci si aspetterebbe da quella popolazione.

    La teoria dei valori estremi (EVT) dice che il MINIMO di tante distanze
    segue (per distanze >= 0) una distribuzione di Weibull. Quindi:
      1. si prende la "coda" delle distanze vicine al primo, ESCLUSO il primo:
         d2, d3, ..., d(k+1), le k distanze piu' piccole dei candidati sbagliati
         (se la risposta e' giusta, sono tutti non-corrispondenti)
      2. si adatta a queste k distanze una Weibull (scipy.stats.weibull_min,
         con inizio fissato a 0 perche' le distanze non sono negative)
      3. si chiede: quanto e' probabile, secondo questa Weibull, una distanza
         piccola come d1?   p = CDF(d1) = P(non-corrispondente <= d1)
         p minuscola -> d1 e' fuori dalla coda -> risposta affidabile
         p grande    -> d1 sembra un non-corrispondente come gli altri -> ambigua

    Come punteggio si usa  -log10(p): "quanti zeri" ha la probabilita'.
    Piu' alto = piu' affidabile, come per gli altri criteri. Il logaritmo
    serve perche' con ArcFace p arriva a 1e-9 e oltre: 1 - p varrebbe 1.0
    per moltissimi probe e non si potrebbero piu' ordinare.

    Differenze rispetto all'SRR e al margine:
      margine      guarda solo d2 (il secondo)
      SRR          guarda tutte le distanze con una regola fissa (2 * d1, d_N)
      Weibull      IMPARA, per ogni probe, la forma della coda dei
                   non-corrispondenti e misura quanto d1 ne e' lontano

PARAMETRO
    tail_size = k, quante distanze della coda usare (default 10). Troppo poche:
    il fit e' instabile; troppe: si includono candidati lontani che non
    descrivono il "bordo" della coda. Nel login, con pochi utenti, si usano
    al massimo N - 1 distanze; servono almeno 3 persone oltre al primo.

COSA C'E' IN QUESTO FILE
    weibull_score       -log10 P(d1) per UN probe (distanze per persona ordinate)
    weibull_scores      lo stesso per ogni riga di una matrice probe x template

MAPPA DELLE CHIAMATE
    weibull_scores -> eval.metrics._per_identity, weibull_score
    weibull_score  -> scipy.stats.weibull_min (fit e logcdf)
    Chi usa questo file da fuori: tests/test_metarecognition.py, scripts/eval_srr.py
"""

from __future__ import annotations

import numpy as np

from secondlook.eval.metrics import _per_identity

MIN_TAIL = 3            # sotto i 3 punti il fit di una Weibull a 2 parametri non ha senso


def weibull_score(d, tail_size: int = 10) -> float:
    """-log10 della probabilita' che un non-corrispondente sia vicino quanto il primo.

    COME LO CALCOLA IL CODICE
        1. coda = d[1 : k+1], le k distanze dopo il primo (k = tail_size,
           o meno se i candidati sono pochi)
        2. weibull_min.fit(coda, floc=0): stima forma e scala della Weibull
           con l'inizio fissato a 0
        3. weibull_min.logcdf(d1): log della probabilita' P(non-corr. <= d1),
           calcolata in forma logaritmica per non perdere precisione
        4. -log(p) / log(10) = -log10(p)

    ESEMPIO (casi della guida, coda di 4 distanze)
        caso A: d1 = 0.30, coda 0.80, 0.85, 0.90, 0.95
                Weibull stretta intorno a 0.9 (forma 17.8, scala 0.90)
                p = 3.1e-09  ->  punteggio = 8.5     d1 e' lontanissimo: affidabile
        caso B: d1 = 0.30, coda 0.32, 0.35, 0.90, 0.95
                Weibull larga (forma 2.35, scala 0.72): la coda arriva fino a 0.32
                p = 0.12     ->  punteggio = 0.92    d1 sembra uno della coda: ambigua

    INPUT       : d = distanze del probe da ogni PERSONA, ordinate (d[0] = la piu' vicina);
                  tail_size = k
    RESTITUISCE : float >= 0; 0 se il fit non e' possibile (coda tutta uguale)
    SOLLEVA     : ValueError con meno di 1 + MIN_TAIL persone, o distanze non ordinate
    CHIAMA      : scipy.stats.weibull_min.fit, scipy.stats.weibull_min.logcdf
    CHIAMATA DA : weibull_scores, tests/test_metarecognition.py
    """
    from scipy import stats            # import qui: serve solo a questo stimatore

    d = np.asarray(d, dtype=float).ravel()
    if d.size < 1 + MIN_TAIL:
        raise ValueError(f"servono almeno {1 + MIN_TAIL} persone (il primo + {MIN_TAIL} di coda)")
    if np.any(np.diff(d) < 0):
        raise ValueError("le distanze devono essere ordinate: d1 <= d2 <= ... <= dN")
    tail = d[1:1 + tail_size]                       # es. [0.80, 0.85, 0.90, 0.95]
    if np.ptp(tail) == 0 or tail[0] <= 0:           # coda tutta uguale o con distanze nulle: niente fit
        return 0.0
    shape, _, scale = stats.weibull_min.fit(tail, floc=0)
    log_p = stats.weibull_min.logcdf(d[0], shape, loc=0, scale=scale)   # log P(non-corr. <= d1)
    return float(max(0.0, -log_p / np.log(10)))     # -log10(p); >= 0 perche' p <= 1


def weibull_scores(dist, gallery_ids, tail_size: int = 10) -> np.ndarray:
    """weibull_score di ogni probe (riga) di una matrice probe x template.

    COME LO CALCOLA IL CODICE
        _per_identity (una distanza per persona, il template piu' vicino),
        poi per ogni riga si ordinano solo le 1 + k distanze piu' piccole
        (np.partition, piu' veloce che ordinare tutta la riga) e si chiama
        weibull_score. Un fit per riga: circa 3 ms, quindi circa 35 secondi
        per gli 11.500 probe di LFW.

    INPUT       : dist = matrice (n_probe x n_template); gallery_ids = persona di
                  ogni colonna. I nomi dei probe non servono (si usa online).
    RESTITUISCE : array di n_probe punteggi (piu' alto = piu' affidabile)
    CHIAMA      : eval.metrics._per_identity, np.partition, weibull_score
    CHIAMATA DA : scripts/eval_srr.py, tests/test_metarecognition.py
    """
    _, reduced = _per_identity(dist, gallery_ids)
    k = min(tail_size, reduced.shape[1] - 1)
    nearest = np.sort(np.partition(reduced, k, axis=1)[:, :k + 1], axis=1)   # d1 ... d(k+1)
    return np.array([weibull_score(row, tail_size=k) for row in nearest])
