"""SRR (System Response Reliability): quanto e' affidabile UNA risposta dell'identificazione.

TEORIA (appunti, "Affidabilita' del riconoscimento"; slide Lezione 3, slide 33)
    Nel primo stadio del login il probe viene confrontato con TUTTA la gallery
    e il sistema risponde con il candidato piu' vicino. L'SRR non dice CHI e'
    il probe: dice se quella risposta e' "sicura" o "ambigua", guardando la
    FORMA di tutte le distanze, non solo la prima.

    Idea: attorno al primo candidato c'e' una "nuvola" di altri candidati con
    distanza simile. Nuvola vuota -> il primo e' isolato -> risposta affidabile.
    Nuvola affollata -> altri somigliano al probe quasi quanto il primo ->
    risposta ambigua -> nel nostro login scatta lo step-up (keystroke).

    Si parte dalle distanze ordinate, UNA PER PERSONA:  d1 <= d2 <= ... <= dN

    PASSO 1: la funzione phi (due versioni, entrambe in [0, 1])
        relative distance   phi = (d2 - d1) / dN
        density ratio       phi = 1 - |Nb| / |N|
                            N  = tutti i candidati TRANNE il primo
                            Nb = quelli di N con distanza < 2 * d1 (la "nuvola")
        Il primo e' escluso da Nb e da N (slide 33): se fosse incluso
        entrerebbe sempre nella nuvola (d1 < 2*d1) e phi non arriverebbe a 1.
        phi alta = risposta affidabile.

    PASSO 2: il valore critico phi_bar
        Un phi "alto" o "basso" non ha senso assoluto: dipende da gallery e
        algoritmo. phi_bar si stima PRIMA, su dati di tuning con ground truth,
        come il punto che bilancia i due errori di stima:
            risposte SBAGLIATE giudicate affidabili   (phi > phi_bar)
            risposte GIUSTE    giudicate inaffidabili (phi <= phi_bar)
        Ha un ruolo simile alla soglia dell'EER.

    PASSO 3: la normalizzazione
        S   = 1 - phi_bar   se phi > phi_bar      (quanto spazio c'e' sopra)
              phi_bar       altrimenti            (quanto spazio c'e' sotto)
        SRR = (phi - phi_bar) / S          in [-1, 1]
        SRR > 0: sopra il punto critico (affidabile); SRR < 0: sotto.
        |SRR| = quanto si e' lontani dal punto critico, in proporzione.

    Nel login: se SRR < soglia di affidabilita' -> step-up. La soglia si
    sceglie in Fase 6b (si fa variare per costruire le curve).

DISTANZE PER PERSONA, NON PER FOTO
    Se una persona ha piu' template in gallery (nel login 5-8 foto), le sue
    foto sarebbero tutte vicine al primo posto e riempirebbero la nuvola anche
    quando la risposta e' giusta e netta. Per questo phi_matrix prima tiene,
    per ogni persona, la distanza dal suo template piu' vicino (come fa
    eval/metrics._per_identity) e solo dopo ordina e calcola phi.

COSA C'E' IN QUESTO FILE
    phi_relative     relative distance di UN probe (distanze per persona ordinate)
    phi_density      density ratio di UN probe
    phi_matrix       phi di TUTTI i probe di una matrice probe x template
    critical_phi     stima di phi_bar dai phi di risposte giuste e sbagliate
    srr              normalizzazione: phi -> SRR in [-1, 1]

MAPPA DELLE CHIAMATE
    phi_matrix -> eval.metrics._per_identity, poi (in forma vettoriale) la
                  stessa formula di phi_relative / phi_density
    critical_phi, srr: indipendenti
    Chi usa questo file da fuori: tests/test_srr.py
    (prossimo passo: valutazione dell'SRR su LFW; Fase 6b: fusion/stepup.py)
"""

from __future__ import annotations

import numpy as np

from secondlook.eval.metrics import _per_identity

KINDS = ("relative", "density")


def _check_sorted(d) -> np.ndarray:
    """Array 1D di almeno 2 distanze, ordinate in modo crescente (errore altrimenti).

    CHIAMATA DA : phi_relative, phi_density
    """
    d = np.asarray(d, dtype=float).ravel()
    if d.size < 2:
        raise ValueError("servono almeno 2 candidati (persone) per calcolare phi")
    if np.any(np.diff(d) < 0):
        raise ValueError("le distanze devono essere ordinate: d1 <= d2 <= ... <= dN")
    return d


def phi_relative(d) -> float:
    """Relative distance: phi = (d2 - d1) / dN.

    TEORIA
        Guarda lo stacco tra il primo e il secondo candidato, in proporzione
        alla distanza piu' grande. Stacco grande = primo ben isolato.
        Guarda solo i primi due posti (e l'ultimo come scala): un affollamento
        dal terzo posto in poi non la cambia.

    ESEMPIO (appunti del corso)
        distanze 0.15, 0.25, 0.28, 0.45
        phi = (0.25 - 0.15) / 0.45
            = 0.10 / 0.45
            = 0.22

    INPUT       : d = distanze del probe da ogni PERSONA della gallery, gia'
                  ordinate (d[0] = la piu' vicina). Almeno 2 valori.
    RESTITUISCE : float in [0, 1] (0 se tutte le distanze sono 0)
    CHIAMA      : _check_sorted
    CHIAMATA DA : tests/test_srr.py (phi_matrix usa la stessa formula su tutte le righe)
    """
    d = _check_sorted(d)
    return float((d[1] - d[0]) / d[-1]) if d[-1] > 0 else 0.0


def phi_density(d) -> float:
    """Density ratio: phi = 1 - |Nb| / |N|.

    TEORIA
        N  = i candidati diversi dal primo (N - 1 persone)
        Nb = quelli di N "dentro la nuvola", cioe' con distanza < 2 * d1
        Conta TUTTI i candidati vicini, non solo il secondo: e' qui che si
        distingue dal margine top-1/top-2. Le note dicono che e' meno
        sensibile agli outlier e di solito funziona meglio della relative distance.

    ESEMPIO (appunti del corso)
        primo a 0.15, altri candidati a 0.25, 0.31, 0.45
        soglia della nuvola = 2 * 0.15 = 0.30
        dentro (< 0.30): solo 0.25          -> |Nb| = 1
        altri candidati                     -> |N|  = 3
        phi = 1 - 1/3 = 0.67

    INPUT       : come phi_relative
    RESTITUISCE : float in [0, 1]; 1 = nessuno nella nuvola
    CHIAMA      : _check_sorted
    CHIAMATA DA : tests/test_srr.py (phi_matrix usa la stessa formula su tutte le righe)
    """
    d = _check_sorted(d)
    others = d[1:]                                  # N: tutti tranne il primo
    nb = int(np.sum(others < 2 * d[0]))             # Nb: dentro la nuvola
    return 1.0 - nb / others.size


def phi_matrix(dist, gallery_ids, kind: str = "density") -> np.ndarray:
    """phi di ogni probe (riga) di una matrice probe x template.

    COME LO CALCOLA IL CODICE
        1. _per_identity: da colonne = template a colonne = persone, tenendo
           per ogni persona la distanza minima
        2. ordina ogni riga in modo crescente -> d1 <= d2 <= ... <= dN
        3. applica la formula scelta a tutte le righe insieme

    ESEMPIO (una riga, Mario con 2 template)
        colonne        Mario  Mario  Anna  Luca  Sara
        distanze       0.30   0.33   0.80  0.85  0.90
        per persona    Mario 0.30, Anna 0.80, Luca 0.85, Sara 0.90
        density ratio: nuvola < 0.60 -> nessuno -> phi = 1 - 0/3 = 1.00
        (calcolato sulle FOTO, la seconda foto di Mario a 0.33 entrerebbe nella
        nuvola: phi = 1 - 1/4 = 0.75, una falsa ambiguita')

    INPUT
        dist        : matrice (n_probe x n_template) di distanze (coseno o chi-quadro)
        gallery_ids : persona di ogni COLONNA; servono almeno 2 persone diverse
        kind        : "relative" o "density"
        I nomi dei probe NON servono: phi si calcola senza sapere chi e' il probe
        (per questo l'SRR si puo' usare online). Vale anche per gli sconosciuti.
    RESTITUISCE : array di n_probe valori di phi
    CHIAMA      : eval.metrics._per_identity, np.sort
    CHIAMATA DA : tests/test_srr.py (prossimo passo: valutazione su LFW)
    """
    if kind not in KINDS:
        raise ValueError(f"kind sconosciuto: {kind!r} (usa 'relative' o 'density')")
    _, reduced = _per_identity(dist, gallery_ids)
    if reduced.shape[1] < 2:
        raise ValueError("servono almeno 2 persone in gallery per calcolare phi")
    d = np.sort(reduced, axis=1)                    # ogni riga: d1 <= d2 <= ... <= dN
    if kind == "relative":
        dn = d[:, -1]
        return np.where(dn > 0, (d[:, 1] - d[:, 0]) / np.where(dn > 0, dn, 1.0), 0.0)
    nb = (d[:, 1:] < 2 * d[:, :1]).sum(axis=1)      # quanti altri dentro la nuvola, riga per riga
    return 1.0 - nb / (d.shape[1] - 1)


def critical_phi(phi_correct, phi_wrong):
    """Stima del valore critico phi_bar: il punto che bilancia i due errori di stima.

    TEORIA
        Servono dati con ground truth (tuning), divisi in:
            phi_correct = phi delle risposte GIUSTE (primo candidato = persona vera)
            phi_wrong   = phi delle risposte SBAGLIATE (primo candidato sbagliato,
                          oppure probe di uno sconosciuto: qualunque risposta e' sbagliata)
        Con la regola "affidabile se phi > phi_bar" gli errori sono:
            e_wrong(t)   = quota di phi_wrong   >  t   (sbagliate prese per affidabili)
            e_correct(t) = quota di phi_correct <= t   (giuste prese per inaffidabili)
        Alzando t il primo scende e il secondo sale. phi_bar e' il t in cui
        sono piu' vicini all'uguaglianza: il t che rende minimo il piu' grande
        dei due. E' il corrispondente, per phi, della soglia dell'EER.

    COME LO CALCOLA IL CODICE
        Prova come soglie tutti i valori di phi osservati (gli errori cambiano
        solo li'). Niente interpolazione: phi del density ratio prende pochi
        valori (frazioni k/(N-1)), molti probe hanno phi uguali, e phi_bar deve
        essere un valore vero per cui la regola "phi > phi_bar" e' esatta.
        A parita' vince la soglia piu' bassa.

    ESEMPIO
        phi_correct = [0.9, 0.8, 0.7, 0.4]
        phi_wrong   = [0.2, 0.3, 0.5, 0.75]
        t = 0.4:  e_wrong = 2/4 (0.5, 0.75)   e_correct = 1/4 (0.4)        max 0.50
        t = 0.5:  e_wrong = 1/4 (0.75)        e_correct = 1/4 (0.4)        max 0.25  <- minimo
        t = 0.7:  e_wrong = 1/4               e_correct = 2/4 (0.4, 0.7)   max 0.50
        -> (0.5, 0.25, 0.25)

    INPUT       : due liste di phi (lunghezze anche diverse), non vuote
    RESTITUISCE : tupla (phi_bar, e_wrong, e_correct)
    CHIAMATA DA : tests/test_srr.py (prossimo passo: valutazione su LFW)
    """
    c = np.asarray(phi_correct, dtype=float).ravel()
    w = np.asarray(phi_wrong, dtype=float).ravel()
    if c.size == 0 or w.size == 0:
        raise ValueError("servono risposte giuste e risposte sbagliate per stimare phi_bar")
    t = np.unique(np.concatenate([c, w]))                     # soglie candidate, ordinate
    e_wrong = (w[None, :] > t[:, None]).mean(axis=1)          # tabella soglia x risposta
    e_correct = (c[None, :] <= t[:, None]).mean(axis=1)
    k = int(np.argmin(np.maximum(e_wrong, e_correct)))        # primo minimo = soglia piu' bassa
    return float(t[k]), float(e_wrong[k]), float(e_correct[k])


def srr(phi, phi_bar: float):
    """Normalizzazione finale: SRR = (phi - phi_bar) / S, in [-1, 1].

    TEORIA
        S = 1 - phi_bar se phi > phi_bar, altrimenti phi_bar: la distanza da
        phi_bar si misura in proporzione allo spazio disponibile da quella parte.
        Cosi' phi = 1 da' sempre +1, phi = 0 da' sempre -1, phi = phi_bar da' 0.

    ESEMPIO (guida del progetto, phi_bar = 0.6)
        caso A, phi = 1.00:  phi > 0.6 -> S = 1 - 0.6 = 0.4
                             SRR = (1.00 - 0.6) / 0.4 = +1.00   affidabile
        caso B, phi = 0.50:  phi <= 0.6 -> S = 0.6
                             SRR = (0.50 - 0.6) / 0.6 = -0.17   step-up

    INPUT       : phi = un numero o un array di phi in [0, 1]; phi_bar in [0, 1]
    RESTITUISCE : float (se phi e' un numero) o array, in [-1, 1]
                  (se S = 0, cioe' phi_bar = 0 e phi = 0, restituisce 0)
    CHIAMATA DA : tests/test_srr.py (Fase 6b: decisione di step-up)
    """
    p = np.asarray(phi, dtype=float)
    s = np.where(p > phi_bar, 1.0 - phi_bar, phi_bar)
    out = np.where(s > 0, (p - phi_bar) / np.where(s > 0, s, 1.0), 0.0)
    return float(out) if out.ndim == 0 else out
