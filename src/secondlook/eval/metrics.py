"""Metriche di valutazione offline (Step 9 del diagramma, Fase 2 della pipeline).

TEORIA
    Un sistema biometrico confronta due campioni e produce un PUNTEGGIO.
    Per noi il punteggio e' una DISTANZA (distanza coseno tra embedding,
    vedi face/embed.cosine_distance):
        distanza piccola  -> i due campioni si somigliano
        si ACCETTA quando distanza <= soglia t
    Negli appunti (p.14) le formule usano una similarita' s con "accetto se
    s >= t": e' la stessa cosa con il verso della disuguaglianza girato.

    Due modi di usare il sistema, due famiglie di metriche:
      VERIFICA 1:1     "sei davvero Mario?" -> un confronto, si/no
                       FAR, FRR, ROC, DET, EER, d'
      IDENTIFICAZIONE  "chi sei?" -> confronto con TUTTA la gallery
                       closed-set (tutti sono registrati): CMS, CMC, RR
                       open-set (puo' arrivare uno sconosciuto): DIR, false alarm

DA DOVE VENGONO GLI INPUT (importante)
    Questo file fa SOLO conti. Non legge immagini e non calcola embedding.
    - Per la VERIFICA riceve due liste di numeri gia' separate da chi chiama:
        genuine  = distanze di confronti tra due campioni della STESSA persona
        impostor = distanze di confronti tra campioni di persone DIVERSE
      Chi siano le persone non serve: conta solo a quale lista appartiene
      ogni numero. La separazione la fa chi chiama usando le etichette
      (ground truth): es. scripts/eval_lfw.py usa il campo `same` di pairs.txt.
    - Per l'IDENTIFICAZIONE riceve una MATRICE di distanze (una riga per
      probe, una colonna per template della gallery) e i NOMI delle persone
      di righe e colonne, perche' qui serve sapere se il piu' vicino e'
      quello giusto. Le matrici le costruisce eval/matrices.cosine_distance_matrix.

MAPPA DELLE CHIAMATE
    verifica:        _as_array <- default_thresholds <- far_frr <- roc_curve
                                                              <- det_curve
                                                              <- eer
                     _as_array <- decidability
    identificazione: _per_identity, _true_rank_and_distance <- cms_curve
                                                            <- open_set_rates <- dir_at_false_alarm
    affidabilita':   error_reject_curve <- prr;  auroc
    Chi usa questo file da fuori:
      scripts/eval_lfw.py -> eer, decidability, cms_curve, dir_at_false_alarm
      scripts/eval_srr.py -> _per_identity, _true_rank_and_distance, error_reject_curve, prr, auroc
      tests/test_metrics.py, tests/test_reliability_metrics.py
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Verifica 1:1
# ---------------------------------------------------------------------------

def _as_array(x) -> np.ndarray:
    """Trasforma una lista (o array di qualsiasi forma) in un array 1D di float.

    ESEMPIO : [0.1, 0.2] -> array([0.1, 0.2]);  [[0.1], [0.2]] -> array([0.1, 0.2])
    SOLLEVA : ValueError se e' vuota (senza punteggi le percentuali sarebbero 0/0)
    CHIAMA  : np.asarray, .ravel (= "appiattisci in una riga")
    CHIAMATA DA : default_thresholds, far_frr, decidability
    """
    a = np.asarray(x, dtype=float).ravel()
    if a.size == 0:
        raise ValueError("serve almeno un punteggio")
    return a


def default_thresholds(genuine, impostor) -> np.ndarray:
    """Tutte le soglie "interessanti" per disegnare ROC/DET e trovare l'EER.

    TEORIA
        FAR e FRR cambiano solo quando la soglia supera uno dei punteggi
        osservati: tra un punteggio e l'altro restano costanti. Quindi basta
        provare come soglie i punteggi stessi, piu' una soglia sotto il
        minimo, dove nessuno viene accettato (FAR = 0, FRR = 1): e' il
        punto di partenza delle curve.

    COME LO CALCOLA IL CODICE
        unisce le due liste, np.unique toglie i doppioni e ordina,
        poi aggiunge in testa (minimo - 1).

    ESEMPIO
        genuine  = [0.1, 0.2, 0.3, 0.6]
        impostor = [0.5, 0.7, 0.8, 0.9]
        -> [-0.9, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
             ^ 0.1 - 1: sotto tutto

    INPUT       : le stesse due liste di distanze di far_frr (vedi sotto)
    RESTITUISCE : array ordinato di soglie, lungo (n. valori diversi + 1)
    CHIAMA      : _as_array, np.unique, np.concatenate
    CHIAMATA DA : far_frr (quando thresholds=None)
    """
    scores = np.unique(np.concatenate([_as_array(genuine), _as_array(impostor)]))
    return np.concatenate([[scores[0] - 1.0], scores])


def far_frr(genuine, impostor, thresholds=None):
    """FAR e FRR per ogni soglia.

    TEORIA (appunti p.14, con distanze al posto di similarita')
        FAR(t) = False Acceptance Rate = impostori ACCETTATI / tentativi impostori
               = #(impostor <= t) / n_impostor
        FRR(t) = False Rejection Rate  = genuini RIFIUTATI / tentativi genuini
               = #(genuine  >  t) / n_genuine
        Alzando t si accetta di piu': FAR sale, FRR scende. Non esiste una
        soglia che li azzeri entrambi se le due distribuzioni si sovrappongono.

    INPUT (cosa sono esattamente i numeri)
        genuine  : lista di distanze, UNA per ogni confronto genuino, cioe'
                   tra due campioni della STESSA persona. Esempio di origine:
                       Mario_foto1 vs Mario_foto2 -> 0.1
                       Anna_foto1  vs Anna_foto2  -> 0.2
                       Luca_foto1  vs Luca_foto2  -> 0.3
                       Sara_foto1  vs Sara_foto2  -> 0.6  (foto venuta male)
                   -> genuine = [0.1, 0.2, 0.3, 0.6]
        impostor : lista di distanze, una per confronto tra persone DIVERSE:
                       Mario vs Anna -> 0.5,  Luca vs Sara -> 0.7,
                       Mario vs Luca -> 0.8,  Anna vs Sara -> 0.9
                   -> impostor = [0.5, 0.7, 0.8, 0.9]
        La funzione riceve SOLO i numeri: i nomi qui sopra non le arrivano e
        non le servono. L'unica informazione sull'identita' e' in QUALE delle
        due liste sta ogni numero, e quella scelta la fa chi chiama usando le
        etichette (in eval_lfw.py: dist[same] e dist[~same], da pairs.txt).
        Le due liste possono avere lunghezze diverse (in un sistema reale gli
        impostori sono molti di piu').
        thresholds : soglie da provare; None = default_thresholds.

    COME LO CALCOLA IL CODICE
        Ordina le due liste. Su una lista ordinata
        np.searchsorted(lista, t, side="right") dice in che posizione
        andrebbe inserito t lasciandola ordinata = quanti valori sono <= t.
        Lo fa per tutte le soglie insieme, senza cicli.
            impostor ordinati [0.5, 0.7, 0.8, 0.9], t = 0.6 -> 1 (solo 0.5 <= 0.6)
        FAR = quel conteggio / n impostori; FRR = 1 - (genuini <= t) / n genuini.

    ESEMPIO (le liste sopra)
        soglia   FAR    FRR
        -0.9     0.00   1.00   nessuno accettato
         0.3     0.00   0.25   rifiutato solo 0.6 (Sara)
         0.5     0.25   0.25   accettato l'impostore 0.5 -> qui FAR = FRR
         0.6     0.25   0.00   ora anche Sara e' accettata
         0.9     1.00   0.00   tutti accettati

    RESTITUISCE : (thresholds, far, frr), tre array della stessa lunghezza;
                  far[k] e frr[k] si riferiscono a thresholds[k]
    CHIAMA      : _as_array, default_thresholds, np.sort, np.searchsorted
    CHIAMATA DA : roc_curve, det_curve, eer, tests/test_metrics.py
    """
    g = np.sort(_as_array(genuine))       # es. [0.1, 0.2, 0.3, 0.6]
    i = np.sort(_as_array(impostor))      # es. [0.5, 0.7, 0.8, 0.9]
    t = default_thresholds(g, i) if thresholds is None else np.asarray(thresholds, dtype=float)
    # searchsorted(..., side="right") conta quanti valori sono <= t, senza cicli
    far = np.searchsorted(i, t, side="right") / i.size        # es. t=0.5 -> 1/4 = 0.25
    frr = 1.0 - np.searchsorted(g, t, side="right") / g.size  # es. t=0.5 -> 1 - 3/4 = 0.25
    return t, far, frr


def roc_curve(genuine, impostor, thresholds=None):
    """Curva ROC: per ogni soglia, il punto (FAR, GAR).

    TEORIA (appunti pp.35-36)
        GAR = Genuine Acceptance Rate = genuini accettati / genuini = 1 - FRR.
        La ROC mette FAR sull'asse x e GAR sull'asse y: un sistema perfetto
        passa per l'angolo (0, 1), cioe' nessun impostore e tutti i genuini.

    ESEMPIO (liste di far_frr): soglia 0.5 -> FAR 0.25, GAR 1 - 0.25 = 0.75

    INPUT       : come far_frr
    RESTITUISCE : (thresholds, far, gar)
    CHIAMA      : far_frr
    CHIAMATA DA : tests/test_metrics.py (in futuro: grafici della relazione)
    """
    t, far, frr = far_frr(genuine, impostor, thresholds)
    return t, far, 1.0 - frr


def det_curve(genuine, impostor, thresholds=None):
    """Curva DET: per ogni soglia, il punto (FAR, FRR).

    TEORIA
        Stessa informazione della ROC, ma con i due ERRORI sugli assi, di
        solito in scala logaritmica: cosi' si vedono bene le differenze tra
        sistemi molto buoni (errori dell'1% contro lo 0.1%).
        Il punto in cui la curva incrocia la diagonale FAR = FRR e' l'EER.

    INPUT       : come far_frr
    RESTITUISCE : (thresholds, far, frr) - identico a far_frr
    CHIAMA      : far_frr
    CHIAMATA DA : per ora nessuno (servira' per i grafici della relazione)
    """
    return far_frr(genuine, impostor, thresholds)


def eer(genuine, impostor):
    """Equal Error Rate: il valore dell'errore nel punto in cui FAR = FRR.

    TEORIA (appunti p.21)
        Un solo numero per riassumere il sistema: la soglia in cui i due
        errori si equivalgono. Piu' e' basso, meglio e'.

    COME LO CALCOLA IL CODICE
        1. calcola FAR e FRR su tutte le soglie (far_frr)
        2. diff = FAR - FRR: all'inizio e' negativo (FAR 0, FRR 1), alla fine
           positivo; si cerca il PRIMO indice con diff >= 0.
           np.argmax su un array di True/False da' la posizione del primo True.
        3. se in quel punto diff vale esattamente 0, l'EER e' li'.
           Altrimenti FAR = FRR cade tra il punto prima (diff < 0) e quello
           (diff > 0): si interpola con una retta tra i due.

    ESEMPIO 1 (incrocio esatto)
        genuine [0.1, 0.2, 0.3, 0.6], impostor [0.5, 0.7, 0.8, 0.9]
        alla soglia 0.5: FAR 0.25 = FRR 0.25 -> restituisce (0.25, 0.5)

    ESEMPIO 2 (serve l'interpolazione)
        genuine [0.1, 0.2, 0.3, 0.6], impostor [0.5, 0.7, 0.8, 0.9, 0.95]
            soglia 0.5: FAR 0.20, FRR 0.25 -> diff = -0.05
            soglia 0.6: FAR 0.20, FRR 0.00 -> diff = +0.20   <- primo >= 0
        alpha = 0.05 / (0.20 - (-0.05)) = 0.05 / 0.25 = 0.2
              = "a che frazione del tratto diff passa per 0"
        EER    = 0.20 + 0.2 * (0.20 - 0.20) = 0.20
        soglia = 0.5  + 0.2 * (0.6  - 0.5 ) = 0.52
        -> restituisce (0.2, 0.52)

    INPUT       : come far_frr (due liste di distanze genuine/impostore)
    RESTITUISCE : tupla (eer, soglia_eer), due float. Es. (0.25, 0.5)
                  -> in eval_lfw.py: rate, thr = eer(...)
    CHIAMA      : far_frr, np.argmax
    CHIAMATA DA : scripts/eval_lfw.py, tests/test_metrics.py
    """
    t, far, frr = far_frr(genuine, impostor)
    diff = far - frr                           # es. [-1, -0.75, -0.5, -0.25, -0.05, 0.2, ...]
    idx = int(np.argmax(diff >= 0))           # primo indice con FAR >= FRR (es. 5)
    if diff[idx] == 0 or idx == 0:            # incrocio esatto (o gia' al primo punto)
        return float((far[idx] + frr[idx]) / 2), float(t[idx])
    # interpolazione tra il punto idx-1 (FAR < FRR) e il punto idx (FAR > FRR)
    alpha = -diff[idx - 1] / (diff[idx] - diff[idx - 1])          # es. 0.05 / 0.25 = 0.2
    rate = far[idx - 1] + alpha * (far[idx] - far[idx - 1])       # es. 0.2
    threshold = t[idx - 1] + alpha * (t[idx] - t[idx - 1])        # es. 0.5 + 0.2*0.1 = 0.52
    return float(rate), float(threshold)


def decidability(genuine, impostor) -> float:
    """Indice d' (decidability, Daugman): quanto sono separate le due distribuzioni.

    TEORIA (appunti p.46)
        Se si fa l'istogramma delle distanze genuine e di quelle impostore si
        ottengono due "campane". d' misura la distanza tra i loro centri in
        unita' di larghezza delle campane:
            d' = |media_genuini - media_impostori| / sqrt((var_genuini + var_impostori) / 2)
        d' grande = campane lontane e strette = facile decidere.
        Diversamente dall'EER non dipende da nessuna soglia.

    ESEMPIO
        genuine  [0.1, 0.2, 0.3, 0.6]  media 0.300  varianza 0.0350
        impostor [0.5, 0.7, 0.8, 0.9]  media 0.725  varianza 0.0219
        d' = |0.300 - 0.725| / sqrt((0.0350 + 0.0219) / 2)
           = 0.425 / sqrt(0.02844)
           = 0.425 / 0.1686  = 2.52
        (su LFW --quick avevamo d' = 8.25: campane separatissime)

    INPUT       : come far_frr (due liste di distanze genuine/impostore)
    RESTITUISCE : float >= 0
    NOTA        : varianza divisa per n (np.var di default, ddof=0), non per n-1
    CHIAMA      : _as_array, .mean, .var, np.sqrt
    CHIAMATA DA : scripts/eval_lfw.py, tests/test_metrics.py
    """
    g, i = _as_array(genuine), _as_array(impostor)
    return float(abs(g.mean() - i.mean()) / np.sqrt((g.var() + i.var()) / 2))


# ---------------------------------------------------------------------------
# Identificazione
# ---------------------------------------------------------------------------
#
# Esempio usato in tutte le funzioni qui sotto.
#
# GALLERY (4 template, colonne): Mario, Anna, Mario, Luca  (Mario ha 2 template)
# PROBE   (3 tentativi, righe):  Mario, Anna, Luca         (chi e' davvero, da etichette)
#
# dist (3 x 4)        Mario  Anna  Mario  Luca
#   probe Mario       0.2    0.9   0.3    0.7
#   probe Anna        0.8    0.6   0.4    0.5
#   probe Luca        0.9    0.85  0.7    0.1
#
# probe_ids   = ["Mario", "Anna", "Luca"]          (riga per riga)
# gallery_ids = ["Mario", "Anna", "Mario", "Luca"] (colonna per colonna)
#
# Qui i nomi SERVONO: per dire se il sistema ha riconosciuto bene, bisogna
# sapere chi e' il probe e a chi appartiene ogni colonna.

def _per_identity(dist, gallery_ids):
    """Da matrice probe x TEMPLATE a matrice probe x PERSONA.

    TEORIA
        Una persona puo' avere piu' template in gallery. Per decidere quanto
        un probe e' vicino a quella persona si prende il template piu'
        vicino (distanza minima).

    COME LO CALCOLA IL CODICE
        per ogni persona g (in ordine alfabetico, da np.unique):
            gallery_ids == g        -> maschera delle colonne di g
            dist[:, maschera].min(axis=1) -> minimo su quelle colonne, per ogni riga
        poi np.stack mette le colonne una accanto all'altra.

    ESEMPIO (matrice d'esempio sopra)
        ids = ["Anna", "Luca", "Mario"]
        reduced          Anna  Luca  Mario
          probe Mario    0.9   0.7   0.2    <- min(0.2, 0.3)
          probe Anna     0.6   0.5   0.4    <- min(0.8, 0.4)
          probe Luca     0.85  0.1   0.7    <- min(0.9, 0.7)

    INPUT
        dist        : matrice (n_probe x n_template) di distanze
        gallery_ids : nome della persona di ogni COLONNA, stessa lunghezza
                      del numero di colonne
    RESTITUISCE : (ids, reduced) - nomi delle persone in ordine alfabetico e
                  matrice (n_probe x n_persone)
    CHIAMA      : np.unique, .min, np.stack
    CHIAMATA DA : cms_curve, open_set_rates
    """
    dist = np.asarray(dist, dtype=float)
    gallery_ids = np.asarray(gallery_ids)
    ids = np.unique(gallery_ids)              # es. ["Anna", "Luca", "Mario"]
    reduced = np.stack([dist[:, gallery_ids == g].min(axis=1) for g in ids], axis=1)
    return ids, reduced


def _true_rank_and_distance(reduced, ids, probe_ids):
    """Per ogni probe: a che posto in classifica arriva la persona GIUSTA, e a che distanza.

    TEORIA
        Il sistema ordina le persone dalla piu' vicina alla piu' lontana.
        Il RANGO della persona giusta e' la sua posizione: 1 = e' la prima.
        Rango = 1 + quante persone SBAGLIATE sono strettamente piu' vicine.
        In caso di parita' il probe e' premiato (e' la scelta piu' comune).

    COME LO CALCOLA IL CODICE
        col       : dizionario nome -> numero di colonna, es. {"Anna":0, "Luca":1, "Mario":2}
        true_col  : per ogni probe, la colonna della sua persona vera
        true_dist : reduced[riga, colonna giusta] per ogni riga
        rank      : conta per ogni riga quante celle sono < true_dist, + 1

    ESEMPIO (reduced di _per_identity)
        probe Mario: giusta = Mario 0.2; piu' vicine: nessuna      -> rango 1
        probe Anna : giusta = Anna  0.6; piu' vicine: Luca 0.5, Mario 0.4 -> rango 3
        probe Luca : giusta = Luca  0.1; piu' vicine: nessuna      -> rango 1
        -> rank = [1, 3, 1], true_dist = [0.2, 0.6, 0.1]

    INPUT
        reduced   : matrice probe x persona (da _per_identity)
        ids       : nomi delle colonne di reduced (da _per_identity)
        probe_ids : nome VERO di ogni RIGA (dalle etichette); ogni nome deve
                    stare in ids, altrimenti KeyError (cms_curve lo controlla prima)
    RESTITUISCE : (rank, true_dist), due array lunghi quanto i probe
    CHIAMA      : np.arange
    CHIAMATA DA : cms_curve, open_set_rates
    """
    col = {g: k for k, g in enumerate(ids)}              # es. {"Anna": 0, "Luca": 1, "Mario": 2}
    rows = np.arange(len(probe_ids))                      # es. [0, 1, 2]
    true_col = np.array([col[p] for p in probe_ids])      # es. [2, 0, 1]
    true_dist = reduced[rows, true_col]                   # es. [0.2, 0.6, 0.1]
    rank = 1 + (reduced < true_dist[:, None]).sum(axis=1)  # es. [1, 3, 1]
    return rank, true_dist


def cms_curve(dist, probe_ids, gallery_ids):
    """Identificazione closed-set: Cumulative Match Score per ogni rango.

    TEORIA (appunti p.23)
        CMS(k) = quota di probe la cui persona giusta e' entro le prime k
                 posizioni della classifica.
        CMS(1) = Recognition Rate (RR): la prima risposta e' giusta.
        La curva CMC e' il grafico di CMS(k) al variare di k; arriva sempre
        a 1 quando k = numero di persone.
        "Closed-set" = si suppone che ogni probe sia di una persona registrata.

    COME LO CALCOLA IL CODICE
        1. _per_identity: da template a persone
        2. controlla che ogni probe sia di una persona in gallery
        3. _true_rank_and_distance: rango della persona giusta per ogni probe
        4. per ogni k = 1..n_persone: quota di ranghi <= k (tutti i k insieme
           con un confronto tra matrici: righe = k, colonne = probe)

    ESEMPIO (matrice d'esempio sopra): rank = [1, 3, 1]
        CMS(1) = 2/3 = 0.667   (Mario e Luca primi)
        CMS(2) = 2/3 = 0.667   (Anna e' terza, non ancora presa)
        CMS(3) = 3/3 = 1.0
        -> array([0.667, 0.667, 1.0])

    INPUT
        dist        : matrice (n_probe x n_template) di distanze,
                      dist[p, g] = distanza tra il probe p e il template g
        probe_ids   : nome VERO della persona di ogni riga (etichette)
        gallery_ids : nome della persona di ogni colonna
    RESTITUISCE : array cms con cms[k-1] = CMS(k), lungo quanto le persone in gallery
    SOLLEVA     : ValueError se un probe e' di una persona non in gallery
    CHIAMA      : _per_identity, _true_rank_and_distance
    CHIAMATA DA : scripts/eval_lfw.py, tests/test_metrics.py
    """
    ids, reduced = _per_identity(dist, gallery_ids)
    missing = set(np.asarray(probe_ids)) - set(ids)
    if missing:
        raise ValueError(f"closed-set: probe di identita' non in gallery: {sorted(missing)[:5]}")
    rank, _ = _true_rank_and_distance(reduced, ids, probe_ids)
    ks = np.arange(1, len(ids) + 1)                       # es. [1, 2, 3]
    # tabella (k x probe): rank <= k ?   poi media su ogni riga
    return (rank[None, :] <= ks[:, None]).mean(axis=1)


def open_set_rates(dist_known, known_ids, dist_unknown, gallery_ids, thresholds, rank=1):
    """Identificazione open-set: DIR(t, k) e false alarm rate per ogni soglia.

    TEORIA (appunti p.21)
        Open-set = puo' presentarsi qualcuno che NON e' registrato. Ora oltre
        al rango serve una soglia: se nessuno e' abbastanza vicino si risponde
        "sconosciuto".
        DIR(t, k) = Detection and Identification Rate = quota di probe
                    REGISTRATI la cui persona giusta e' entro il rango k
                    E ha distanza <= t (riconosciuto e accettato)
        FA(t)     = false alarm = quota di probe SCONOSCIUTI che vengono
                    accettati, cioe' con distanza MINIMA dalla gallery <= t
                    (la posizione non conta: basta che qualcuno li accetti)
        FRR(t)    = 1 - DIR(t, 1)

    COME LO CALCOLA IL CODICE
        registrati : rango e distanza giusta (come in cms_curve), poi per ogni
                     soglia conta quelli con d_true <= t E rango <= k
        sconosciuti: per ogni riga la distanza minima (.min(axis=1)), poi per
                     ogni soglia conta quelli <= t

    ESEMPIO (matrice d'esempio sopra come dist_known, piu' 2 sconosciuti)
        dist_unknown       Mario  Anna  Mario  Luca
          sconosciuto 1    0.7    0.9   0.8    0.6   -> minimo 0.6
          sconosciuto 2    0.95   0.3   0.9    0.85  -> minimo 0.3
        registrati: rank [1, 3, 1], d_true [0.2, 0.6, 0.1]; rango 1 ok per Mario e Luca
        soglia 0.2 : DIR = 2/3 (Mario 0.2, Luca 0.1)   FA = 0/2 = 0
        soglia 0.35: DIR = 2/3                          FA = 1/2 (0.3 accettato)
        soglia 0.5 : DIR = 2/3 (Anna e' rango 3: mai)   FA = 1/2

    INPUT
        dist_known   : matrice (n_registrati x n_template), righe = probe di
                       persone che HANNO un template in gallery
        known_ids    : nome VERO di ogni riga di dist_known
        dist_unknown : matrice (n_sconosciuti x n_template), righe = probe di
                       persone SENZA template in gallery (per loro non serve il
                       nome: qualunque accettazione e' sbagliata)
        gallery_ids  : nome della persona di ogni colonna (uguale per le due matrici)
        thresholds   : lista di soglie da provare
        rank         : k di DIR(t, k), di solito 1
    RESTITUISCE : (thresholds, dir_rates, false_alarm_rates), tre array uguali di lunghezza
    CHIAMA      : _per_identity, _true_rank_and_distance
    CHIAMATA DA : dir_at_false_alarm, tests/test_metrics.py
    """
    t = np.asarray(thresholds, dtype=float)
    ids, reduced = _per_identity(dist_known, gallery_ids)
    r, d_true = _true_rank_and_distance(reduced, ids, known_ids)   # es. [1,3,1], [0.2,0.6,0.1]
    ok_rank = r <= rank                                             # es. [True, False, True]
    # tabella (soglia x probe): accettato E rango giusto?  poi media su ogni riga
    dir_rates = ((d_true[None, :] <= t[:, None]) & ok_rank[None, :]).mean(axis=1)
    best_unknown = np.asarray(dist_unknown, dtype=float).min(axis=1)  # es. [0.6, 0.3]
    fa_rates = (best_unknown[None, :] <= t[:, None]).mean(axis=1)
    return t, dir_rates, fa_rates


def dir_at_false_alarm(dist_known, known_ids, dist_unknown, gallery_ids, target_fa, rank=1):
    """DIR alla soglia piu' alta che tiene il false alarm <= target_fa.

    TEORIA
        In un sistema reale prima si decide quanti falsi allarmi si
        tollerano (es. 1 sconosciuto su 100 = 1%), poi si guarda quanti
        utenti veri vengono riconosciuti a quella soglia. E' il numero che
        eval_lfw.py stampa come "false alarm 1% -> DIR ...".

    COME LO CALCOLA IL CODICE
        1. distanza minima di ogni sconosciuto, ordinata
        2. allowed = quanti sconosciuti si possono accettare = floor(target * n)
        3. soglia = la distanza dell'allowed-esimo sconosciuto (cosi' ne
           passano esattamente allowed); se allowed = 0, un numero appena
           sotto il piu' piccolo (np.nextafter), cosi' non ne passa nessuno
        4. open_set_rates con quella sola soglia

    ESEMPIO (sconosciuti con minimi [0.6, 0.3] -> ordinati [0.3, 0.6], n = 2)
        target 0.5: allowed = floor(0.5 * 2) = 1 -> soglia 0.3
                    -> (DIR 0.667, soglia 0.3, FA 0.5)
        target 0.1: allowed = floor(0.1 * 2) = 0 -> soglia appena sotto 0.3
                    -> (DIR 0.667, soglia 0.29999..., FA 0.0)

    INPUT       : come open_set_rates; target_fa = false alarm massimo, es. 0.01
    RESTITUISCE : (dir, soglia, false_alarm_effettivo), tre float.
                  Il FA effettivo puo' essere < target (es. con 150 sconosciuti
                  e target 1%: allowed = 1 -> FA = 1/150 = 0.67%)
    CHIAMA      : open_set_rates, np.sort, np.floor, np.nextafter
    CHIAMATA DA : scripts/eval_lfw.py, tests/test_metrics.py
    """
    best_unknown = np.sort(np.asarray(dist_unknown, dtype=float).min(axis=1))  # es. [0.3, 0.6]
    n = best_unknown.size
    allowed = int(np.floor(target_fa * n))     # quanti falsi allarmi sono tollerati
    if allowed == 0:
        threshold = np.nextafter(best_unknown[0], -np.inf)   # il float subito sotto il minimo
    else:
        threshold = best_unknown[allowed - 1]                # -1 perche' gli indici partono da 0
    _, dir_rates, fa_rates = open_set_rates(dist_known, known_ids, dist_unknown,
                                            gallery_ids, [threshold], rank)
    return float(dir_rates[0]), float(threshold), float(fa_rates[0])


# ---------------------------------------------------------------------------
# Valutazione di un criterio di affidabilita' (Fase 5)
# ---------------------------------------------------------------------------
#
# Qui non si valuta il riconoscimento, ma un CRITERIO che dice quanto e'
# affidabile ogni risposta (SRR, margine, ...). Gli input sono sempre:
#   scores  = valore del criterio per ogni risposta, PIU' ALTO = PIU' AFFIDABILE
#   correct = True se la risposta era giusta (dalle etichette)
# Nessuna soglia: si guarda il criterio a TUTTE le soglie insieme.
#
# Esempio usato nelle funzioni qui sotto (5 risposte, G = giusta, S = sbagliata):
#   scores  = [0.9, 0.8, 0.6, 0.4, 0.2]
#   correct = [ G,   G,   S,   G,   S ]

def error_reject_curve(scores, correct):
    """Curva errore contro rifiuto: quanti errori restano se si rifiutano le risposte meno affidabili.

    TEORIA
        Si mettono in fila le risposte dalla MENO alla PIU' affidabile e si
        immagina di rifiutarne (nel login: mandarle allo step-up) una quota
        crescente, partendo dal fondo della fila. Per ogni quota r si guarda
        l'errore che RESTA tra quelle accettate, contato su tutte le risposte:
            errore(r) = risposte sbagliate non ancora rifiutate / N
        Parte dalla quota di errori totale (r = 0) e arriva a 0 (r = 1).
        Un buon criterio la fa scendere in fretta: mette gli errori in fondo.

    COME LO CALCOLA IL CODICE
        Ordina i valori distinti del criterio dal piu' piccolo. Risposte con lo
        STESSO valore non hanno un ordine tra loro: vengono rifiutate tutte
        insieme, e tra un gruppo e il successivo la curva e' una retta (e'
        quello che succederebbe in media rifiutandole in ordine casuale).
        Molto importante per il density ratio, che ha moltissimi valori uguali.

    ESEMPIO
        rifiutate   0      1      2      3      4      5
        quota r     0.0    0.2    0.4    0.6    0.8    1.0
        rifiutata   -      0.2 S  0.4 G  0.6 S  0.8 G  0.9 G
        errore      0.40   0.20   0.20   0.00   0.00   0.00

    INPUT       : scores (piu' alto = piu' affidabile), correct (True/False), stessa lunghezza
    RESTITUISCE : (r, errore), due array: i punti della curva, da r = 0 a r = 1
    CHIAMATA DA : prr, scripts/eval_srr.py, tests/test_reliability_metrics.py
    """
    s = _as_array(scores)
    c = np.asarray(correct, dtype=bool).ravel()
    if c.size != s.size:
        raise ValueError("scores e correct devono avere la stessa lunghezza")
    n = s.size
    values, group = np.unique(s, return_inverse=True)            # valori distinti, dal piu' piccolo
    size = np.bincount(group, minlength=values.size)              # quante risposte per valore
    wrong = np.bincount(group, weights=~c, minlength=values.size)  # quante sbagliate per valore
    rejected = np.concatenate([[0], np.cumsum(size)])             # rifiutate dopo ogni gruppo
    wrong_left = c.size - c.sum() - np.concatenate([[0], np.cumsum(wrong)])
    return rejected / n, wrong_left / n


def prr(scores, correct) -> float:
    """PRR (Prediction Rejection Ratio): la curva errore-rifiuto riassunta in un numero.

    TEORIA
        Si confronta l'area sotto la curva del criterio con due riferimenti
        (e0 = quota di risposte sbagliate):
            a caso   : rifiutando a caso l'errore cala in linea retta, area = e0 / 2
            oracolo  : rifiuta prima tutte le sbagliate, area = e0^2 / 2
        PRR = (area a caso - area criterio) / (area a caso - area oracolo)
        1 = criterio perfetto, 0 = non meglio del caso (negativo = peggio del caso).

    ESEMPIO (curva di error_reject_curve, e0 = 0.4, punti ogni 0.2)
        area criterio = 0.2 * [(0.40+0.20)/2 + (0.20+0.20)/2 + (0.20+0.00)/2] = 0.12
        area a caso   = 0.40 / 2   = 0.20
        area oracolo  = 0.40^2 / 2 = 0.08
        PRR = (0.20 - 0.12) / (0.20 - 0.08) = 0.08 / 0.12 = 0.67

    INPUT       : come error_reject_curve
    RESTITUISCE : float; nan se le risposte sono tutte giuste o tutte sbagliate
                  (non c'e' niente da separare)
    CHIAMA      : error_reject_curve (l'area si calcola con i trapezi)
    CHIAMATA DA : scripts/eval_srr.py, tests/test_reliability_metrics.py
    """
    r, err = error_reject_curve(scores, correct)
    e0 = err[0]
    if e0 == 0 or e0 == 1:
        return float("nan")
    area = float(np.sum((err[1:] + err[:-1]) / 2 * np.diff(r)))   # trapezi
    area_random, area_oracle = e0 / 2, e0 ** 2 / 2
    return float((area_random - area) / (area_random - area_oracle))


def auroc(scores_correct, scores_wrong) -> float:
    """AUROC: probabilita' che una risposta giusta abbia un valore piu' alto di una sbagliata.

    TEORIA
        Si prendono tutte le coppie (giusta, sbagliata) e si conta in quante il
        criterio da' il valore piu' alto alla giusta; i pareggi contano 1/2.
        1 = separa perfettamente, 0.5 = come tirare a caso. E' l'area sotto la
        curva ROC del criterio (da cui il nome), ma si calcola cosi' senza soglie.

    COME LO CALCOLA IL CODICE
        Ordina le sbagliate; per ogni giusta, np.searchsorted conta in un colpo
        quante sbagliate sono strettamente sotto (side="left") e quante sono
        uguali (differenza tra side="right" e side="left").

    ESEMPIO
        giuste {0.9, 0.8, 0.4}, sbagliate {0.6, 0.2}: 6 coppie
        0.9 > 0.6 si'   0.9 > 0.2 si'
        0.8 > 0.6 si'   0.8 > 0.2 si'
        0.4 > 0.6 no    0.4 > 0.2 si'       AUROC = 5/6 = 0.83

    INPUT       : due liste di valori del criterio (giuste e sbagliate), non vuote
    RESTITUISCE : float in [0, 1]
    CHIAMATA DA : scripts/eval_srr.py, tests/test_reliability_metrics.py
    """
    pos = _as_array(scores_correct)
    neg = np.sort(_as_array(scores_wrong))
    below = np.searchsorted(neg, pos, side="left")
    ties = np.searchsorted(neg, pos, side="right") - below
    return float(np.mean((below + 0.5 * ties) / neg.size))
