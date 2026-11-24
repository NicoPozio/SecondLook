"""LBP a griglia (Local Binary Patterns): il descrittore "classico" del volto.

TEORIA (appunti, 2D Face Recognition, par. LBP)
    E' la baseline NON deep del progetto, da confrontare con ArcFace
    (domanda di ricerca D3: feature classiche contro feature deep).
    Nessuna rete neurale e nessun addestramento: il vettore si calcola con
    regole fisse, scelte a mano.

    1. CODICE LBP DI UN PIXEL. Si guarda il quadrato 3 x 3 intorno al pixel.
       Il pixel centrale fa da soglia: ogni vicino diventa 1 se e' >= del
       centro, 0 se e' piu' scuro. Gli 8 bit, letti in un ordine fisso, sono
       un numero da 0 a 255:
           LBP = somma su p = 0..7 di  s(g_p - g_c) * 2^p
           s(x) = 1 se x >= 0, altrimenti 0
       (g_c = valore del centro, g_p = valore del vicino p)
       Il codice descrive la "forma" locale dell'immagine: zona piatta,
       bordo, angolo, punto chiaro, ... e NON la luminosita' assoluta.
       Per questo LBP resiste bene ai cambi di illuminazione uniformi.

    2. PATTERN UNIFORMI. Un codice e' "uniforme" se, girando in tondo sugli
       8 bit, passa da 0 a 1 o da 1 a 0 al massimo 2 volte. Sono 58 codici
       su 256 e corrispondono alle strutture utili (bordi, angoli, zone
       piatte); tutti gli altri finiscono in un'unica casella "non uniforme".
       Totale: 58 + 1 = 59 caselle invece di 256.

    3. ISTOGRAMMI A GRIGLIA (LBPH). Un solo istogramma di tutto il volto
       perderebbe "dove" sta ogni struttura (occhi, bocca...). Allora si
       divide il volto in una griglia k x k di celle, si conta quante volte
       compare ogni codice in ogni cella e si mettono gli istogrammi uno
       dopo l'altro. Con k = 7 e 59 caselle: 7 * 7 * 59 = 2891 numeri.

    4. DISTANZA CHI-QUADRO tra due vettori LBP (istogrammi):
           chi2(a, b) = somma su i di  (a_i - b_i)^2 / (a_i + b_i)
       (le caselle con a_i + b_i = 0 contano 0). 0 = istogrammi identici.
       Si usa al posto del coseno perche' pesa le differenze in modo
       relativo: una differenza su una casella poco piena conta di piu'.

    Il volto in ingresso e' lo STESSO ritaglio allineato 112 x 112 che usa
    ArcFace (occhi sempre nella stessa posizione): cosi' il confronto tra i
    due metodi riguarda solo il descrittore, non il rilevamento.

COSA C'E' IN QUESTO FILE
    lbp_codes            immagine in grigi -> immagine dei codici LBP (0..255)
    UNIFORM_LOOKUP       tabella codice (0..255) -> casella (0..58)
    lbp_histogram        immagine dei codici -> istogrammi a griglia concatenati
    lbp_descriptor       ritaglio del volto (BGR o grigi) -> vettore LBP
    chi2_distance        distanza chi-quadro tra DUE vettori
    chi2_distance_matrix distanze chi-quadro probe x gallery (a blocchi)

MAPPA DELLE CHIAMATE
    lbp_descriptor -> lbp_codes, lbp_histogram (-> UNIFORM_LOOKUP)
    chi2_distance_matrix -> (stessa formula di chi2_distance, su blocchi)
    Chi usa questo file da fuori: tests/test_lbp.py
    (prossimo passo: scripts/eval_lfw.py con --method lbp)
"""

from __future__ import annotations

import cv2
import numpy as np

# Ordine dei vicini: si parte dall'angolo in alto a sinistra e si gira in
# senso orario. (dy, dx) = spostamento rispetto al pixel centrale.
#
#     p0  p1  p2          (-1,-1) (-1, 0) (-1,+1)
#     p7  c   p3    =     ( 0,-1)    c    ( 0,+1)
#     p6  p5  p4          (+1,-1) (+1, 0) (+1,+1)
#
# Il vicino p ha peso 2^p: p0 vale 1, p1 vale 2, ..., p7 vale 128.
NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]


def lbp_codes(gray: np.ndarray) -> np.ndarray:
    """Immagine dei codici LBP: un numero 0..255 per ogni pixel interno.

    COME LO CALCOLA IL CODICE
        Invece di un doppio ciclo for su tutti i pixel, lavora su "fette"
        dell'immagine: per il vicino (dy, dx) prende l'immagine spostata di
        (dy, dx) e la confronta in un colpo solo con l'immagine dei centri.
        I pixel del bordo esterno non hanno 8 vicini e vengono saltati:
        un'immagine H x W da' codici (H-2) x (W-2). Es. 112 x 112 -> 110 x 110.
        int16: senza, la sottrazione tra numeri uint8 (0..255) "fa il giro"
        (es. 10 - 20 darebbe 246 invece di -10).

    ESEMPIO (un solo pixel, centro 50)
        60  20  70          p0=60 >= 50 -> 1    p1=20 -> 0    p2=70 -> 1
        90  50  40    ->    p3=40 -> 0          p4=50 >= 50 -> 1 (uguale conta 1)
        10  80  50          p5=80 -> 1          p6=10 -> 0    p7=90 -> 1

        bit p0..p7 = 1 0 1 0 1 1 0 1
        codice = 1*1 + 0*2 + 1*4 + 0*8 + 1*16 + 1*32 + 0*64 + 1*128
               = 1 + 4 + 16 + 32 + 128
               = 181
        Zona tutta uguale (es. tutti 50): ogni vicino e' >= -> 11111111 = 255.

    INPUT       : gray = immagine in livelli di grigio, matrice H x W (H, W >= 3)
    RESTITUISCE : matrice (H-2) x (W-2) di uint8, il codice di ogni pixel interno
    CHIAMA      : solo numpy
    CHIAMATA DA : lbp_descriptor, tests/test_lbp.py
    """
    g = np.asarray(gray, dtype=np.int16)
    if g.ndim != 2 or min(g.shape) < 3:
        raise ValueError(f"serve un'immagine in grigi di almeno 3 x 3, arrivata {g.shape}")
    h, w = g.shape
    center = g[1:h - 1, 1:w - 1]                      # tutti i pixel interni
    codes = np.zeros(center.shape, dtype=np.uint8)
    for p, (dy, dx) in enumerate(NEIGHBOURS):
        neighbour = g[1 + dy:h - 1 + dy, 1 + dx:w - 1 + dx]   # stessa forma di center, spostata
        codes |= (neighbour >= center).astype(np.uint8) << p  # bit p acceso dove s(g_p - g_c) = 1
    return codes


def _transitions(code: int) -> int:
    """Quante volte il bit cambia (0->1 o 1->0) girando in tondo sugli 8 bit.

    ESEMPIO : 181 = bit p0..p7 1 0 1 0 1 1 0 1 (e poi si torna a p0 = 1)
              cambi: 1-0, 0-1, 1-0, 0-1, 1-1, 1-0, 0-1, 1-1  -> 6  (non uniforme)
              255 = 11111111 -> 0 cambi (uniforme); 15 = 00001111 -> 2 (uniforme)
    CHIAMATA DA : _uniform_lookup
    """
    bits = [(code >> p) & 1 for p in range(8)]
    return sum(bits[p] != bits[(p + 1) % 8] for p in range(8))


def _uniform_lookup() -> np.ndarray:
    """Tabella da codice LBP (0..255) a casella dell'istogramma (0..58).

    COME LO CALCOLA IL CODICE
        Scorre i codici da 0 a 255: i codici uniformi (<= 2 cambi) ricevono
        le caselle 0, 1, 2, ... nell'ordine in cui si incontrano; tutti gli
        altri ricevono la casella 58.

    ESEMPIO : 0 -> casella 0, 1 -> 1, 2 -> 2, 3 -> 3, 4 -> 4, 5 -> 58 (4 cambi),
              ..., 255 -> 57 (l'ultimo uniforme)
    RESTITUISCE : array di 256 interi
    CHIAMATA DA : definizione di UNIFORM_LOOKUP (una volta, all'import)
    """
    table = np.full(256, 58, dtype=np.uint8)
    next_bin = 0
    for code in range(256):
        if _transitions(code) <= 2:
            table[code] = next_bin
            next_bin += 1
    assert next_bin == 58               # P(P-1) + 2 = 8*7 + 2 = 58 uniformi, come negli appunti
    return table


UNIFORM_LOOKUP = _uniform_lookup()      # calcolata una volta sola


def lbp_histogram(codes: np.ndarray, grid: int = 7, uniform: bool = True) -> np.ndarray:
    """Istogrammi LBP a griglia, concatenati in un unico vettore.

    COME LO CALCOLA IL CODICE
        1. se uniform: ogni codice 0..255 diventa la sua casella 0..58
           (UNIFORM_LOOKUP[codes]); altrimenti si tengono 256 caselle
        2. divide le righe in `grid` fasce e le colonne in `grid` fasce
           (np.array_split: se non si dividono esattamente, alcune fasce
           hanno un pixel in piu'. Es. 110 pixel in 7 fasce: 5 da 16 e 2 da 15)
        3. per ogni cella: np.bincount conta quante volte compare ogni
           casella, poi divide per il numero di pixel della cella, cosi'
           ogni istogramma somma 1 anche se le celle hanno dimensioni diverse
        4. mette gli istogrammi uno dopo l'altro, riga per riga della griglia

    ESEMPIO
        codici 110 x 110, grid = 7, uniform = True
        -> 49 celle, ognuna con 59 caselle -> vettore di 49 * 59 = 2891 numeri
        posizioni 0..58 = cella in alto a sinistra, 59..117 = la cella alla
        sua destra, ..., 2832..2890 = cella in basso a destra.

    INPUT
        codes   : immagine dei codici, da lbp_codes
        grid    : numero di celle per lato (k della griglia k x k)
        uniform : True = 59 caselle per cella, False = 256
    RESTITUISCE : vettore float32 di grid*grid*59 (o grid*grid*256) numeri;
                  ogni blocco di una cella somma 1
    CHIAMA      : UNIFORM_LOOKUP, np.array_split, np.bincount
    CHIAMATA DA : lbp_descriptor, tests/test_lbp.py
    """
    bins = UNIFORM_LOOKUP[codes] if uniform else codes
    n_bins = 59 if uniform else 256
    if grid < 1 or grid > min(bins.shape):
        raise ValueError(f"griglia {grid} x {grid} non valida per codici {bins.shape}")
    hists = []
    for band in np.array_split(bins, grid, axis=0):            # fasce orizzontali
        for cell in np.array_split(band, grid, axis=1):        # celle della fascia
            h = np.bincount(cell.ravel(), minlength=n_bins).astype(np.float32)
            hists.append(h / cell.size)                        # frequenze, somma 1
    return np.concatenate(hists)


def lbp_descriptor(face: np.ndarray, grid: int = 7, uniform: bool = True) -> np.ndarray:
    """Vettore LBP di un volto: e' l'equivalente LBP dell'embedding ArcFace.

    COME LO CALCOLA IL CODICE
        1. se il ritaglio e' a colori (BGR, come lo da' OpenCV) lo porta in
           grigi: LBP guarda solo la luminosita'
        2. lbp_codes -> immagine dei codici
        3. lbp_histogram -> istogrammi della griglia concatenati

    ESEMPIO
        ritaglio allineato 112 x 112 x 3 -> grigi 112 x 112 -> codici 110 x 110
        -> vettore di 2891 numeri (grid 7, uniformi)
        Per confronto, l'embedding ArcFace dello stesso volto ha 512 numeri.

    INPUT
        face    : ritaglio del volto, BGR (H x W x 3) oppure grigi (H x W), uint8.
                  Va dato gia' ritagliato e allineato (come per ArcFace).
        grid, uniform : vedi lbp_histogram
    RESTITUISCE : vettore float32 (vedi lbp_histogram)
    CHIAMA      : cv2.cvtColor, lbp_codes, lbp_histogram
    CHIAMATA DA : tests/test_lbp.py (prossimo passo: eval_lfw.py --method lbp)
    """
    img = np.asarray(face)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return lbp_histogram(lbp_codes(gray), grid=grid, uniform=uniform)


def chi2_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Distanza chi-quadro tra due vettori LBP.

    TEORIA
        chi2(a, b) = somma su i di (a_i - b_i)^2 / (a_i + b_i)
        Le caselle vuote in entrambi (a_i + b_i = 0) contano 0.
        E' 0 se i due istogrammi sono identici e cresce con le differenze.
        Alcuni testi mettono un 1/2 davanti: cambia solo la scala, non
        l'ordine delle distanze, quindi non cambia EER, CMS o DIR.

    ESEMPIO
        a = [0.5, 0.5, 0.0]
        b = [1.0, 0.0, 0.0]
        casella 0: (0.5 - 1.0)^2 / (0.5 + 1.0) = 0.25 / 1.5 = 0.1667
        casella 1: (0.5 - 0.0)^2 / (0.5 + 0.0) = 0.25 / 0.5 = 0.5
        casella 2: entrambe 0                              = 0
        chi2 = 0.1667 + 0.5 + 0 = 0.6667

    INPUT       : a, b = due vettori LBP della stessa lunghezza (da lbp_descriptor).
                  Come per cosine_distance, la funzione non sa di chi sono:
                  genuino o impostore lo sa chi chiama.
    RESTITUISCE : float >= 0. Con istogrammi che sommano 1 per cella, ogni cella
                  contribuisce al massimo 2, quindi il massimo e' 2 * grid * grid
                  (98 con grid = 7). NON e' sulla scala 0..2 del coseno: le soglie
                  di ArcFace non valgono per LBP.
                  Su volti veri (foto di prova t1.jpg, grid 7): stesso volto con
                  rumore aggiunto ~18.5, volti di altre persone ~32-41.
    CHIAMA      : solo numpy
    CHIAMATA DA : tests/test_lbp.py
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    num = (a - b) ** 2
    den = a + b
    # dove den = 0 anche num = 0: si mette 1 al denominatore per non dividere per zero
    return float(np.sum(num / np.where(den > 0, den, 1.0)))


def chi2_distance_matrix(probes: np.ndarray, gallery: np.ndarray, block: int = 16) -> np.ndarray:
    """dist[p, g] = chi-quadro tra il probe p e il template g (versione a matrice).

    TEORIA
        Stesso ruolo di eval/matrices.cosine_distance_matrix, ma per LBP.
        Il coseno si riduce a un prodotto tra matrici; il chi-quadro no
        (c'e' una divisione casella per casella), quindi si calcola
        "a blocchi" di probe: per `block` probe alla volta si costruisce
        una tabella (block x n_gallery x n_caselle) e la si somma.

    COME LO CALCOLA IL CODICE
        per ogni blocco di probe:
            P[:, None, :] - G[None, :, :]  -> differenze di ogni probe del blocco
                                              con ogni template (tutte le caselle)
            stessa cosa con la somma, poi (diff^2 / somma) sommato sulle caselle
        Il blocco serve a non esaurire la memoria. Con LFW: 1.677 template x
        2.891 caselle x 16 probe x 4 byte = circa 310 MB per blocco.
        `block` cambia solo memoria e velocita', non il risultato.

    ESEMPIO (vettori a 3 caselle)
        probes  = [[0.5, 0.5, 0.0]]
        gallery = [[1.0, 0.0, 0.0],
                   [0.5, 0.5, 0.0]]
        dist    = [[0.6667, 0.0]]       probe identico al template 1

    INPUT
        probes  : matrice (n_probe x n_caselle), una riga = vettore LBP di un probe
        gallery : matrice (n_gallery x n_caselle), una riga = vettore LBP di un template
        Come per il coseno, l'ordine delle righe e' quello che le lega alle persone.
        block   : quanti probe elaborare insieme
    RESTITUISCE : matrice float32 (n_probe x n_gallery) di distanze >= 0
    CHIAMA      : solo numpy
    CHIAMATA DA : tests/test_lbp.py (prossimo passo: eval_lfw.py --method lbp)
    """
    P = np.asarray(probes, dtype=np.float32)
    G = np.asarray(gallery, dtype=np.float32)
    out = np.empty((P.shape[0], G.shape[0]), dtype=np.float32)
    for start in range(0, P.shape[0], block):
        p = P[start:start + block, None, :]           # (b, 1, n_caselle)
        num = (p - G[None, :, :]) ** 2                # (b, n_gallery, n_caselle)
        den = p + G[None, :, :]
        out[start:start + block] = np.sum(num / np.where(den > 0, den, 1.0), axis=2)
    return out
