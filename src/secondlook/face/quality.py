"""Indici di qualita' del campione: SP (posa), SI (illuminazione), SY (simmetria).

TEORIA (appunti, cap. 4.4 "Sample quality")
    A differenza di SRR, margine e Weibull, che guardano le DISTANZE dopo il
    confronto con la gallery, gli indici di qualita' guardano la FOTO, prima
    del confronto. Rispondono a "questa foto e' adatta al riconoscimento?".
    Nel progetto servono a vedere se una foto "bella" garantisce una risposta
    affidabile (domanda D1) e a spiegare perche' certi campioni sbagliano.
    Tutti e tre stanno in [0, 1]: 1 = qualita' migliore.

    SP (posa)          SP = alfa*(1 - roll) + beta*(1 - yaw) + gamma*(1 - pitch)
        roll  : angolo della linea tra i centri degli occhi
        yaw   : dalla simmetria, confrontando le distanze dl e dr dei due occhi
                dalla punta del naso: in un volto frontale sono uguali
        pitch : dal rapporto tra distanza occhi-naso e distanza naso-mento
    SI (illuminazione) SI = 1 - F(std(m_c)),  m_c = grigio medio di regioni del volto
    SY (simmetria)     SY = somma di sym(P_i, P_j) su coppie di punti simmetrici

SCELTE NOSTRE (gli appunti non le danno; vanno dichiarate nella relazione)
    alfa = beta = gamma = 1/3
    roll normalizzato  = |angolo| / 45 gradi, al massimo 1
    yaw normalizzato   = |dl - dr| / (dl + dr)                   (gia' in [0, 1])
    pitch normalizzato = |r - r0| / r0, al massimo 1, con r = (occhi-naso)/(naso-mento)
                         e r0 = valore "frontale", stimato sulle persone di tuning
    regioni di SI      = fronte, guancia sinistra, guancia destra, naso, mento:
                         rettangoli fissi sul ritaglio ALLINEATO 112 x 112
    F(x)               = x / 127.5 (la deviazione standard piu' grande possibile
                         per medie tra 0 e 255): SI resta in [0, 1] senza tarare nulla
    sym(P_i, P_j)      = 1 - |P_i - riflesso(P_j)| / distanza tra gli occhi, minimo 0
    SY                 = MEDIA (non somma) su 48 coppie: resta in [0, 1]

PUNTI DEL VOLTO
    kps5  : i 5 punti del detector (occhio sx, occhio dx, naso, bocca sx, bocca dx)
    lm106 : i 106 punti del modello 2d106det di buffalo_l. Ricavati su 300 volti
            di LFW: il mento e' il punto 0, il contorno 1..16 e' lo specchio di
            17..32, e ci sono 48 coppie simmetriche (SYMMETRIC_PAIRS).

COSA C'E' IN QUESTO FILE
    pose_measures   roll e yaw normalizzati e rapporto r per il pitch (foto originale)
    pitch_norm      r -> pitch normalizzato, dato r0
    score_pose      SP
    score_illumination  SI, sul ritaglio allineato
    align_points    porta punti della foto originale nel ritaglio allineato
    score_symmetry  SY, sui 106 punti allineati
    quality_measures    le misure grezze di una foto: [roll, yaw, r, SI, SY]

MAPPA DELLE CHIAMATE
    quality_measures -> pose_measures, score_illumination, align_points, score_symmetry
    (SP si calcola dopo, in eval_srr.py, perche' serve r0 stimato sul tuning:
     score_pose(roll, yaw, pitch_norm(r, r0)))
    Chi usa questo file da fuori: eval/matrices.embed_rows (method="quality"),
    scripts/eval_srr.py, tests/test_quality.py
"""

from __future__ import annotations

import cv2
import numpy as np

CHIN = 0                                     # indice del mento nei 106 punti
# 48 coppie (punto, suo specchio): contorno, occhi, sopracciglia, bocca, naso
SYMMETRIC_PAIRS = (
    [(i, i + 16) for i in range(1, 17)]                                       # contorno del viso
    + [(33, 87), (34, 88), (35, 93), (36, 91), (37, 90), (38, 92),
       (39, 89), (40, 94), (41, 96), (42, 95)]                                 # occhi
    + [(43, 101), (44, 100), (45, 99), (46, 97), (47, 98),
       (48, 105), (49, 104), (50, 102), (51, 103)]                             # sopracciglia
    + [(52, 61), (54, 57), (55, 58), (56, 59), (63, 67), (64, 68), (65, 69), (66, 70)]   # bocca
    + [(75, 81), (76, 82), (77, 83), (78, 84), (79, 85)]                       # naso
)
# posizioni di riferimento dei 5 punti nel ritaglio 112 x 112 (le stesse di insightface)
TEMPLATE_EYES = np.array([[38.2946, 51.6963], [73.5318, 51.5014]])
AXIS_X = TEMPLATE_EYES[:, 0].mean()                                  # 55.9: asse verticale del viso
EYE_DISTANCE = TEMPLATE_EYES[1, 0] - TEMPLATE_EYES[0, 0]             # 35.2 pixel
# regioni di SI sul ritaglio allineato: (x1, y1, x2, y2), destra e sinistra NELL'IMMAGINE
SI_REGIONS = {
    "fronte": (40, 20, 72, 36),
    "guancia_sinistra": (22, 62, 36, 82),
    "guancia_destra": (76, 62, 90, 82),
    "naso": (50, 56, 62, 70),
    "mento": (46, 100, 66, 114),
}
WEIGHTS = (1 / 3, 1 / 3, 1 / 3)              # alfa, beta, gamma


def pose_measures(kps5, lm106):
    """Roll e yaw normalizzati e rapporto r per il pitch, sulla foto ORIGINALE.

    COME LO CALCOLA IL CODICE
        occhi e naso dai 5 punti del detector, mento dal punto 0 dei 106
        roll  = angolo della linea tra gli occhi, |angolo| / 45, massimo 1
        yaw   = |dl - dr| / (dl + dr), dl e dr = distanze degli occhi dal naso
        r     = distanza (meta' degli occhi -> naso) / distanza (naso -> mento)
        Si usa la foto originale e non il ritaglio: nel ritaglio il roll e'
        gia' stato corretto dall'allineamento e varrebbe sempre circa 0.

    ESEMPIO
        occhi (40, 50) e (80, 50), naso (60, 70), mento (60, 120)
        roll: occhi alla stessa altezza -> angolo 0 -> 0
        yaw : dl = dr = radice(20^2 + 20^2) = 28.3 -> 0
        r   = (70 - 50) / (120 - 70) = 20 / 50 = 0.4

    INPUT       : kps5 (5 x 2), lm106 (106 x 2), coordinate della stessa foto
    RESTITUISCE : (roll, yaw, r), tre float
    CHIAMATA DA : quality_measures, tests/test_quality.py
    """
    kps5, lm106 = np.asarray(kps5, float), np.asarray(lm106, float)
    eye_l, eye_r, nose = kps5[0], kps5[1], kps5[2]
    chin = lm106[CHIN]
    angle = np.degrees(np.arctan2(eye_r[1] - eye_l[1], eye_r[0] - eye_l[0]))
    roll = min(abs(angle) / 45.0, 1.0)
    dl, dr = np.linalg.norm(eye_l - nose), np.linalg.norm(eye_r - nose)
    yaw = abs(dl - dr) / (dl + dr)
    r = np.linalg.norm((eye_l + eye_r) / 2 - nose) / max(np.linalg.norm(nose - chin), 1e-6)
    return float(roll), float(yaw), float(r)


def pitch_norm(r, r0: float):
    """Pitch normalizzato: |r - r0| / r0, al massimo 1.

    ESEMPIO : r0 = 0.40;  r = 0.40 -> 0 (frontale);  r = 0.50 -> 0.25;  r = 1.0 -> 1
    INPUT   : r = numero o array (da pose_measures); r0 = valore frontale (dal tuning)
    CHIAMATA DA : scripts/eval_srr.py, tests/test_quality.py
    """
    return np.minimum(np.abs(np.asarray(r, float) - r0) / r0, 1.0)


def score_pose(roll, yaw, pitch, weights=WEIGHTS):
    """SP = alfa*(1 - roll) + beta*(1 - yaw) + gamma*(1 - pitch), in [0, 1].

    ESEMPIO (pesi 1/3): roll 0, yaw 0.1, pitch 0.25
        SP = (1 - 0)/3 + (1 - 0.1)/3 + (1 - 0.25)/3 = (1 + 0.9 + 0.75) / 3 = 0.883
    INPUT   : roll, yaw, pitch normalizzati (numeri o array); weights = (alfa, beta, gamma)
    CHIAMATA DA : scripts/eval_srr.py, tests/test_quality.py
    """
    a, b, c = weights
    return a * (1 - np.asarray(roll)) + b * (1 - np.asarray(yaw)) + c * (1 - np.asarray(pitch))


def score_illumination(crop) -> float:
    """SI = 1 - std(m_c) / 127.5 sul ritaglio allineato.

    COME LO CALCOLA IL CODICE
        porta il ritaglio in grigi, calcola il grigio medio m_c di ognuna delle
        5 regioni (SI_REGIONS), poi la deviazione standard di quei 5 valori.
        Luce uniforme -> regioni con grigio simile -> std piccola -> SI vicino a 1.
        Ombra su meta' viso -> una guancia scura e una chiara -> SI piu' basso.

    ESEMPIO
        m_c = [150, 150, 150, 150, 150]  ->  std 0     ->  SI = 1
        m_c = [150, 60, 180, 150, 150]   ->  std 40.7  ->  SI = 1 - 40.7/127.5 = 0.68

    INPUT       : crop = ritaglio allineato 112 x 112, BGR o grigi
    RESTITUISCE : float in [0, 1]
    CHIAMATA DA : quality_measures, tests/test_quality.py
    """
    img = np.asarray(crop)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    means = [gray[y1:y2, x1:x2].mean() for (x1, y1, x2, y2) in SI_REGIONS.values()]
    return float(1.0 - np.std(means) / 127.5)


def align_points(points, kps5, size: int = 112):
    """Porta dei punti dalla foto originale al ritaglio allineato.

    COME LO CALCOLA IL CODICE
        estimate_norm di insightface calcola, dai 5 punti, la stessa
        trasformazione (rotazione + scala + spostamento) che norm_crop usa per
        fare il ritaglio; la si applica ai punti: [x, y, 1] @ M^T.

    INPUT       : points (n x 2) e kps5 (5 x 2), coordinate della foto originale
    RESTITUISCE : array (n x 2) nelle coordinate del ritaglio 112 x 112
    CHIAMA      : insightface.utils.face_align.estimate_norm
    CHIAMATA DA : quality_measures
    """
    from insightface.utils.face_align import estimate_norm

    M = estimate_norm(np.asarray(kps5, float), size)
    pts = np.asarray(points, float)
    return np.c_[pts, np.ones(len(pts))] @ M.T


def score_symmetry(lm106_aligned) -> float:
    """SY = media di sym(P_i, P_j) sulle 48 coppie simmetriche, nel ritaglio allineato.

    COME LO CALCOLA IL CODICE
        nel ritaglio allineato l'asse del viso e' la retta verticale x = 55.9
        (a meta' tra le posizioni di riferimento degli occhi). Per ogni coppia
        si riflette P_j rispetto all'asse (x -> 2*55.9 - x) e si misura quanto
        cade lontano da P_i, in proporzione alla distanza tra gli occhi (35.2):
            sym = 1 - distanza / 35.2,   minimo 0
        Volto girato (yaw) o espressione asimmetrica -> coppie storte -> SY piu' basso.

    ESEMPIO (una coppia)
        P_i = (40, 60), P_j = (72, 62)
        riflesso di P_j = (2*55.9 - 72, 62) = (39.8, 62)
        distanza = radice(0.2^2 + 2^2) = 2.01  ->  sym = 1 - 2.01/35.2 = 0.94

    INPUT       : lm106_aligned = 106 punti gia' nel ritaglio allineato (align_points)
    RESTITUISCE : float in [0, 1]
    CHIAMATA DA : quality_measures, tests/test_quality.py
    """
    p = np.asarray(lm106_aligned, float)
    i, j = np.array(SYMMETRIC_PAIRS).T
    mirrored = p[j].copy()
    mirrored[:, 0] = 2 * AXIS_X - mirrored[:, 0]
    sym = 1 - np.linalg.norm(p[i] - mirrored, axis=1) / EYE_DISTANCE
    return float(np.mean(np.clip(sym, 0, 1)))


def quality_measures(crop, kps5, lm106) -> np.ndarray:
    """Le misure di qualita' di una foto: [roll, yaw, r, SI, SY].

    SP non si calcola qui perche' il pitch normalizzato richiede r0, che si
    stima sulle persone di tuning: si salva r e si fa il resto in eval_srr.py.

    INPUT       : (crop, kps5, lm106) come restituiti da FaceEmbedder.quality_inputs
    RESTITUISCE : array float32 di 5 numeri
    CHIAMA      : pose_measures, score_illumination, align_points, score_symmetry
    CHIAMATA DA : eval/matrices.embed_rows (method="quality"), tests/test_quality.py
    """
    roll, yaw, r = pose_measures(kps5, lm106)
    si = score_illumination(crop)
    sy = score_symmetry(align_points(lm106, kps5))
    return np.array([roll, yaw, r, si, sy], dtype=np.float32)
