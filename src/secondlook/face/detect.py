"""Viola-Jones contro SCRFD: quanto conta il detector (Fase 5, passo 5.7).

TEORIA (appunti, rilevamento del volto)
    VIOLA-JONES (2001) e' il detector classico del corso. Scorre l'immagine con
    finestre di tante dimensioni e su ogni finestra applica una CASCATA di
    classificatori costruiti con AdaBoost su FEATURE DI HAAR: differenze di
    luminosita' tra rettangoli chiari e scuri (es. la fascia degli occhi e' piu'
    scura delle guance). Una finestra che supera tutti gli stadi e' un volto.
    In OpenCV e' cv2.CascadeClassifier con il file haarcascade_frontalface_default.xml.

    SCRFD (det_10g.onnx di buffalo_l) e' il detector deep che usiamo: da' il
    riquadro E i 5 punti (occhi, naso, bocca) con cui si ALLINEA il volto.
    Viola-Jones da' SOLO il riquadro: senza punti non si puo' allineare.

    Per capire quanto conta il detector senza confonderlo con l'allineamento si
    confrontano tre varianti, tutte con ArcFace:
        arcface          SCRFD + allineamento (il nostro sistema, gia' valutato)
        arcface_noalign  riquadro SCRFD, NESSUN allineamento (controllo)
        arcface_vj       riquadro Viola-Jones, nessun allineamento
    arcface_noalign contro arcface    -> effetto dell'allineamento
    arcface_vj      contro noalign    -> effetto del detector (a parita' di ritaglio)

ATTENZIONE: OpenCV 5
    In OpenCV 5 Viola-Jones e' stato spostato nel pacchetto "contrib": con il
    solo opencv-python 5 cv2.CascadeClassifier NON esiste. Serve opencv-python
    4.x (oppure opencv-contrib-python). haar_cascade() lo segnala con un errore chiaro.

COSA C'E' IN QUESTO FILE
    haar_cascade      carica Viola-Jones (scarica il file .xml in data/models se manca)
    viola_jones       riquadri dei volti trovati da Viola-Jones in una foto
    iou               sovrapposizione tra due riquadri (Intersection over Union)
    square_crop       ritaglio quadrato 112 x 112 attorno a un riquadro, SENZA allineamento
    detector_vector   per una foto: riquadri SCRFD e VJ ed embedding ArcFace dei due
                      ritagli non allineati (1032 numeri, nan dove manca il volto)

MAPPA DELLE CHIAMATE
    detector_vector -> FaceEmbedder._detect_and_choose, viola_jones, select_face,
                       square_crop, FaceEmbedder.embed_crop
    Chi usa questo file da fuori: eval/matrices.embed_rows (method="detectors"),
    scripts/eval_detectors.py, tests/test_detect.py
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import cv2
import numpy as np

HAAR_FILE = "haarcascade_frontalface_default.xml"
HAAR_URL = "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/" + HAAR_FILE
HAAR_MD5 = "a03f92a797e309e76e6a034ab9e02616"      # lo stesso file distribuito con opencv-python 4.x
CROP_SCALE = 1.2                                    # lato del ritaglio = 1.2 * radice(area del riquadro)


def haar_cascade(models_dir: str | Path = "data/models"):
    """Carica il classificatore di Viola-Jones (frontale), scaricando il file se manca.

    COME LO CALCOLA IL CODICE
        1. se cv2 non ha CascadeClassifier (OpenCV 5 senza contrib): errore chiaro
        2. se data/models/haarcascade_frontalface_default.xml non c'e', lo scarica
           da GitHub e controlla l'impronta md5 (file identico a quello ufficiale)
        3. cv2.CascadeClassifier(file)

    INPUT       : models_dir = cartella dove tenere il file (data/ e' esclusa da git)
    RESTITUISCE : cv2.CascadeClassifier pronto all'uso
    SOLLEVA     : RuntimeError se OpenCV non ha Viola-Jones o il file e' corrotto
    CHIAMATA DA : viola_jones (se non riceve un classificatore), eval/matrices.embed_rows
    """
    if not hasattr(cv2, "CascadeClassifier"):
        raise RuntimeError("questa versione di OpenCV non ha Viola-Jones (CascadeClassifier): "
                           "in OpenCV 5 serve opencv-contrib-python, oppure usare opencv-python<5")
    path = Path(models_dir) / HAAR_FILE
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(HAAR_URL) as r:
            path.write_bytes(r.read())
    if hashlib.md5(path.read_bytes()).hexdigest() != HAAR_MD5:
        raise RuntimeError(f"il file {path} non e' quello atteso: cancellalo e riprova")
    return cv2.CascadeClassifier(str(path))


def viola_jones(img: np.ndarray, cascade=None) -> list[np.ndarray]:
    """Tutti i volti trovati da Viola-Jones, come riquadri [x1, y1, x2, y2].

    COME LO CALCOLA IL CODICE
        porta la foto in grigi (le feature di Haar guardano solo la luminosita')
        e chiama detectMultiScale con i parametri piu' comuni:
            scaleFactor = 1.1   finestre che crescono del 10% a ogni passo
            minNeighbors = 5    un volto deve essere trovato da almeno 5 finestre vicine
                                (meno falsi volti, qualche volto vero perso)
            minSize = 30 x 30   niente volti piu' piccoli di 30 pixel
        OpenCV restituisce (x, y, larghezza, altezza): si converte in (x1, y1, x2, y2).

    ESEMPIO : foto LFW 250 x 250 -> di solito un riquadro come [70, 75, 186, 191]
    INPUT       : img = foto BGR; cascade = classificatore (None = haar_cascade())
    RESTITUISCE : lista di array [x1, y1, x2, y2] (vuota se non trova volti)
    CHIAMA      : haar_cascade, cv2.detectMultiScale
    CHIAMATA DA : detector_vector, tests/test_detect.py
    """
    cascade = cascade if cascade is not None else haar_cascade()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return [np.array([x, y, x + w, y + h], dtype=float) for (x, y, w, h) in found]


def iou(a, b) -> float:
    """Intersection over Union: area comune / area totale dei due riquadri.

    ESEMPIO
        a = [0, 0, 10, 10] (area 100), b = [5, 0, 15, 10] (area 100)
        intersezione = [5, 0, 10, 10] = 50;  unione = 100 + 100 - 50 = 150
        IoU = 50 / 150 = 0.33
        Riquadri identici -> 1; senza parti in comune -> 0.
        Di solito un volto "trovato" vuol dire IoU >= 0.5 con quello di riferimento.

    INPUT       : a, b = riquadri [x1, y1, x2, y2]
    RESTITUISCE : float in [0, 1]
    CHIAMATA DA : scripts/eval_detectors.py, tests/test_detect.py
    """
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def square_crop(img: np.ndarray, box, size: int = 112, scale: float = CROP_SCALE) -> np.ndarray:
    """Ritaglio quadrato attorno a un riquadro, ridimensionato a size x size, SENZA allineamento.

    COME LO CALCOLA IL CODICE
        lato = scale * radice(larghezza * altezza) del riquadro, centrato sul suo centro.
        Si usa la radice dell'area (non il lato piu' lungo) perche' SCRFD e Viola-Jones
        disegnano riquadri di forma diversa (su LFW: VJ piu' largo del 22% e piu'
        basso del 13%), ma di area simile: cosi' il volto ha circa la stessa
        grandezza nel ritaglio con i due detector.
        Se il quadrato esce dalla foto, i bordi si riempiono ripetendo i pixel del bordo.

    ESEMPIO
        riquadro SCRFD [80, 60, 175, 195]: 95 x 135, radice(95*135) = 113
        lato = 1.2 * 113 = 136, centro (127.5, 127.5) -> quadrato [59.5, 59.5, 195.5, 195.5]
        -> ridimensionato a 112 x 112

    INPUT       : img = foto BGR; box = [x1, y1, x2, y2]; size = lato finale; scale = margine
    RESTITUISCE : immagine BGR size x size
    CHIAMA      : cv2.copyMakeBorder, cv2.resize
    CHIAMATA DA : detector_vector, tests/test_detect.py
    """
    x1, y1, x2, y2 = [float(v) for v in box]
    side = scale * np.sqrt(max(x2 - x1, 1) * max(y2 - y1, 1))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    left, top = int(round(cx - side / 2)), int(round(cy - side / 2))
    s = int(round(side))
    pad = s                                                    # abbastanza per qualunque uscita dai bordi
    padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    crop = padded[top + pad: top + pad + s, left + pad: left + pad + s]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)


def detector_vector(img: np.ndarray, embedder, cascade, select: str = "center") -> np.ndarray:
    """Tutto il confronto tra detector per UNA foto, in un unico vettore di 1032 numeri.

    COME LO CALCOLA IL CODICE
        posizioni   0..3     riquadro SCRFD del volto scelto       (nan se SCRFD non trova volti)
        posizioni   4..7     riquadro Viola-Jones del volto scelto (nan se VJ non trova volti)
        posizioni   8..519   ArcFace sul ritaglio NON allineato del riquadro SCRFD (controllo)
        posizioni 520..1031  ArcFace sul ritaglio NON allineato del riquadro VJ
        Il volto si sceglie con la stessa regola per i due detector (select_face,
        "center" per LFW). Un solo passaggio sulla foto per tutte e due le varianti.

    INPUT       : img = foto BGR; embedder = FaceEmbedder; cascade = haar_cascade(); select
    RESTITUISCE : array float32 di 1032 numeri
    CHIAMA      : embedder._detect_and_choose, viola_jones, select_face, square_crop,
                  embedder.embed_crop
    CHIAMATA DA : eval/matrices.embed_rows (method="detectors")
    """
    from secondlook.face.embed import FaceSample, select_face

    out = np.full(1032, np.nan, dtype=np.float32)
    try:
        bboxes, _, k, _ = embedder._detect_and_choose(img, select)
        box_s = bboxes[k, :4]
        out[0:4] = box_s
        out[8:520] = embedder.embed_crop(square_crop(img, box_s))
    except ValueError:                                         # SCRFD non trova volti
        pass
    found = viola_jones(img, cascade)
    if found:
        cands = [FaceSample(np.empty(0, np.float32), b, 1.0) for b in found]
        box_v = select_face(cands, img.shape, select).bbox
        out[4:8] = box_v
        out[520:1032] = embedder.embed_crop(square_crop(img, box_v))
    return out
