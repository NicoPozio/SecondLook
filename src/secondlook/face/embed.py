"""Rilevamento del volto ed embedding ArcFace con insightface.

TEORIA
    Un sistema biometrico del volto lavora in 3 passi (appunti, cap. "Face"):
      1. DETECTION: trovare DOVE sono i volti nella foto (riquadri = bbox)
         e 5 punti di riferimento per volto (occhi, naso, angoli della bocca).
      2. ALIGNMENT: usare quei 5 punti per ruotare/scalare il volto in modo
         che gli occhi finiscano sempre nella stessa posizione, e ritagliarlo
         a 112 x 112 pixel. Cosi' la rete vede volti "messi in posa" uguale.
      3. FEATURE EXTRACTION: una rete neurale trasforma il ritaglio in un
         vettore di 512 numeri (embedding = template). Volti della stessa
         persona danno vettori che puntano quasi nella stessa direzione.

    Modelli usati (pacchetto buffalo_l di insightface, gia' addestrati):
      - det_10g.onnx   = SCRFD, il detector (passi 1 e 2 usano il suo output)
      - w600k_r50.onnx = ResNet-50 addestrata con la loss ArcFace sul dataset
                         WebFace600K (NON su LFW: LFW lo usiamo solo per testare)

COSA C'E' IN QUESTO FILE
    FaceSample        contenitore: embedding + riquadro + confidenza di UN volto
    cosine_distance   distanza tra DUE embedding (0 = stessa direzione)
    load_image        legge un file immagine dal disco
    FaceEmbedder      carica i modelli; .faces() = tutti i volti, .embed() = un volto,
                      .crop() = ritaglio allineato 112 x 112 di un volto (per LBP)
    select_face       sceglie un volto tra quelli trovati ("largest" o "center")
    _area, _center_distance   funzioni di supporto per select_face

MAPPA DELLE CHIAMATE
    load_image ──> FaceEmbedder.embed ─┐
                                       ├──> _detect_and_choose ──> select_face ──> _area / _center_distance
                   FaceEmbedder.crop ──┘
    embed: poi alignment + ArcFace -> FaceSample (risultato)
    crop:  poi alignment (norm_crop) -> ritaglio 112 x 112 (va a face/lbp.py)
    cosine_distance: usata DOPO, per confrontare due FaceSample.embedding

    Chi usa questo file da fuori:
      eval/matrices.py  embed_rows  -> load_image, FaceEmbedder.embed / FaceEmbedder.crop
      scripts/compare_faces.py      -> load_image, FaceEmbedder.embed, cosine_distance
      tests/test_embed.py, tests/test_face_selection.py
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class FaceSample:
    """UN volto trovato in UNA immagine.

    TEORIA
        Nel linguaggio del corso e' il "template" di un campione: il vettore
        che si salva in gallery (enrollment) o che si confronta (probe).
        Insieme al vettore teniamo dove stava il volto e quanto il detector
        era sicuro, utili per scegliere il volto e per il controllo qualita'.

    CAMPI (esempio reale: volto centrale della foto di prova t1.jpg)
        embedding : np.ndarray di 512 float32, gia' normalizzato (lunghezza 1).
                    Es. inizia con [0.0585, -0.0614, 0.0207, ...].
                    Non dice "chi" e': e' solo una posizione nello spazio a
                    512 dimensioni. Il nome della persona lo sa chi chiama
                    (es. dal nome del file / dal manifest), non questo oggetto.
        bbox      : np.ndarray di 4 int [x1, y1, x2, y2] in pixel dell'immagine
                    originale: (x1, y1) angolo in alto a sinistra,
                    (x2, y2) angolo in basso a destra. Es. [466, 268, 573, 415]
                    -> largo 573-466 = 107 px, alto 415-268 = 147 px.
        det_score : float in [0, 1], confidenza del detector che sia un volto.
                    Es. 0.9197.

    CREATO DA : FaceEmbedder.faces, FaceEmbedder.embed, i test (volti finti)
    USATO DA  : select_face (legge bbox), cosine_distance (legge embedding),
                eval/matrices.embed_rows (salva embedding nella cache)
    """

    embedding: np.ndarray  # 512 float normalizzati: ||embedding|| = 1
    bbox: np.ndarray       # [x1, y1, x2, y2] interi: alto-sinistra e basso-destra
    det_score: float       # confidenza del detector, 0..1


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Distanza coseno tra due embedding: 1 - cos(angolo tra i vettori).

    TEORIA
        Il coseno dell'angolo tra due vettori misura se puntano nella stessa
        direzione, ignorando quanto sono lunghi:
            cos = (a . b) / (||a|| * ||b||)
        Similarita' -> distanza:  d = 1 - cos.
            stessa direzione  -> cos =  1 -> d = 0   (stessa persona, idealmente)
            perpendicolari    -> cos =  0 -> d = 1   (persone diverse, tipico)
            opposti           -> cos = -1 -> d = 2   (massimo teorico)
        Nel corso le formule di FAR/FRR (p.14) usano una similarita' s con
        "accetto se s >= t"; noi usiamo la distanza con "accetto se d <= t".

    COME LO CALCOLA IL CODICE
        1. divide ogni vettore per la sua lunghezza (norma L2) -> lunghezza 1
        2. con vettori lunghi 1 il coseno e' solo il prodotto scalare np.dot
        3. restituisce 1 - prodotto scalare

    ESEMPIO (vettori a 3 dimensioni per fare i conti a mano)
        a = [3, 4, 0]  ||a|| = 5  -> [0.6, 0.8, 0]
        b = [4, 3, 0]  ||b|| = 5  -> [0.8, 0.6, 0]
        dot = 0.6*0.8 + 0.8*0.6 + 0*0 = 0.96  ->  distanza = 0.04 (molto simili)
        Con ArcFace reale su t1.jpg: stesso volto leggermente disturbato ~0.036,
        il volto piu' simile tra le ALTRE persone della foto ~0.94.

    INPUT
        a, b : due embedding (np.ndarray della stessa lunghezza, per noi 512),
               di solito FaceSample.embedding di due foto diverse. La funzione
               NON sa a chi appartengono: e' chi chiama che sa se e' un
               confronto genuino (stessa persona) o impostore (persone diverse).
               Non serve che siano gia' normalizzati: lo fa la funzione.

    RESTITUISCE : float in [0, 2]

    CHIAMA       : np.linalg.norm, np.dot
    CHIAMATA DA  : scripts/compare_faces.py, tests/test_embed.py,
                   tests/test_face_selection.py.
                   (Per tanti confronti insieme si usa invece la versione a
                   matrice eval/matrices.cosine_distance_matrix, stessa formula.)
    """
    a = a / np.linalg.norm(a)        # es. [3,4,0] / 5 = [0.6, 0.8, 0]
    b = b / np.linalg.norm(b)        # es. [4,3,0] / 5 = [0.8, 0.6, 0]
    return float(1.0 - np.dot(a, b))  # 1 - 0.96 = 0.04


def load_image(path: str | Path) -> np.ndarray:
    """Legge un'immagine dal disco come matrice di pixel BGR.

    TEORIA
        Un'immagine a colori e' una griglia altezza x larghezza di pixel,
        ognuno con 3 numeri 0..255. OpenCV li mette in ordine Blu, Verde,
        Rosso (BGR, non RGB), ed e' proprio l'ordine che si aspetta insightface.

    COME LO CALCOLA IL CODICE
        cv2.imread legge il file; se non ci riesce (percorso sbagliato,
        file rotto) NON da' errore ma restituisce None: per questo
        controlliamo noi e lanciamo FileNotFoundError.

    ESEMPIO
        load_image("t1.jpg").shape  -> (886, 1280, 3)
              886 righe (altezza), 1280 colonne (larghezza), 3 canali
        pixel in alto a sinistra img[0, 0] -> [126, 172, 206]  (B, G, R)

    INPUT
        path : percorso del file (stringa o Path), es. "data/lfw/lfw/Kate_Winslet/
               Kate_Winslet_0002.jpg". Per LFW arriva dalla colonna "path" del
               manifest (eval/datasets.lfw_manifest).

    RESTITUISCE : np.ndarray uint8 di forma (altezza, larghezza, 3), BGR
    SOLLEVA     : FileNotFoundError se il file non si legge

    CHIAMA       : cv2.imread
    CHIAMATA DA  : eval/matrices.embed_rows, scripts/compare_faces.py
    """
    img = cv2.imread(str(path))      # str() perche' cv2 non accetta oggetti Path
    if img is None:                  # cv2 segnala l'errore cosi', non con un'eccezione
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


class FaceEmbedder:
    """Carica i modelli e trasforma un'immagine in embedding di volto.

    TEORIA
        Raggruppa i 3 passi detection -> alignment -> ArcFace (vedi inizio
        file). I modelli si caricano UNA volta nel costruttore (operazione
        lenta, qualche secondo) e poi si riusano per tutte le immagini.
        Il pacchetto buffalo_l viene scaricato in ~/.insightface al primo uso.

    METODI
        faces(img)  -> TUTTI i volti, ognuno con il suo embedding
        embed(img)  -> UN solo volto scelto, embedding solo di quello (piu' veloce)
        crop(img)   -> UN solo volto scelto, ritaglio allineato 112 x 112 (per LBP)
        quality_inputs(img) -> ritaglio + 5 punti + 106 punti (per gli indici di qualita')
        embed_crop(crop) -> ArcFace su un ritaglio 112 x 112 gia' pronto (confronto tra detector)

    CHIAMA       : insightface.app.FaceAnalysis (costruttore)
    CREATO DA    : scripts/eval_lfw.py, scripts/compare_faces.py, i test
    """

    def __init__(self, model_pack: str = "buffalo_l", det_size: int = 640, landmarks: bool = False):
        """Carica detector (SCRFD) e riconoscitore (ArcFace), e se richiesto il modello a 106 punti.

        PARAMETRI
            model_pack : nome del pacchetto di modelli insightface ("buffalo_l").
            det_size   : lato in pixel del quadrato in cui il detector vede la
                         foto. Es. t1.jpg 1280 x 886 con det_size=640:
                         scala 640/1280 = 0.5 -> 640 x 443, il resto riempito
                         di nero fino a 640 x 640. I riquadri trovati vengono
                         poi riportati alle coordinate della foto originale.
                         Piu' grande = trova volti piu' piccoli ma e' piu' lento.
            landmarks  : True = carica anche 2d106det.onnx (106 punti del volto,
                         stesso pacchetto buffalo_l), che serve solo agli indici
                         di qualita' (quality_inputs). Default False: non cambia nulla
                         per chi usa embed o crop.

        CHIAMA : FaceAnalysis(...), FaceAnalysis.prepare(...)
        """
        from insightface.app import FaceAnalysis  # import qui: e' pesante, solo se serve

        self._app = FaceAnalysis(
            name=model_pack,
            # SCRFD e ArcFace (niente eta'/genere); i 106 punti solo se richiesti
            allowed_modules=["detection", "recognition"] + (["landmark_2d_106"] if landmarks else []),
            providers=["CPUExecutionProvider"],            # onnxruntime su CPU, niente GPU
        )
        # ctx_id=-1 = CPU; det_size = dimensione dell'input del detector (vedi sopra)
        self._app.prepare(ctx_id=-1, det_size=(det_size, det_size))

    def faces(self, img: np.ndarray) -> list[FaceSample]:
        """Tutti i volti dell'immagine, dal piu' sicuro al meno sicuro.

        COME LO CALCOLA IL CODICE
            self._app.get(img) fa detection + alignment + ArcFace su OGNI volto
            trovato. Poi convertiamo ogni risultato in FaceSample e ordiniamo
            per det_score decrescente.

        ESEMPIO (t1.jpg, foto di gruppo con 6 persone)
            restituisce 6 FaceSample:
              det 0.9197  bbox [466, 268, 573, 415]   <- il piu' sicuro, primo
              det 0.9118  bbox [745, 338, 845, 479]
              ...
              det 0.8714  bbox [903,  62,1013, 205]   <- ultimo
            ognuno con il suo embedding di 512 numeri.
            Foto senza volti -> lista vuota [] (nessun errore).

        INPUT
            img : immagine BGR (altezza, larghezza, 3) uint8, es. da load_image.

        RESTITUISCE : list[FaceSample], ordinata per det_score decrescente

        CHIAMA       : FaceAnalysis.get, FaceSample
        CHIAMATA DA  : tests/test_embed.py, tests/test_face_selection.py
                       (test_embed_matches_faces: serve come "risposta giusta"
                       con cui confrontare embed)
        """
        found = self._app.get(img)           # lista di oggetti Face di insightface
        samples = [
            FaceSample(
                embedding=f.normed_embedding.astype(np.float32),  # gia' diviso per la norma
                bbox=f.bbox.astype(int),                          # da float a pixel interi
                det_score=float(f.det_score),
            )
            for f in found
        ]
        return sorted(samples, key=lambda s: s.det_score, reverse=True)

    def embed(self, img: np.ndarray, select: str = "largest") -> FaceSample:
        """Embedding di UN volto dell'immagine, scelto con `select`.

        TEORIA
            In una foto possono esserci piu' volti, ma un campione biometrico
            deve riguardare UNA persona. Quale volto prendere dipende da come
            sono fatte le foto:
              "largest" : il riquadro piu' grande. Nel login l'utente e' la
                          persona piu' vicina alla webcam (default).
              "center"  : il riquadro piu' vicino al centro. In LFW la persona
                          etichettata e' sempre centrata, ma ai bordi possono
                          esserci altre persone anche piu' grandi.
                          Es. Kate_Winslet_0002: volto centrale 7070 px² (a 4 px
                          dal centro) e un altro volto 8624 px² (a 119 px).
                          Con "largest" la coppia genuina dava distanza 0.979
                          (sembrava un impostore!), con "center" 0.253.

        COME LO CALCOLA IL CODICE
            1. solo DETECTION (veloce) su tutta la foto: riquadri + 5 punti
               per ogni volto, ancora nessun embedding.
            2. sceglie un volto con select_face, guardando solo i riquadri.
               (1 e 2 stanno in _detect_and_choose, condivisa con crop per LBP)
            3. ALIGNMENT + ArcFace SOLO sul volto scelto. Cosi' una foto con 6
               volti costa 1 embedding invece di 6. Il risultato e' identico a
               quello di faces() per lo stesso volto (lo verifica il test
               test_embed_matches_faces).

        ESEMPIO (t1.jpg, 1280 x 886, centro dell'immagine (640, 443))
            passo 1: bboxes forma (6, 5), kpss forma (6, 5, 2)
                bboxes[0] = [466.08, 268.62, 573.59, 415.53, 0.92]
                             x1      y1      x2      y2      det_score
                kpss[0]   = [[491.9, 321.8],   occhio sinistro
                             [541.9, 332.1],   occhio destro
                             [507.7, 366.4],   naso
                             [485.9, 369.7],   bocca sinistra
                             [533.7, 378.4]]   bocca destra
            passo 2:
                "largest" -> bbox [903, 62, 1013, 205], area 15730
                             (batte [466,268,573,415], area 15729, per 1 px²!)
                "center"  -> bbox [466, 268, 573, 415], a 157.6 px dal centro
            passo 3: ArcFace da' un vettore grezzo lungo ~25.91; diviso per la
                lunghezza -> normed_embedding = [0.0585, -0.0614, 0.0207, ...]

        INPUT
            img    : UNA immagine BGR (altezza, larghezza, 3) uint8, es. da
                     load_image. Non serve sapere di chi e': la funzione
                     restituisce solo il vettore; associarlo a una persona
                     (es. "Kate_Winslet", sample_id "Kate_Winslet_0002") e'
                     compito di chi chiama (embed_rows lo salva nella cache
                     sotto il sample_id preso dal manifest).
            select : "largest" o "center" (vedi TEORIA). Altro -> ValueError.

        RESTITUISCE : FaceSample del volto scelto (embedding di 512 float32,
                      bbox in pixel interi, det_score)
        SOLLEVA     : ValueError("No face detected") se non c'e' nessun volto
                      (embed_rows lo intercetta e mette il campione in cache.failed);
                      ValueError da select_face se `select` non e' valido.

        CHIAMA       : _detect_and_choose (-> SCRFD, select_face),
                       insightface Face, models["recognition"].get (alignment + ArcFace)
        CHIAMATA DA  : eval/matrices.embed_rows, scripts/compare_faces.py,
                       tests/test_embed.py, tests/test_face_selection.py
        """
        # 1. e 2. detection + scelta del volto: la STESSA funzione usata da crop() per LBP
        bboxes, kpss, k, chosen = self._detect_and_choose(img, select)
        # 3. alignment + ArcFace solo sul volto scelto
        from insightface.app.common import Face

        # Face = il contenitore che il modello di riconoscimento di insightface si aspetta
        face = Face(bbox=bboxes[k, :4], kps=kpss[k], det_score=bboxes[k, 4])
        # .get usa i 5 punti per allineare e ritagliare a 112x112, poi ArcFace;
        # scrive dentro face sia .embedding (grezzo, lungo ~25.9) sia .normed_embedding (lungo 1)
        self._app.models["recognition"].get(img, face)
        return FaceSample(
            embedding=face.normed_embedding.astype(np.float32),  # es. [0.0585, -0.0614, 0.0207, ...]
            bbox=chosen.bbox,
            det_score=chosen.det_score,
        )

    def _detect_and_choose(self, img: np.ndarray, select: str):
        """Passi 1 e 2 comuni a embed() (ArcFace) e crop() (LBP): rileva i volti e ne sceglie uno.

        TEORIA
            Per confrontare ArcFace e LBP in modo onesto, il volto deve essere
            trovato, scelto e ritagliato ESATTAMENTE nello stesso modo per i
            due metodi: cosi' la differenza nei risultati dipende solo dal
            descrittore. Per garantirlo, questi passi esistono in un solo
            posto, questa funzione, e la usano sia embed() sia crop().

        COME LO CALCOLA IL CODICE
            1. SCRFD trova tutti i volti: riquadri + 5 punti per volto
            2. select_face sceglie un volto guardando solo i riquadri
            3. k = posizione del volto scelto, per ritrovare i suoi 5 punti

        ESEMPIO (t1.jpg, select="center")
            bboxes forma (6, 5), kpss forma (6, 5, 2), k = 0,
            chosen.bbox = [466, 268, 573, 415]
            (i valori completi sono nell'ESEMPIO di embed)

        INPUT       : img = immagine BGR; select = "largest" o "center"
        RESTITUISCE : (bboxes, kpss, k, chosen)
                      bboxes = tutti i riquadri [x1, y1, x2, y2, det_score]
                      kpss   = i 5 punti di ogni volto
                      k      = indice del volto scelto
                      chosen = FaceSample del volto scelto (embedding vuoto)
        SOLLEVA     : ValueError("No face detected") se non c'e' nessun volto
        CHIAMA      : det_model.detect (SCRFD), FaceSample, select_face
        CHIAMATA DA : embed, crop
        """
        # max_num=0 = nessun limite al numero di volti.
        #    bboxes: una riga per volto [x1, y1, x2, y2, det_score]   es. forma (6, 5)
        #    kpss:   5 punti (x, y) per volto                         es. forma (6, 5, 2)
        bboxes, kpss = self._app.det_model.detect(img, max_num=0)
        if len(bboxes) == 0:
            raise ValueError("No face detected")
        # FaceSample "provvisori": embedding vuoto (np.empty(0)), servono solo
        # riquadro e confidenza per poter usare select_face
        candidates = [
            FaceSample(embedding=np.empty(0, np.float32), bbox=b[:4].astype(int), det_score=float(b[4]))
            for b in bboxes      # b = [466.08, 268.62, 573.59, 415.53, 0.92] -> bbox [466, 268, 573, 415]
        ]
        # scelta del volto, prima di calcolare qualsiasi embedding
        chosen = select_face(candidates, img.shape, select)
        # k = posizione del volto scelto nella lista (es. 0 con "center"), per
        # ritrovare le sue coordinate float e i suoi 5 punti in bboxes/kpss.
        # "is" confronta l'oggetto stesso, non il contenuto.
        k = next(i for i, c in enumerate(candidates) if c is chosen)
        return bboxes, kpss, k, chosen

    def crop(self, img: np.ndarray, select: str = "largest") -> np.ndarray:
        """Il ritaglio allineato 112 x 112 del volto scelto: l'ingresso di LBP.

        TEORIA
            E' ESATTAMENTE l'immagine che ArcFace riceve dentro embed():
            stessa detection, stessa scelta del volto (_detect_and_choose) e
            stesso allineamento (norm_crop di insightface, con gli stessi
            5 punti e la stessa dimensione). Cosi' LBP e ArcFace vedono lo
            stesso volto, pixel per pixel, e il confronto tra i due riguarda
            solo il descrittore. Lo verifica il test
            test_crop_is_what_arcface_sees: passando questo ritaglio ad
            ArcFace si ottiene lo stesso embedding di embed().

        COME LO CALCOLA IL CODICE
            1. _detect_and_choose -> 5 punti del volto scelto
            2. norm_crop: trasformazione (rotazione + scala + spostamento) che
               porta i 5 punti nelle posizioni standard, e ritaglio del
               quadrato di lato input_size (112 per il modello di buffalo_l)

        ESEMPIO
            t1.jpg (1280 x 886 x 3), select="center" -> immagine 112 x 112 x 3,
            con gli occhi del volto centrale nelle posizioni standard

        INPUT       : come embed (img BGR, select "largest" o "center")
        RESTITUISCE : immagine BGR uint8 di forma (112, 112, 3)
        SOLLEVA     : ValueError("No face detected") se non c'e' nessun volto
        CHIAMA      : _detect_and_choose, insightface face_align.norm_crop
        CHIAMATA DA : eval/matrices.embed_rows (method="lbp"),
                      tests/test_face_selection.py
        """
        from insightface.utils.face_align import norm_crop

        _, kpss, k, _ = self._detect_and_choose(img, select)
        size = self._app.models["recognition"].input_size[0]   # 112, la stessa che usa ArcFace
        return norm_crop(img, landmark=kpss[k], image_size=size)

    def embed_crop(self, crop: np.ndarray) -> np.ndarray:
        """Embedding ArcFace di un ritaglio 112 x 112 GIA' pronto (nessun rilevamento, nessun allineamento).

        TEORIA
            Serve al confronto tra detector (face/detect.py): li' il ritaglio viene
            da un riquadro (SCRFD o Viola-Jones) senza allineamento, e ArcFace va
            applicato cosi' com'e'. Se il ritaglio e' quello di crop(), il risultato
            e' identico a embed() (lo controlla test_crop_is_what_arcface_sees).

        INPUT       : crop = immagine BGR 112 x 112
        RESTITUISCE : 512 float32, normalizzati (lunghezza 1)
        CHIAMA      : models["recognition"].get_feat
        CHIAMATA DA : face/detect.detector_vector, tests/test_detect.py
        """
        raw = self._app.models["recognition"].get_feat(crop).flatten()
        return (raw / np.linalg.norm(raw)).astype(np.float32)

    def quality_inputs(self, img: np.ndarray, select: str = "largest"):
        """Tutto quello che serve agli indici di qualita' (face/quality.py) per il volto scelto.

        TEORIA
            Gli indici di qualita' (SP, SI, SY) si calcolano PRIMA del confronto,
            sulla foto. Lo stesso volto deve essere quello usato da ArcFace e LBP,
            quindi rilevamento e scelta del volto sono di nuovo _detect_and_choose.

        COME LO CALCOLA IL CODICE
            1. _detect_and_choose -> 5 punti del volto scelto
            2. il modello a 106 punti (2d106det) sul riquadro del volto scelto
            3. il ritaglio allineato 112 x 112, identico a crop()

        ESEMPIO : t1.jpg, select="center" -> crop (112, 112, 3), kps5 (5, 2), lm106 (106, 2)

        INPUT       : come embed; l'oggetto va creato con landmarks=True
        RESTITUISCE : (crop, kps5, lm106) - ritaglio BGR allineato, 5 punti e 106 punti
                      nelle coordinate della foto ORIGINALE
        SOLLEVA     : ValueError se manca il modello a 106 punti o non c'e' nessun volto
        CHIAMA      : _detect_and_choose, insightface Face, models["landmark_2d_106"].get, norm_crop
        CHIAMATA DA : eval/matrices.embed_rows (method="quality"), tests/test_quality.py
        """
        from insightface.app.common import Face
        from insightface.utils.face_align import norm_crop

        if "landmark_2d_106" not in self._app.models:
            raise ValueError("crea FaceEmbedder(landmarks=True) per usare quality_inputs")
        bboxes, kpss, k, _ = self._detect_and_choose(img, select)
        face = Face(bbox=bboxes[k, :4], kps=kpss[k], det_score=bboxes[k, 4])
        self._app.models["landmark_2d_106"].get(img, face)   # scrive face.landmark_2d_106
        size = self._app.models["recognition"].input_size[0]
        crop = norm_crop(img, landmark=kpss[k], image_size=size)
        return crop, kpss[k].astype(np.float32), face.landmark_2d_106.astype(np.float32)


def select_face(samples: list[FaceSample], image_shape, select: str = "largest") -> FaceSample:
    """Sceglie UN volto tra quelli trovati nella stessa immagine.

    TEORIA : vedi FaceEmbedder.embed (perche' "largest" nel login e "center" su LFW).

    COME LO CALCOLA IL CODICE
        "largest" : max sulla lista usando come criterio l'area del riquadro (_area)
        "center"  : min sulla lista usando come criterio la distanza tra il centro
                    del riquadro e il centro dell'immagine (_center_distance)
        A parita' vince il primo della lista (comportamento di max/min).

    ESEMPIO (come nel test test_face_selection, foto 250 x 250 come LFW)
        CENTRALE = bbox [75, 70, 175, 190]  area 100*120 = 12000, a 5 px dal centro
        BORDO    = bbox [180, 60, 290, 190] area 110*130 = 14300, a 110 px dal centro
        "largest" -> BORDO       "center" -> CENTRALE

    INPUT
        samples     : lista NON vuota di FaceSample trovati nella STESSA foto
                      (serve solo il campo bbox; l'embedding puo' essere vuoto,
                      come in embed). Lista vuota -> errore di max/min.
        image_shape : forma dell'immagine, es. img.shape = (886, 1280, 3);
                      si usano solo i primi due numeri (altezza, larghezza),
                      e solo per "center".
        select      : "largest" o "center"

    RESTITUISCE : lo stesso oggetto FaceSample scelto (non una copia: per questo
                  embed puo' ritrovarlo con "is")
    SOLLEVA     : ValueError se select non e' "largest" o "center"

    CHIAMA       : _area, _center_distance
    CHIAMATA DA  : FaceEmbedder._detect_and_choose (quindi embed e crop), tests/test_face_selection.py
    """
    if select == "largest":
        return max(samples, key=lambda s: _area(s.bbox))
    if select == "center":
        h, w = image_shape[:2]       # es. (886, 1280, 3) -> h = 886, w = 1280
        return min(samples, key=lambda s: _center_distance(s.bbox, w, h))
    raise ValueError(f"unknown face selection: {select!r}")


def _area(bbox: np.ndarray) -> int:
    """Area del riquadro in pixel²: larghezza * altezza.

    ESEMPIO : [466, 268, 573, 415] -> (573-466) * (415-268) = 107 * 147 = 15729
    INPUT   : bbox [x1, y1, x2, y2] (vedi FaceSample)
    CHIAMATA DA : select_face
    """
    return int((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))


def _center_distance(bbox: np.ndarray, width: int, height: int) -> float:
    """Distanza in pixel tra il centro del riquadro e il centro dell'immagine.

    COME LO CALCOLA : centro riquadro = media degli angoli; poi Pitagora
                      (np.hypot(dx, dy) = radice di dx² + dy²).
    ESEMPIO (t1.jpg, 1280 x 886 -> centro immagine (640, 443))
        bbox [466, 268, 573, 415] -> centro (519.5, 341.5)
        dx = 519.5 - 640 = -120.5, dy = 341.5 - 443 = -101.5
        distanza = radice(120.5² + 101.5²) = 157.6
    INPUT   : bbox [x1, y1, x2, y2]; width e height dell'immagine in pixel
              (attenzione all'ordine: img.shape da' prima l'altezza)
    CHIAMA  : np.hypot
    CHIAMATA DA : select_face
    """
    cx = (bbox[0] + bbox[2]) / 2     # es. (466 + 573) / 2 = 519.5
    cy = (bbox[1] + bbox[3]) / 2     # es. (268 + 415) / 2 = 341.5
    return float(np.hypot(cx - width / 2, cy - height / 2))  # es. 157.6
