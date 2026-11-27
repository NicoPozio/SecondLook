"""Incertezza appresa: la lunghezza del vettore di MagFace (Fase 5, passo 5.10).

TEORIA (Meng e colleghi, "MagFace", CVPR 2021)
    MagFace e' una rete di riconoscimento come ArcFace: dal volto allineato 112 x 112
    produce un vettore di 512 numeri. Ogni vettore ha una DIREZIONE e una LUNGHEZZA
    (in inglese magnitude, da cui il nome):
        lunghezza = radice della somma dei quadrati dei 512 numeri
    Il riconoscimento (distanza coseno) guarda solo la direzione: per lui la lunghezza
    e' un'informazione buttata via.
      - In ArcFace la lunghezza non significa niente: l'addestramento non se ne occupa.
      - In MagFace la lunghezza e' ADDESTRATA a misurare la qualita' del volto. Durante
        l'addestramento il margine richiesto a ogni foto cresce con la lunghezza, e un
        termine in piu' spinge le lunghezze verso l'alto: una foto facile (frontale,
        nitida) riesce a rispettare anche il margine piu' severo e "si puo' permettere"
        una lunghezza grande; una foto difficile (sfocata, di profilo) no, e resta corta.
        Le lunghezze stanno tra 10 e 110 (limiti fissati in addestramento).
    Per questo la pipeline la chiama INCERTEZZA APPRESA: e' la rete stessa a dire
    quanto si fida della foto, e l'ha imparato dai dati (non da formule come SP, SI, SY).
    Come SP, SI e SY guarda solo la foto, non la gallery.

ESEMPIO A 3 NUMERI (invece di 512)
    f = (3, 4, 0)     lunghezza = radice(9 + 16 + 0)    = 5
    g = (30, 40, 0)   lunghezza = radice(900 + 1600 + 0) = 50
    stessa direzione (per il coseno sono identici), qualita' diversa per MagFace.

SCELTE (dal codice ufficiale, inference/gen_feat.py e README)
    - modello iResNet100 addestrato su MS1MV2, convertito in ONNX una volta da
      tools/magface_to_onnx.py (cosi' il progetto non dipende da PyTorch)
    - volto allineato a 112 x 112 con i 5 punti, con la funzione di insightface:
      e' lo stesso ritaglio di FaceEmbedder.crop (stesso volto di ArcFace, LBP e qualita')
    - colori BGR (come cv2.imread) e pixel divisi per 255, senza togliere la media
    Su LFW: foto nitide circa 26-30, le stesse sfocate o rimpicciolite circa 20-25.

COSA C'E' IN QUESTO FILE
    MAGFACE_FILE    nome del modello ONNX in data/models
    MagFace         carica il modello; magnitude(crop) -> lunghezza del vettore

MAPPA DELLE CHIAMATE
    Chi usa questo file da fuori: eval/matrices.embed_rows (method="magface"),
    tests/test_magface.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

MAGFACE_FILE = "magface_iresnet100.onnx"


class MagFace:
    """La rete MagFace (ONNX) usata solo per la lunghezza del vettore.

    ESEMPIO
        mf = MagFace()                              # carica data/models/magface_iresnet100.onnx
        crop = FaceEmbedder().crop(img, select="center")
        mf.magnitude(crop)                          # es. 28.3 (foto nitida)
    """

    def __init__(self, models_dir: str | Path = "data/models"):
        """Carica il modello ONNX.

        INPUT   : models_dir = cartella con magface_iresnet100.onnx
        SOLLEVA : FileNotFoundError, con le istruzioni, se il modello non c'e'
        CHIAMA  : onnxruntime.InferenceSession
        """
        import onnxruntime as ort

        path = Path(models_dir) / MAGFACE_FILE
        if not path.exists():
            raise FileNotFoundError(
                f"manca {path}: scarica i pesi iResNet100 di MagFace (README di "
                "github.com/IrvingMeng/MagFace) e convertili con tools/magface_to_onnx.py")
        self._session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self._input = self._session.get_inputs()[0].name

    @staticmethod
    def prepare(crop: np.ndarray) -> np.ndarray:
        """Dal ritaglio BGR 112 x 112 all'input della rete: 1 x 3 x 112 x 112, valori in [0, 1].

        COME LO CALCOLA IL CODICE
            immagine (alto, largo, colore) -> (colore, alto, largo), come fa torchvision
            ToTensor nel codice ufficiale; ogni pixel diviso per 255 (0..255 -> 0..1).
            I colori restano BGR, come li legge cv2.imread nel codice ufficiale.

        ESEMPIO : pixel BGR (255, 128, 0) -> (1.0, 0.502, 0.0)
        """
        if crop.shape != (112, 112, 3):
            raise ValueError(f"serve un ritaglio 112 x 112 x 3, non {crop.shape}")
        return (crop.astype(np.float32).transpose(2, 0, 1) / 255.0)[None]

    def magnitude(self, crop: np.ndarray) -> float:
        """Lunghezza del vettore MagFace del ritaglio: piu' alta = foto piu' affidabile.

        COME LO CALCOLA IL CODICE
            vettore = rete(prepare(crop))       512 numeri, NON normalizzati
            lunghezza = radice(somma dei quadrati) = np.linalg.norm(vettore)

        INPUT       : crop = ritaglio BGR allineato 112 x 112 (FaceEmbedder.crop)
        RESTITUISCE : float (circa 20-30 su LFW; limiti di addestramento 10-110)
        CHIAMA      : prepare, onnxruntime
        CHIAMATA DA : eval/matrices.embed_rows (method="magface"), tests/test_magface.py
        """
        feature = self._session.run(None, {self._input: self.prepare(crop)})[0]
        return float(np.linalg.norm(feature))
