"""Embedding dei campioni e matrici delle distanze (Step 4 e 5, Fase 2).

TEORIA
    Per valutare un sistema di identificazione bisogna confrontare OGNI
    probe con OGNI template della gallery. Con 7.484 probe e 1.680 template
    di LFW sono 12,5 milioni di distanze: si mettono in una MATRICE
        righe   = probe (tentativi)
        colonne = template della gallery
        dist[p, g] = distanza coseno tra probe p e template g
    che e' l'input di eval/metrics.cms_curve e open_set_rates.

    La matrice contiene solo numeri: CHI e' ogni riga e ogni colonna lo sa
    chi la costruisce, tenendo a parte la lista dei subject_id nello stesso
    ordine (in eval_lfw.py: k_ids per le righe, g_ids per le colonne).

    Calcolare gli embedding di migliaia di foto richiede tempo (decine di
    minuti per tutto LFW su CPU), quindi si salvano in una CACHE su disco:
    se lo script si interrompe, al riavvio riparte da dove era arrivato.

COSA C'E' IN QUESTO FILE
    cosine_distance_matrix  tutte le distanze probe x gallery in un colpo
    EmbeddingCache          dizionario sample_id -> embedding, salvato in .npz
    embed_rows              calcola gli embedding che mancano nella cache

MAPPA DELLE CHIAMATE
    embed_rows -> EmbeddingCache.missing, face.load_image,
                  FaceEmbedder.embed (ArcFace) oppure FaceEmbedder.crop + lbp_descriptor (LBP),
                  EmbeddingCache.save
    Chi usa questo file da fuori:
      scripts/eval_lfw.py -> EmbeddingCache, embed_rows, cosine_distance_matrix
      tests/test_lfw_tools.py -> cosine_distance_matrix, EmbeddingCache
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np


def cosine_distance_matrix(probes: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """dist[p, g] = 1 - coseno tra il probe p e il template g.

    TEORIA
        E' la stessa formula di face/embed.cosine_distance, ma per tutte le
        coppie insieme. Dopo aver reso ogni vettore lungo 1, il coseno di
        due vettori e' il loro prodotto scalare; e il prodotto tra matrici
        P @ G.T calcola proprio il prodotto scalare di ogni riga di P con
        ogni riga di G:
            (n_probe x 512) @ (512 x n_gallery) = (n_probe x n_gallery)
        Molto piu' veloce di due cicli for annidati.

    COME LO CALCOLA IL CODICE
        1. converte in float32 (es. liste -> array)
        2. divide ogni RIGA per la sua lunghezza (axis=1 = lungo la riga;
           keepdims=True lascia la forma (n, 1) cosi' la divisione e' riga per riga)
        3. 1 - prodotto tra matrici

    ESEMPIO (vettori a 2 dimensioni invece di 512, per fare i conti a mano)
        probes  = [[1, 0],      -> gia' lungo 1
                   [0, 2]]      -> diventa [0, 1]
        gallery = [[1, 0],
                   [1, 1],      -> diventa [0.707, 0.707]
                   [0, 1]]
        dist =   [[0.000, 0.293, 1.000],    probe 0: identico al template 0
                  [1.000, 0.293, 0.000]]    probe 1: identico al template 2

    INPUT
        probes  : matrice (n_probe x 512), una riga = embedding di un probe
        gallery : matrice (n_gallery x 512), una riga = embedding di un template
        Di solito ottenute con EmbeddingCache.get(lista di sample_id): l'ordine
        delle righe e' quello della lista, ed e' quello che collega ogni riga
        alla sua persona.
    RESTITUISCE : matrice (n_probe x n_gallery) di distanze in [0, 2]
    CHIAMA      : np.linalg.norm, @ (prodotto tra matrici)
    CHIAMATA DA : scripts/eval_lfw.py (D_known, D_unknown), tests/test_lfw_tools.py
    """
    p = np.asarray(probes, dtype=np.float32)
    g = np.asarray(gallery, dtype=np.float32)
    p = p / np.linalg.norm(p, axis=1, keepdims=True)   # ogni riga lunga 1
    g = g / np.linalg.norm(g, axis=1, keepdims=True)
    return 1.0 - p @ g.T                               # (P x d) @ (d x G) = (P x G)


class EmbeddingCache:
    """Embedding indicizzati per sample_id, salvati su disco in un file .npz.

    TEORIA
        Un embedding dipende solo dalla foto e dal modello, quindi si calcola
        una volta e si riusa. La cache ricorda anche le foto in cui il
        detector NON ha trovato volti, per non riprovarle ogni volta.

    CAMPI
        path    : file su disco, es. data/lfw/embeddings_arcface_center.npz
        vectors : dict sample_id -> vettore (512 float32 per ArcFace, 2891 per LBP)
                  es. {"Kate_Winslet_0002": array([... 512 numeri ...]), ...}
        failed  : set di sample_id senza volto rilevato, es. {"Persona_X_0003"}

    FORMATO DEL FILE .npz (un archivio di array numpy)
        sample_ids : array di N stringhe        ["Aaron_Eckhart_0001", ...]
        embeddings : matrice N x 512 (N x 2891 per LBP)   riga k = vettore di sample_ids[k]
        failed     : array di stringhe          i sample_id senza volto

    CREATA DA : scripts/eval_lfw.py, tests/test_lfw_tools.py
    USATA DA  : embed_rows (riempie), eval_lfw.py (get, failed, vectors)
    """

    def __init__(self, path: str | Path):
        """Apre la cache: se il file esiste la ricarica, altrimenti parte vuota.

        COME LO CALCOLA IL CODICE
            zip(sample_ids, embeddings) accoppia il k-esimo nome con la k-esima
            riga -> dizionario. allow_pickle=False: si caricano solo array,
            mai codice (piu' sicuro).
        CHIAMA : np.load
        """
        self.path = Path(path)
        self.vectors: dict[str, np.ndarray] = {}
        self.failed: set[str] = set()      # campioni in cui il detector non ha trovato volti
        if self.path.exists():
            data = np.load(self.path, allow_pickle=False)
            self.vectors = dict(zip(data["sample_ids"].tolist(), data["embeddings"]))
            self.failed = set(data["failed"].tolist())

    def save(self) -> None:
        """Scrive la cache su disco.

        COME LO CALCOLA IL CODICE
            1. ordina i sample_id e impila i loro embedding in una matrice N x 512
               (cache vuota -> matrice 0 x 512)
            2. scrive prima su "<nome>_tmp.npz" e poi lo rinomina sul file vero:
               se il programma si interrompe a meta' della scrittura, il file
               vecchio resta integro.
        CHIAMA : np.stack, np.savez, Path.replace
        CHIAMATA DA : embed_rows, tests/test_lfw_tools.py
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        ids = sorted(self.vectors)
        emb = np.stack([self.vectors[i] for i in ids]) if ids else np.zeros((0, 512), np.float32)
        tmp = self.path.with_name(self.path.stem + "_tmp.npz")   # es. embeddings_arcface_center_tmp.npz
        np.savez(tmp, sample_ids=np.array(ids), embeddings=emb,
                 failed=np.array(sorted(self.failed), dtype=str))
        tmp.replace(self.path)

    def missing(self, rows: list[dict]) -> list[dict]:
        """Le righe di manifest il cui embedding manca ancora.

        ESEMPIO : 3 righe, una gia' in vectors e una in failed -> resta 1 riga
        INPUT   : rows = righe di manifest (serve solo la chiave "sample_id")
        RESTITUISCE : le righe ne' in vectors ne' in failed, nello stesso ordine
        CHIAMATA DA : embed_rows, scripts/eval_lfw.py
        """
        return [r for r in rows if r["sample_id"] not in self.vectors and r["sample_id"] not in self.failed]

    def get(self, sample_ids) -> np.ndarray:
        """Matrice degli embedding dei sample_id dati, NELLO STESSO ORDINE.

        ESEMPIO : get(["Anna_Rossi_0001", "Bruno_Verdi_0001"]) -> matrice 2 x 512,
                  riga 0 = Anna, riga 1 = Bruno
        INPUT   : sample_ids = lista di sample_id tutti presenti in vectors
                  (altrimenti KeyError: eval_lfw.py filtra prima quelli senza volto)
        RESTITUISCE : matrice (len(sample_ids) x 512)
        CHIAMA  : np.stack
        CHIAMATA DA : scripts/eval_lfw.py, tests/test_lfw_tools.py
        """
        return np.stack([self.vectors[s] for s in sample_ids])


def embed_rows(rows: list[dict], embedder, cache: EmbeddingCache, save_every: int = 250,
               select: str = "largest", method: str = "arcface") -> None:
    """Calcola il vettore (ArcFace o LBP) dei campioni che mancano nella cache.

    TEORIA
        E' il passo "feature extraction" applicato a tutto il dataset.
        Ogni foto passa da: lettura -> detection -> scelta del volto ->
        alignment -> descrittore. Il descrittore dipende da `method`:
            "arcface": FaceEmbedder.embed           -> embedding di 512 numeri
            "lbp"    : FaceEmbedder.crop + lbp_descriptor -> vettore LBP di 2891 numeri
            "quality": FaceEmbedder.quality_inputs + quality_measures -> 5 misure di
                       qualita' [roll, yaw, r, SI, SY] (embedder con landmarks=True)
            "detectors": face.detect.detector_vector -> riquadri SCRFD e Viola-Jones e
                       ArcFace sui due ritagli non allineati (1032 numeri, nan dove manca)
            "magface": FaceEmbedder.crop + face.magface.MagFace.magnitude -> 1 numero,
                       la lunghezza del vettore MagFace (incertezza appresa)
        Detection, scelta del volto e alignment sono IDENTICI per i due
        metodi (FaceEmbedder._detect_and_choose + norm_crop), cosi' il
        confronto ArcFace contro LBP riguarda solo il descrittore.
        La cache va tenuta separata per metodo (eval_lfw.py usa un file diverso).

    COME LO CALCOLA IL CODICE
        per ogni riga che manca nella cache:
            prova embedder.embed(load_image(path)) e salva l'embedding sotto
            il sample_id della riga -> e' qui che il vettore viene LEGATO
            alla persona (tramite sample_id -> subject_id del manifest)
            se non trova volti (ValueError) mette il sample_id in cache.failed
        ogni save_every foto (e alla fine) salva su disco e stampa la velocita'.

    ESEMPIO (method="lbp")
        riga {"sample_id": "Kate_Winslet_0002", "path": ".../Kate_Winslet_0002.jpg"}
        -> load_image -> crop (112 x 112 x 3) -> lbp_descriptor -> 2891 numeri
        -> cache.vectors["Kate_Winslet_0002"]

    ESEMPIO di stampa (ogni 250 foto)
        "  250/13233 immagini  (1.0 img/s, mancano ~216 min)"
            rate = 250 foto / 250 s = 1.0 img/s
            eta  = (13233 - 250) / 1.0 = 12983 s -> / 60 = ~216 min

    INPUT
        rows       : righe di manifest (servono "sample_id" e "path")
        embedder   : un FaceEmbedder gia' creato (i modelli si caricano una volta)
        cache      : la EmbeddingCache da riempire (modificata sul posto)
        save_every : ogni quante foto salvare su disco (default 250)
        select     : quale volto prendere se ce n'e' piu' di uno:
                     "largest" (il piu' grande) o "center" (il piu' vicino al
                     centro, giusto per LFW: vedi FaceEmbedder.embed)
        method     : "arcface" (default, come prima), "lbp", "quality", "detectors" o "magface"
    RESTITUISCE : nulla; il risultato e' dentro cache.vectors / cache.failed e su disco
    CHIAMA      : EmbeddingCache.missing, face.load_image, embedder.embed (arcface)
                  oppure embedder.crop + face.lbp.lbp_descriptor (lbp),
                  oppure embedder.quality_inputs + face.quality.quality_measures (quality),
                  oppure face.detect.detector_vector (detectors),
                  oppure FaceEmbedder.crop + face.magface.MagFace.magnitude (magface),
                  EmbeddingCache.save
    CHIAMATA DA : scripts/eval_lfw.py (nessun test la chiama: richiede il modello e
                  delle foto; la parte di embedding e' verificata in test_face_selection)
    """
    from secondlook.face import load_image
    from secondlook.face.lbp import lbp_descriptor
    from secondlook.face.quality import quality_measures

    if method == "arcface":
        describe = lambda img: embedder.embed(img, select=select).embedding
    elif method == "lbp":
        describe = lambda img: lbp_descriptor(embedder.crop(img, select=select))
    elif method == "quality":
        describe = lambda img: quality_measures(*embedder.quality_inputs(img, select=select))
    elif method == "detectors":
        from secondlook.face.detect import detector_vector, haar_cascade

        cascade = haar_cascade()
        describe = lambda img: detector_vector(img, embedder, cascade, select=select)
    elif method == "magface":
        from secondlook.face.magface import MagFace

        magface = MagFace()
        describe = lambda img: np.array([magface.magnitude(embedder.crop(img, select=select))], dtype=np.float32)
    else:
        raise ValueError(f"metodo sconosciuto: {method!r} (usa 'arcface', 'lbp', 'quality', 'detectors' o 'magface')")

    todo = cache.missing(rows)
    if not todo:
        return
    start = time.time()
    for k, r in enumerate(todo, 1):              # k parte da 1: e' il numero di foto fatte
        try:
            cache.vectors[r["sample_id"]] = describe(load_image(r["path"]))
        except ValueError:                       # "No face detected"
            cache.failed.add(r["sample_id"])
        if k % save_every == 0 or k == len(todo):   # ogni 250 foto, e all'ultima
            cache.save()
            rate = k / (time.time() - start)         # foto al secondo
            eta = (len(todo) - k) / rate             # secondi che mancano
            print(f"  {k}/{len(todo)} immagini  ({rate:.1f} img/s, mancano ~{eta / 60:.0f} min)")
