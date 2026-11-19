"""Confronta due foto di volto e stampa la loro distanza coseno ArcFace.

TEORIA
    E' una VERIFICA 1:1 a mano: due foto, un confronto, una decisione
    "stessa persona / persone diverse" con una soglia fissa.
    Serve per provare velocemente il modello su foto proprie, non per
    valutarlo (per quello c'e' scripts/eval_lfw.py, che misura EER, ecc.).

USO (dalla cartella SecondLook)
    uv run python scripts/compare_faces.py foto1.jpg foto2.jpg

ESEMPIO DI STAMPA (numeri indicativi)
    cosine distance: 0.2530 (probably same person)
    detector confidence: 0.912 / 0.874
    (con due foto di Kate_Winslet di LFW scegliendo il volto centrale la
    distanza era 0.253; qui pero' si usa il default "largest", vedi sotto)

MAPPA DELLE CHIAMATE
    main -> FaceEmbedder (carica i modelli)
         -> load_image (x2) -> FaceEmbedder.embed (x2)
         -> cosine_distance
    Tutte in src/secondlook/face/embed.py.
"""

import argparse

from secondlook.face import FaceEmbedder, cosine_distance, load_image

# Soglia indicativa, scelta a mano prima di avere misure: e' solo una convenzione.
# LA SOGLIA VERA VA FISSATA IN SEGUITO sui dati di tuning
# (es. quella dell'EER o quella a FAR fissato, da eval/metrics.py).
# Per confronto, su LFW --quick: distanza media genuini 0.320, impostori 0.997,
# soglia dell'EER 0.821. Quindi 0.6 e' piu' severa: rifiuta qualche genuino in piu'.
REFERENCE_THRESHOLD = 0.6


def main() -> None:
    """Legge le due foto da riga di comando, le confronta e stampa il verdetto.

    COME LO CALCOLA IL CODICE
        1. argparse legge i due percorsi scritti dopo il nome dello script
        2. FaceEmbedder() carica SCRFD + ArcFace (qualche secondo)
        3. per ogni foto: load_image -> embed -> FaceSample (512 numeri + bbox + det_score).
           embed usa select="largest": se in una foto ci sono piu' persone si
           prende il volto piu' grande
        4. cosine_distance tra i due embedding
        5. distanza < 0.6 -> "probably same person", altrimenti "different people"

    INPUT (da riga di comando)
        image_a, image_b : percorsi di due file immagine. Lo script non sa di
                           chi sono: siamo noi, guardando il verdetto, a sapere
                           se doveva essere un confronto genuino o impostore.
    STAMPA      : la distanza, il verdetto e la confidenza del detector per le due foto
    SOLLEVA     : FileNotFoundError (foto illeggibile), ValueError (nessun volto)
    CHIAMA      : FaceEmbedder, load_image, FaceEmbedder.embed, cosine_distance
    CHIAMATA DA : riga di comando
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_a")      # primo percorso dopo il nome dello script
    parser.add_argument("image_b")      # secondo percorso
    args = parser.parse_args()

    embedder = FaceEmbedder()
    a = embedder.embed(load_image(args.image_a))     # FaceSample della prima foto
    b = embedder.embed(load_image(args.image_b))     # FaceSample della seconda foto
    distance = cosine_distance(a.embedding, b.embedding)   # es. 0.2530

    verdict = "probably same person" if distance < REFERENCE_THRESHOLD else "probably different people"
    print(f"cosine distance: {distance:.4f} ({verdict})")
    print(f"detector confidence: {a.det_score:.3f} / {b.det_score:.3f}")


if __name__ == "__main__":
    main()
