"""
Compare two face photos and print their ArcFace cosine distance
"""

import argparse

from secondlook.face import FaceEmbedder, cosine_distance, load_image

#0.6 valore arbitrario, di solito persone diverse hanno piu di 0.8 come distanza ma sono semplici convenzioni
#LA THRESHOLD VERA E PROPRIA VA FISSATA IN SEGUITO
REFERENCE_THRESHOLD = 0.6


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_a")
    parser.add_argument("image_b")
    args = parser.parse_args()

    embedder = FaceEmbedder()
    a = embedder.embed(load_image(args.image_a))
    b = embedder.embed(load_image(args.image_b))
    distance = cosine_distance(a.embedding, b.embedding)

    verdict = "probably same person" if distance < REFERENCE_THRESHOLD else "probably different people"
    print(f"cosine distance: {distance:.4f} ({verdict})")
    print(f"detector confidence: {a.det_score:.3f} / {b.det_score:.3f}")


if __name__ == "__main__":
    main()