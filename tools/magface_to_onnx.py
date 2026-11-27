"""Converte i pesi ufficiali di MagFace (PyTorch) in un modello ONNX, UNA volta sola.

PERCHE'
    I pesi di MagFace (Meng e colleghi, CVPR 2021) sono distribuiti come file PyTorch
    (.pth). Il progetto esegue i modelli con onnxruntime, come quelli di insightface:
    convertendo il modello in ONNX una volta, SecondLook non ha bisogno di PyTorch
    tra le dipendenze. Questo script e' l'unico pezzo che usa PyTorch, e serve solo
    a chi deve rifare la conversione: chi ha gia' il file .onnx non lo esegue.

COSA SERVE (fuori dal progetto, solo per questo script)
    pip install torch onnx
    git clone https://github.com/IrvingMeng/MagFace     (per la rete iresnet100)
    i pesi iResNet100 / MS1MV2 dal README di MagFace (Google Drive), es.
        data/models/magface_epoch_00025.pth

USO (dalla cartella SecondLook)
    python tools/magface_to_onnx.py --repo ../MagFace --pth data/models/magface_epoch_00025.pth
    -> data/models/magface_iresnet100.onnx

COSA FA
    1. costruisce la rete iresnet100 con il codice ufficiale (models/iresnet.py)
    2. carica i pesi: nel file i nomi hanno un prefisso in piu' ("features.module.xxx"),
       che si toglie come fa il codice ufficiale (inference/network_inf.py, clean_dict_inf);
       si controlla che TUTTI i pesi della rete siano stati trovati
    3. esporta in ONNX con il lotto di dimensione variabile (1 o piu' foto insieme)
    4. CONTROLLO: stesse foto casuali in PyTorch e in onnxruntime -> i vettori devono
       coincidere (differenza massima stampata a schermo, deve essere ~1e-4 o meno)
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def load_magface(repo: Path, pth: Path):
    """Rete iresnet100 di MagFace con i pesi ufficiali, pronta per l'inferenza.

    COME LO CALCOLA IL CODICE
        model = iresnet100(num_classes=512)   (512 = lunghezza del vettore)
        nei pesi ogni nome e' "features.module.<nome della rete>": si tengono le parti
        dopo i primi due pezzi e si controlla che nome e forma coincidano con la rete.

    INPUT       : repo = cartella del codice di MagFace; pth = file dei pesi
    RESTITUISCE : la rete PyTorch in modalita' valutazione (eval)
    SOLLEVA     : RuntimeError se qualche peso della rete manca nel file
    """
    import torch

    sys.path.insert(0, str(repo))
    from models import iresnet

    model = iresnet.iresnet100(pretrained=False, num_classes=512)
    # weights_only=False: il file ufficiale contiene anche oggetti non-tensore (es. l'epoca).
    # Caricare un .pth esegue codice Python: farlo solo con il file ufficiale del README.
    state = torch.load(pth, map_location="cpu", weights_only=False)["state_dict"]
    target = model.state_dict()
    clean = {}
    for k, v in state.items():
        name = ".".join(k.split(".")[2:])                    # "features.module.conv1.weight" -> "conv1.weight"
        if name in target and v.shape == target[name].shape:
            clean[name] = v
    missing = set(target) - set(clean)
    if missing:
        raise RuntimeError(f"{len(missing)} pesi della rete non trovati nel file, es. {sorted(missing)[:3]}")
    model.load_state_dict(clean)
    return model.eval()


def main() -> None:
    import onnxruntime as ort
    import torch

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True, help="cartella del codice ufficiale di MagFace")
    parser.add_argument("--pth", default="data/models/magface_epoch_00025.pth")
    parser.add_argument("--out", default="data/models/magface_iresnet100.onnx")
    args = parser.parse_args()

    model = load_magface(Path(args.repo), Path(args.pth))
    dummy = torch.rand(1, 3, 112, 112)                       # una foto finta: 3 colori, 112 x 112, valori in [0, 1]
    torch.onnx.export(model, dummy, args.out, input_names=["input"], output_names=["feature"],
                      dynamic_axes={"input": {0: "batch"}, "feature": {0: "batch"}},
                      opset_version=17, dynamo=False)

    # controllo: PyTorch e onnxruntime devono dare gli stessi vettori
    x = torch.rand(4, 3, 112, 112)
    with torch.no_grad():
        ref = model(x).numpy()
    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"input": x.numpy()})[0]
    diff = float(np.abs(ref - out).max())
    print(f"salvato {args.out}")
    print(f"lunghezze PyTorch   {np.linalg.norm(ref, axis=1).round(3)}")
    print(f"lunghezze ONNX      {np.linalg.norm(out, axis=1).round(3)}")
    print(f"differenza massima tra i vettori: {diff:.2e}")
    if diff > 1e-3:
        raise SystemExit("ATTENZIONE: la conversione non e' fedele")


if __name__ == "__main__":
    main()
