"""Dataset nel formato comune del progetto (Step 0, Fase 2).

TEORIA
    Per valutare il sistema servono campioni di cui si sa CHI e' la persona
    (ground truth = etichetta vera). Useremo due fonti diverse, LFW e la
    nostra raccolta, e per non scrivere codice diverso per ognuna tutti i
    campioni vengono descritti con la stessa tabella, il MANIFEST.
    Ogni campione e' una riga (un dizionario Python) con queste colonne:

    subject_id   chi e' la persona            es. "George_W_Bush"
    source       da quale dataset viene        "lfw" oppure "own"
    session      sessione di acquisizione      "" per LFW (non ha sessioni)
    trait        tratto biometrico             "face" oppure "keystroke"
    sample_id    nome unico del campione       es. "George_W_Bush_0001"
    path         percorso del file             es. "data/lfw/lfw/George_W_Bush/George_W_Bush_0001.jpg"
    split        tuning / test                 "" finche' non si assegna

    E' il manifest a portare l'IDENTITA': gli embedding e le distanze sono
    solo numeri, e si risale alla persona tramite sample_id -> subject_id.

    LFW (Labeled Faces in the Wild): 13.233 foto 250 x 250 di 5.749 persone,
    1.680 con almeno 2 foto, 4.069 con una sola. Il file pairs.txt fissa il
    protocollo ufficiale di verifica: 6.000 coppie in 10 gruppi (fold).

COSA C'E' IN QUESTO FILE
    download_lfw          scarica ed estrae LFW (usa _download)
    lfw_manifest          una riga di manifest per ogni foto di LFW
    write_manifest / read_manifest   salva / rilegge il manifest come CSV
    Pair, read_lfw_pairs  le 6.000 coppie ufficiali (usa lfw_sample_id)
    split_gallery_probes  divide i campioni in gallery / registrati / sconosciuti

MAPPA DELLE CHIAMATE (scripts/eval_lfw.py le usa in quest'ordine)
    download_lfw -> lfw_manifest -> write_manifest
                 -> read_lfw_pairs -> lfw_sample_id
                 -> split_gallery_probes
    Altri: tests/test_lfw_tools.py (usa tutte, su un mini LFW finto)
"""

from __future__ import annotations

import csv
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

# ordine delle colonne nel file CSV del manifest
MANIFEST_COLUMNS = ["subject_id", "source", "session", "trait", "sample_id", "path", "split"]

# Copie ufficiali di LFW ospitate su figshare (le stesse usate da scikit-learn).
# Il sito originale dell'Universita' del Massachusetts e' spesso irraggiungibile.
LFW_URLS = {
    "lfw.tgz": "https://ndownloader.figshare.com/files/5976018",   # archivio con tutte le foto
    "pairs.txt": "https://ndownloader.figshare.com/files/5976006",  # le 6000 coppie
}


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _download(url: str, dest: Path) -> None:
    """Scarica un file da internet mostrando l'avanzamento.

    COME LO CALCOLA IL CODICE
        Scrive prima su "<nome>.part" e rinomina solo alla fine: se il
        download si interrompe resta un .part, mai un file a meta' con il
        nome giusto (che al giro dopo verrebbe creduto completo).
        Legge a blocchi da 1 MB (1 << 20 = 2^20 = 1.048.576 byte).
        `while chunk := ...` = "leggi un blocco e continua finche' non e' vuoto".

    ESEMPIO : stampa "  52.4 / 172.7 MB" aggiornando la stessa riga (\\r)
    INPUT   : url da scaricare; dest = percorso finale, es. data/lfw/lfw.tgz
    CHIAMA  : urllib.request.urlopen, Path.replace
    CHIAMATA DA : download_lfw
    """
    tmp = dest.with_suffix(dest.suffix + ".part")    # es. lfw.tgz -> lfw.tgz.part
    print(f"scarico {dest.name} ...")
    with urllib.request.urlopen(url) as response, open(tmp, "wb") as out:
        total = int(response.headers.get("Content-Length") or 0)   # dimensione, se il server la dice
        done = 0
        while chunk := response.read(1 << 20):          # blocchi da 1 MB
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:6.1f} / {total / 1e6:.1f} MB", end="")
    print()
    tmp.replace(dest)            # rinomina solo a download finito: niente file a meta'


def download_lfw(root: str | Path = "data/lfw") -> Path:
    """Scarica ed estrae LFW in root, solo se non c'e' gia'.

    COME LO CALCOLA IL CODICE
        - se manca la cartella delle foto: scarica lfw.tgz (se manca) e lo estrae
        - se manca pairs.txt: lo scarica
        Ogni passo si salta se il suo file esiste: rilanciare non riscarica nulla.
        filter="data" e' un controllo di sicurezza di Python 3.12+: impedisce
        che un archivio scriva fuori dalla cartella. Le versioni piu' vecchie
        non lo conoscono (TypeError) e allora si estrae senza.

    STRUTTURA FINALE
        root/lfw/<Nome_Persona>/<Nome_Persona>_0001.jpg
        root/pairs.txt
        Se il download non funziona si possono mettere a mano gli stessi file
        in quella posizione: la funzione li trova e non scarica nulla.

    INPUT       : root = cartella di destinazione (default "data/lfw", esclusa da git)
    RESTITUISCE : root come Path
    CHIAMA      : _download, tarfile.open, extractall
    CHIAMATA DA : scripts/eval_lfw.py
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    images = root / "lfw"
    if not images.is_dir():
        archive = root / "lfw.tgz"
        if not archive.exists():
            _download(LFW_URLS["lfw.tgz"], archive)
        print("estraggo le immagini ...")
        with tarfile.open(archive) as tar:
            try:
                tar.extractall(root, filter="data")   # filtro di sicurezza (Python recenti)
            except TypeError:                         # Python piu' vecchi non lo conoscono
                tar.extractall(root)
    pairs = root / "pairs.txt"
    if not pairs.exists():
        _download(LFW_URLS["pairs.txt"], pairs)
    return root


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def lfw_manifest(root: str | Path = "data/lfw") -> list[dict]:
    """Una riga di manifest per ogni foto di LFW.

    TEORIA
        In LFW l'etichetta e' il nome della cartella: tutte le foto in
        lfw/Kate_Winslet/ sono di Kate Winslet. Da li' si ricava subject_id.

    COME LO CALCOLA IL CODICE
        per ogni sottocartella (= persona), in ordine alfabetico:
            per ogni .jpg, in ordine: una riga con
                subject_id = nome della cartella
                sample_id  = nome del file senza ".jpg" (img.stem)
                path       = percorso con "/" anche su Windows (as_posix)

    ESEMPIO
        data/lfw/lfw/Kate_Winslet/Kate_Winslet_0002.jpg  diventa
        {"subject_id": "Kate_Winslet", "source": "lfw", "session": "",
         "trait": "face", "sample_id": "Kate_Winslet_0002",
         "path": "data/lfw/lfw/Kate_Winslet/Kate_Winslet_0002.jpg", "split": ""}
        Su tutto LFW: 13.233 righe.

    INPUT       : root = cartella che contiene lfw/ (come lasciata da download_lfw)
    RESTITUISCE : list[dict], ordinata per persona e poi per numero di foto
    SOLLEVA     : FileNotFoundError se root/lfw non esiste
    CHIAMA      : Path.iterdir, Path.glob
    CHIAMATA DA : scripts/eval_lfw.py, tests/test_lfw_tools.py
    """
    images = Path(root) / "lfw"
    if not images.is_dir():
        raise FileNotFoundError(f"cartella di LFW non trovata: {images}")
    rows = []
    for person_dir in sorted(p for p in images.iterdir() if p.is_dir()):   # es. .../Kate_Winslet
        for img in sorted(person_dir.glob("*.jpg")):                       # es. Kate_Winslet_0001.jpg
            rows.append({
                "subject_id": person_dir.name,   # "Kate_Winslet"
                "source": "lfw",
                "session": "",
                "trait": "face",
                "sample_id": img.stem,           # "Kate_Winslet_0001"
                "path": img.as_posix(),
                "split": "",
            })
    return rows


def write_manifest(rows: list[dict], path: str | Path) -> None:
    """Salva il manifest come file CSV (si apre anche con Excel).

    ESEMPIO del file prodotto:
        subject_id,source,session,trait,sample_id,path,split
        Kate_Winslet,lfw,,face,Kate_Winslet_0001,data/lfw/lfw/Kate_Winslet/Kate_Winslet_0001.jpg,

    INPUT   : rows = lista di dizionari con le chiavi di MANIFEST_COLUMNS;
              path = file da scrivere, es. data/lfw/manifest.csv
    NOTA    : newline="" evita righe vuote in piu' su Windows
    CHIAMA  : csv.DictWriter
    CHIAMATA DA : scripts/eval_lfw.py, tests/test_lfw_tools.py
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()      # prima riga: i nomi delle colonne
        writer.writerows(rows)


def read_manifest(path: str | Path) -> list[dict]:
    """Rilegge un manifest CSV: restituisce le stesse righe scritte da write_manifest.

    NOTA        : dal CSV ogni valore torna come stringa
    CHIAMA      : csv.DictReader
    CHIAMATA DA : tests/test_lfw_tools.py (in futuro: la nostra raccolta)
    """
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Coppie ufficiali di LFW (protocollo di verifica)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Pair:
    """UNA coppia del protocollo di verifica di LFW.

    TEORIA
        Ogni coppia e' un confronto 1:1 da fare. Se same = True la distanza
        finira' nella lista dei GENUINI, altrimenti in quella degli IMPOSTORI
        (eval/metrics.far_frr). E' qui che nasce la separazione genuine/impostor.
        frozen=True = non si puo' modificare dopo averla creata.

    CAMPI (esempio)
        sample_a : "Abel_Pacheco_0001"   sample_id della prima foto
        sample_b : "Abel_Pacheco_0004"   sample_id della seconda foto
        same     : True                  stessa persona (coppia genuina)
        fold     : 0                     gruppo 0..9 (600 coppie per gruppo)

    CREATA DA : read_lfw_pairs
    USATA DA  : scripts/eval_lfw.py (embedding da calcolare, distanze, 10-fold)
    """

    sample_a: str     # sample_id, es. "Abel_Pacheco_0001"
    sample_b: str
    same: bool        # True = stessa persona (coppia genuina)
    fold: int         # 0..9: LFW divide le 6000 coppie in 10 gruppi da 600


def lfw_sample_id(name: str, number: str | int) -> str:
    """Nome + numero di pairs.txt -> sample_id del manifest.

    ESEMPIO : ("Abel_Pacheco", "4") -> "Abel_Pacheco_0004"
              (:04d = intero scritto con 4 cifre, zeri davanti, come nei file di LFW)
    CHIAMATA DA : read_lfw_pairs, tests/test_lfw_tools.py
    """
    return f"{name}_{int(number):04d}"


def read_lfw_pairs(root: str | Path = "data/lfw") -> list[Pair]:
    """Legge pairs.txt e restituisce le coppie ufficiali.

    FORMATO DEL FILE
        prima riga: quanti gruppi e quante coppie per tipo, es. "10\\t300"
        poi, per ogni gruppo, 300 righe "stessa persona" (3 campi):
            Abel_Pacheco    1    4          -> foto 1 e foto 4 di Abel_Pacheco
        seguite da 300 righe "persone diverse" (4 campi):
            Nome_A    1    Nome_B    2      -> foto 1 di A e foto 2 di B
        Totale 10 x (300 + 300) = 6.000 coppie.

    COME LO CALCOLA IL CODICE
        - salta le righe vuote, conta le altre con k = 0, 1, 2, ...
        - fold = k // (2 * 300) = k // 600 -> righe 0..599 gruppo 0,
          600..1199 gruppo 1, ...  (// = divisione intera)
        - 3 campi -> coppia genuina, 4 campi -> impostore, altro -> errore
        - alla fine controlla che siano proprio 6.000 (file incompleto = errore)

    ESEMPIO (mini file del test: prima riga "1\\t1" = 1 gruppo, 1 coppia per tipo)
        Anna_Rossi    1   3                  -> Pair("Anna_Rossi_0001", "Anna_Rossi_0003", True, 0)
        Bruno_Verdi   2   Carla_Bianchi  1   -> Pair("Bruno_Verdi_0002", "Carla_Bianchi_0001", False, 0)

    INPUT       : root = cartella che contiene pairs.txt
    RESTITUISCE : list[Pair], nell'ordine del file
    SOLLEVA     : ValueError per righe malformate o numero di coppie sbagliato
    CHIAMA      : lfw_sample_id, Pair
    CHIAMATA DA : scripts/eval_lfw.py, tests/test_lfw_tools.py
    """
    lines = (Path(root) / "pairs.txt").read_text(encoding="utf-8").split("\n")
    n_folds, n_per_type = (int(x) for x in lines[0].split())   # es. 10, 300
    pairs = []
    for k, line in enumerate(x for x in lines[1:] if x.strip()):
        parts = line.split()                    # spezza su spazi/tab: es. ["Abel_Pacheco", "1", "4"]
        fold = k // (2 * n_per_type)            # es. k = 650 -> 650 // 600 = gruppo 1
        if len(parts) == 3:
            name, a, b = parts
            pairs.append(Pair(lfw_sample_id(name, a), lfw_sample_id(name, b), True, fold))
        elif len(parts) == 4:
            name_a, a, name_b, b = parts
            pairs.append(Pair(lfw_sample_id(name_a, a), lfw_sample_id(name_b, b), False, fold))
        else:
            raise ValueError(f"riga non valida in pairs.txt: {line!r}")
    if len(pairs) != n_folds * 2 * n_per_type:
        raise ValueError(f"attese {n_folds * 2 * n_per_type} coppie, lette {len(pairs)}")
    return pairs


# ---------------------------------------------------------------------------
# Divisione per l'identificazione
# ---------------------------------------------------------------------------

def split_gallery_probes(rows: list[dict]):
    """Divide i campioni in gallery, probe di persone registrate e probe di sconosciuti.

    TEORIA
        Identificazione = confrontare un probe con TUTTA la gallery.
        Gallery e probe della stessa persona devono essere campioni diversi
        (nel nostro sistema: sessione 1 = enrollment, sessione 2 = login).
        LFW non ha sessioni, quindi lo si simula con foto diverse:
          - persona con 2 o piu' foto: la prima va in gallery (e' il suo
            template), le altre sono probe di una persona REGISTRATA
            -> servono per CMS (closed-set) e DIR (open-set)
          - persona con 1 sola foto: non ha nulla in gallery, quindi la sua
            foto e' un probe di uno SCONOSCIUTO -> serve per il false alarm

    COME LO CALCOLA IL CODICE
        1. raggruppa le righe per subject_id (setdefault crea la lista se manca)
        2. per ogni persona, ordina le sue foto per sample_id e smista

    ESEMPIO (mini LFW del test)
        Anna_Rossi    3 foto -> gallery: Anna_Rossi_0001
                                registrati: Anna_Rossi_0002, Anna_Rossi_0003
        Bruno_Verdi   2 foto -> gallery: Bruno_Verdi_0001
                                registrati: Bruno_Verdi_0002
        Carla_Bianchi 1 foto -> sconosciuti: Carla_Bianchi_0001
        Su tutto LFW: gallery 1.680, registrati 13.233 - 1.680 - 4.069 = 7.484,
        sconosciuti 4.069.

    INPUT       : rows = righe di manifest (es. da lfw_manifest); si usano
                  subject_id per raggruppare e sample_id per ordinare
    RESTITUISCE : (gallery, known_probes, unknown_probes), tre liste di righe
                  di manifest (dizionari), quindi ognuna porta il suo subject_id
    CHIAMA      : sorted, dict.setdefault
    CHIAMATA DA : scripts/eval_lfw.py, tests/test_lfw_tools.py
    """
    by_subject: dict[str, list[dict]] = {}
    for r in rows:
        by_subject.setdefault(r["subject_id"], []).append(r)   # es. {"Anna_Rossi": [r1, r2, r3], ...}
    gallery, known, unknown = [], [], []
    for subject in sorted(by_subject):
        samples = sorted(by_subject[subject], key=lambda r: r["sample_id"])
        if len(samples) >= 2:
            gallery.append(samples[0])          # prima foto = template
            known.extend(samples[1:])           # le altre = tentativi di login
        else:
            unknown.append(samples[0])          # nessun template = sconosciuto
    return gallery, known, unknown
