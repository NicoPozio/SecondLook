# SecondLook

Login multibiometrico a step-up: il sistema riconosce l'utente dal volto e chiede un secondo tratto (il ritmo di digitazione di una frase) solo quando la risposta del volto non è affidabile. L'affidabilità è misurata con l'SRR (System Response Reliability) su embedding facciali ArcFace.

Progetto d'esame di Biometric Systems, Sapienza Università di Roma.

## Requisiti

- Python 3.10 o superiore
- [uv](https://docs.astral.sh/uv/) per ambiente e dipendenze
- Git
- Circa 300 MB liberi per i modelli pre-addestrati

Su Windows, se l'installazione di insightface fallisce durante la compilazione, servono i [Build Tools di Visual Studio](https://visualstudio.microsoft.com/visual-cpp-build-tools/) con il componente "Sviluppo di applicazioni desktop con C++". In alternativa si può usare WSL.

## Installazione

```bash
git clone <url-del-repository> SecondLook
cd SecondLook
uv sync
```

`uv sync` crea l'ambiente virtuale in `.venv` e installa le versioni esatte indicate in `uv.lock`, così tutto il gruppo lavora con le stesse librerie.

Al primo utilizzo insightface scarica automaticamente il pacchetto di modelli `buffalo_l` (circa 280 MB) in `~/.insightface/models`. Dal pacchetto usiamo due modelli: il detector SCRFD per rilevare e allineare il volto, e una ResNet-50 addestrata con loss ArcFace per l'embedding a 512 dimensioni. I pesi sono distribuiti per uso di ricerca non commerciale.

## Verifica dell'installazione

```bash
uv run pytest -v
```

Tutti i test devono passare. I test sul modello scaricano `buffalo_l` se non è già presente, quindi la prima esecuzione richiede qualche minuto.

## Confronto tra due volti

```bash
uv run python scripts/compare_faces.py data/foto_a.jpg data/foto_b.jpg
```

Lo script stampa la distanza coseno tra i due volti (0 significa identici) e la confidenza del detector. Se in una foto ci sono più persone, viene usato il volto più grande. Indicativamente, la stessa persona resta sotto 0.6 e persone diverse superano 0.8; la soglia definitiva sarà fissata dalla valutazione su LFW.

## Struttura

```
SecondLook/
  pyproject.toml        dipendenze e configurazione di pytest
  uv.lock               versioni esatte delle dipendenze
  src/secondlook/
    face/embed.py       rilevamento, allineamento ed embedding ArcFace
  scripts/              script da riga di comando
  tests/                test con pytest
  data/                 dati locali, esclusi da git
```

## Dati

Le foto e le registrazioni dei soggetti vanno in `data/`, che è esclusa da git: i dati biometrici non devono finire nel repository.

## Se serve un requirements.txt

Per ambienti senza uv (ad esempio Colab o Kaggle) si può generare dal lock, invece di scriverlo a mano:

```bash
uv export --format requirements-txt --no-hashes > requirements.txt
```

Va rigenerato ogni volta che si aggiunge una dipendenza.