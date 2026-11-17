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
git clone https://github.com/NicoPozio/SecondLook.git
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

## Test di precisione dei timestamp

Il keystroke misura per quanto resta premuto ogni tasto e quanto passa tra un tasto e il successivo, cioè differenze di pochi millisecondi. I browser però arrotondano di proposito i tempi degli eventi, per motivi di sicurezza, e di quanto lo fanno dipende dal browser e dalle sue impostazioni. Prima di costruire l'app di acquisizione va quindi verificato che il browser scelto sia abbastanza preciso.

La pagina `tools/timestamp_test.html` fa questa verifica. Si avvia con:

```bash
uv run python -m http.server 8000 --directory tools
```

e si apre nel browser da provare all'indirizzo `http://localhost:8000/timestamp_test.html`. Il server serve perché la pagina giri nelle stesse condizioni dell'app, che sarà anch'essa servita da un server. Nella pagina:

1. **Misura il timer**: stima il passo dell'orologio del browser (`performance.now()`).
2. Si digita la frase almeno 5 volte, premendo Invio dopo ognuna: la pagina registra `keydown` e `keyup`, stima il passo dei timestamp degli eventi (`event.timeStamp`) e calcola hold e flight time.
3. Si legge il verdetto: passo fino a 1 ms OK, fino a 5 ms accettabile, oltre non adatto. Le soglie sono una regola pratica basata sull'errore di arrotondamento (passo / √12) confrontato con la variabilità naturale di chi digita.
4. **Esporta JSON** salva misure ed eventi grezzi nella cartella Download.

Tutto è calcolato nel browser e nessun dato lascia il computer. I JSON esportati contengono il ritmo di digitazione di una persona, quindi **non vanno messi nel repository**: se servono vicino al progetto, vanno in `data/`, che è esclusa da git.

Primi risultati (17 novembre 2026, Windows): **Chrome 157** ed **Edge 157** hanno entrambi un passo di 0.1 ms sia per l'orologio sia per gli eventi, con un errore tipico di arrotondamento di circa 0.03 ms. Sono quindi equivalenti per il keystroke (Edge usa lo stesso motore di Chrome). Firefox non è stato ancora provato. Qualunque browser si scelga, va usato con la stessa versione per tutte le sessioni di raccolta, e il test va ripetuto sul computer usato per la raccolta.

Gli eventi vanno sempre letti con `event.timeStamp`, l'istante in cui il browser ha creato l'evento, e non con `performance.now()` dentro il gestore: il ritardo con cui il codice riceve un evento varia e può arrivare a 20-30 ms.

## Struttura

```
SecondLook/
  pyproject.toml        dipendenze e configurazione di pytest
  uv.lock               versioni esatte delle dipendenze
  src/secondlook/
    face/embed.py       rilevamento, allineamento ed embedding ArcFace
    reliability/        stimatori di affidabilità: SRR, margine, Weibull (da fare)
    keystroke/          feature di digitazione e matcher (da fare)
    fusion/             normalizzazione, step-up, aggiornamento template (da fare)
    eval/               dataset, matrici delle distanze, metriche (da fare)
    app/                app web di acquisizione e demo (da fare)
  scripts/              script da riga di comando
  tools/                strumenti di supporto, es. il test dei timestamp
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
