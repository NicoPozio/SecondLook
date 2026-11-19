# SecondLook

Login multibiometrico a step-up: il sistema riconosce l'utente dal volto e chiede un secondo tratto (il ritmo di digitazione di una frase) solo quando la risposta del volto non è affidabile. L'affidabilità è misurata con l'SRR (System Response Reliability) su embedding facciali ArcFace.

Progetto d'esame di Biometric Systems, Sapienza Università di Roma.

## Stato del progetto

| Fase | Contenuto | Stato |
|---|---|---|
| 0 | test di precisione dei timestamp del browser | fatto |
| 1 | embedding del volto con insightface (`face/embed.py`) | fatto |
| 2 | metriche, caricamento di LFW, matrici delle distanze, valutazione di ArcFace su LFW | fatto: accuratezza 99,83 % su LFW |
| 3 e successive | SRR, keystroke, fusione step-up, app di acquisizione | da fare |

## Requisiti

- Python 3.10 o superiore
- [uv](https://docs.astral.sh/uv/) per ambiente e dipendenze
- Git
- Circa 300 MB liberi per i modelli pre-addestrati, più circa 400 MB se si scarica LFW

Su Windows, se l'installazione di insightface fallisce durante la compilazione, servono i [Build Tools di Visual Studio](https://visualstudio.microsoft.com/visual-cpp-build-tools/) con il componente "Sviluppo di applicazioni desktop con C++". In alternativa si può usare WSL.

## Installazione

```bash
git clone https://github.com/NicoPozio/SecondLook.git
cd SecondLook
uv sync
```

`uv sync` crea l'ambiente virtuale in `.venv` e installa le versioni esatte indicate in `uv.lock`, così tutto il gruppo lavora con le stesse librerie.

Al primo utilizzo insightface scarica automaticamente il pacchetto di modelli `buffalo_l` (circa 280 MB) in `~/.insightface/models`. Dal pacchetto usiamo due modelli: il detector SCRFD (`det_10g.onnx`) per rilevare il volto e trovare i 5 punti di riferimento usati per l'allineamento, e una ResNet-50 addestrata con loss ArcFace sul dataset WebFace600K (`w600k_r50.onnx`) per l'embedding a 512 dimensioni. I pesi sono distribuiti per uso di ricerca non commerciale.

## Verifica dell'installazione

```bash
uv run pytest -v
```

Devono passare tutti i 30 test. I test sul modello scaricano `buffalo_l` se non è già presente, quindi la prima esecuzione richiede qualche minuto.

| File di test | Cosa controlla |
|---|---|
| `tests/test_embed.py` | distanza coseno, rilevamento dei volti, stesso volto più vicino degli altri, errore se non c'è nessun volto |
| `tests/test_face_selection.py` | scelta del volto quando in una foto ce n'è più di uno (`largest` e `center`) |
| `tests/test_metrics.py` | FAR, FRR, ROC, EER, d′, CMS, DIR, calcolati su esempi piccoli verificabili a mano |
| `tests/test_lfw_tools.py` | manifest, lettura di `pairs.txt`, divisione gallery/probe, matrice delle distanze, cache degli embedding, su un mini LFW finto |

I test usano solo dati piccoli o finti e non richiedono LFW.

## Confronto tra due volti

```bash
uv run python scripts/compare_faces.py data/foto_a.jpg data/foto_b.jpg
```

Lo script stampa la distanza coseno tra i due volti (0 significa identici) e la confidenza del detector. Se in una foto ci sono più persone, viene usato il volto più grande. La soglia dello script (0.6) è solo indicativa. Su LFW la distanza media è 0.33 per la stessa persona e 1.00 per persone diverse, e la soglia dell'EER è 0.818; la soglia definitiva per il login sarà fissata sui nostri dati.

## Scelta del volto

`FaceEmbedder.embed(img, select=...)` calcola l'embedding di **un solo** volto per immagine. Prima rileva tutti i volti, poi ne sceglie uno guardando solo i riquadri, e infine calcola allineamento ed ArcFace solo su quello.

- `select="largest"` (predefinito): il volto più grande. È la scelta giusta per il login, dove l'utente è la persona più vicina alla webcam.
- `select="center"`: il volto più vicino al centro dell'immagine. Si usa per LFW, dove la persona etichettata è sempre centrata ma ai bordi possono comparire volti più grandi di altre persone.

Il codice che chiama `embed()` senza `select` continua a funzionare come prima.

## Valutazione di ArcFace su LFW

```bash
uv run python scripts/eval_lfw.py --quick    # prova veloce: 600 coppie, 100 persone
uv run python scripts/eval_lfw.py            # LFW completo: i numeri da riportare
```

Lo script scarica LFW in `data/lfw` se non c'è (foto e `pairs.txt`), crea il manifest e calcola gli embedding delle foto necessarie. Gli embedding vengono salvati in una cache (`data/lfw/embeddings_arcface_center.npz`): se lo script si interrompe, al riavvio riparte da dove era arrivato, e le esecuzioni successive durano circa un minuto.

Calcolare gli embedding di tutto LFW (13.233 foto) su CPU richiede da circa 2 ore in su, a seconda del computer. Un portatile a batteria o in risparmio energetico può essere anche 10-20 volte più lento.

Lo script fa tre valutazioni:

| Valutazione | Dati usati | Cosa misura |
|---|---|---|
| **Verifica 1:1** | le 6.000 coppie ufficiali di `pairs.txt` (10 gruppi) | EER, d′, distanze medie, accuratezza con il protocollo a 10 gruppi |
| **Identificazione closed-set** | tutto LFW: prima foto di ogni persona con almeno 2 foto in gallery, le altre come probe | Recognition Rate, CMS(5), CMS(10) |
| **Identificazione open-set** | come sopra, più le persone con una sola foto come sconosciuti | DIR al rango 1 con false alarm all'1 % e al 10 % |

Con `--quick` si usano solo il primo gruppo di coppie, le prime 100 persone e 100 sconosciuti, e l'accuratezza a 10 gruppi non viene calcolata. I risultati vengono salvati in `data/lfw/results_arcface.json` (oppure `results_arcface_quick.json`).

### Risultati (LFW completo, 19 novembre 2026)

| | Risultato |
|---|---|
| Foto senza volto rilevato | 18 su 13.233 (17 coppie escluse dalla verifica) |
| **Accuratezza di verifica (10 gruppi)** | **99,83 % (± 0,20)** |
| EER | 0,30 % alla soglia 0,818 |
| Distanza media genuini / impostori | 0,327 / 0,996 |
| d′ | 7,97 |
| Closed-set (1.677 persone, 7.467 probe): RR / CMS(5) / CMS(10) | 99,77 % / 99,84 % / 99,87 % |
| Open-set (4.067 sconosciuti): DIR con false alarm 0,98 % | 99,64 % (soglia 0,675) |
| Open-set: DIR con false alarm 9,98 % | 99,72 % (soglia 0,733) |

L'accuratezza di verifica coincide con quella dichiarata da insightface per lo stesso modello (`buffalo_l`, 99,83 %). EER, CMS e DIR non hanno un valore ufficiale di confronto su LFW: la divisione gallery/probe è nostra.

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
  pyproject.toml          dipendenze e configurazione di pytest
  uv.lock                 versioni esatte delle dipendenze
  src/secondlook/
    face/embed.py         rilevamento (SCRFD), scelta del volto, allineamento ed embedding ArcFace
    eval/datasets.py      manifest nel formato comune, download di LFW, coppie di pairs.txt, divisione gallery/probe
    eval/matrices.py      cache degli embedding (.npz) e matrice delle distanze probe x gallery
    eval/metrics.py       FAR, FRR, ROC, DET, EER, d', CMS/CMC, DIR e false alarm open-set
    reliability/          stimatori di affidabilità: SRR, margine, Weibull (da fare)
    keystroke/            feature di digitazione e matcher (da fare)
    fusion/               normalizzazione, step-up, aggiornamento template (da fare)
    app/                  app web di acquisizione e demo (da fare)
  scripts/
    compare_faces.py      distanza tra due foto
    eval_lfw.py           valutazione completa di ArcFace su LFW
  tools/
    timestamp_test.html   test di precisione dei timestamp del browser
  tests/                  test con pytest (30 test)
  data/                   dati locali, esclusi da git (LFW, cache, risultati, raccolte)
```

Ogni file del codice ha commenti in italiano che uniscono teoria e implementazione. Per ogni funzione indicano cosa riceve, cosa restituisce, un esempio numerico, quali funzioni chiama e da chi è chiamata.

## Dati

Tutto ciò che sta in `data/` è escluso da git:

| Cosa | Dove |
|---|---|
| foto di LFW e `pairs.txt` | `data/lfw/lfw/`, `data/lfw/pairs.txt` |
| manifest di LFW | `data/lfw/manifest.csv` |
| cache degli embedding | `data/lfw/embeddings_arcface_center.npz` |
| risultati della valutazione | `data/lfw/results_arcface.json` |
| raccolte del gruppo (volto, keystroke, JSON dei timestamp) | sotto `data/` |

I dati biometrici, le foto e i JSON con il ritmo di digitazione delle persone **non devono finire nel repository**.

## Se serve un requirements.txt

Per ambienti senza uv (ad esempio Colab o Kaggle) si può generare dal lock, invece di scriverlo a mano:

```bash
uv export --format requirements-txt --no-hashes > requirements.txt
```

Va rigenerato ogni volta che si aggiunge una dipendenza.
