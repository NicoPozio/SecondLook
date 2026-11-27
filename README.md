# SecondLook

Login multibiometrico a step-up: il sistema riconosce l'utente dal volto e chiede un secondo tratto (il ritmo di digitazione di una frase) solo quando la risposta del volto non è affidabile. L'affidabilità è misurata con l'SRR (System Response Reliability) su embedding facciali ArcFace.

Progetto d'esame di Biometric Systems, Sapienza Università di Roma.

## Stato del progetto

| Fase | Contenuto | Stato |
|---|---|---|
| 0 | test di precisione dei timestamp del browser | fatto |
| 1 | embedding del volto con insightface (`face/embed.py`) | fatto |
| 2 | metriche, caricamento di LFW, matrici delle distanze, valutazione di ArcFace e LBP su LFW | fatto: ArcFace 99,83 %, LBP 66,00 % di accuratezza su LFW |
| 5 | stimatori di affidabilità: SRR (`reliability/srr.py`) valutato su LFW | in corso: SRR, margine, PRR/AUROC fatti; Weibull e qualità da fare |
| 3, 4, 6-8 | app di acquisizione, raccolta, keystroke, fusione step-up, demo | in corso (altri binari) / da fare |

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

Devono passare tutti i 67 test. I test sul modello scaricano `buffalo_l` se non è già presente, quindi la prima esecuzione richiede qualche minuto.

| File di test | Cosa controlla |
|---|---|
| `tests/test_embed.py` | distanza coseno, rilevamento dei volti, stesso volto più vicino degli altri, errore se non c'è nessun volto |
| `tests/test_face_selection.py` | scelta del volto quando in una foto ce n'è più di uno (`largest` e `center`); il ritaglio per LBP è identico a quello che riceve ArcFace |
| `tests/test_lbp.py` | codici LBP calcolati a mano, pattern uniformi, istogrammi a griglia, distanza chi-quadro |
| `tests/test_srr.py` | relative distance, density ratio, valore critico e SRR sugli esempi a mano degli appunti e della guida |
| `tests/test_margin.py` | margine top-1/top-2 (casi A e B della guida), una distanza per persona, distanza dalla soglia |
| `tests/test_reliability_metrics.py` | curva errore contro rifiuto, PRR e AUROC sugli esempi a mano, pareggi |
| `tests/test_eval_srr.py` | divisione delle persone tuning/test, i quattro criteri, conteggio degli errori, PRR e AUROC per tipo di errore |
| `tests/test_ten_fold.py` | accuratezza a 10 gruppi di `eval_lfw.py`: conto a mano, soglia mai scelta sul gruppo d'esame |
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

`FaceEmbedder.crop(img, select=...)` restituisce invece il **ritaglio allineato 112 × 112** del volto scelto, che è l'ingresso di LBP. Rilevamento e scelta del volto sono nella stessa funzione usata da `embed()` (`_detect_and_choose`), e il ritaglio è fatto con la stessa funzione di insightface (`norm_crop`) che prepara il volto per ArcFace. LBP e ArcFace ricevono quindi esattamente la stessa immagine; il test `test_crop_is_what_arcface_sees` lo verifica.

## LBP: la baseline classica

`src/secondlook/face/lbp.py` implementa l'LBP a griglia del corso, senza reti neurali:

1. per ogni pixel, gli 8 vicini del quadrato 3 × 3 diventano 1 se sono ≥ del centro e 0 altrimenti: il codice LBP è un numero da 0 a 255;
2. i 58 codici "uniformi" (al massimo 2 passaggi 0/1 in tondo) hanno ognuno una casella, tutti gli altri una casella in comune: 59 caselle;
3. il volto si divide in una griglia 7 × 7 e per ogni cella si calcola l'istogramma dei codici; i 49 istogrammi concatenati danno un vettore di 2.891 numeri;
4. due vettori si confrontano con la distanza chi-quadro, Σ (a − b)² / (a + b), che va da 0 a 98 (non da 0 a 2 come il coseno).

## Valutazione di ArcFace e LBP su LFW

```bash
uv run python scripts/eval_lfw.py --quick                 # ArcFace, prova veloce: 600 coppie, 100 persone
uv run python scripts/eval_lfw.py                         # ArcFace, LFW completo: i numeri da riportare
uv run python scripts/eval_lfw.py --method lbp --quick    # LBP, prova veloce
uv run python scripts/eval_lfw.py --method lbp            # LBP, LFW completo
```

Con `--method` cambiano solo il descrittore, la distanza (coseno per ArcFace, chi-quadro per LBP), la cache e il file dei risultati. Rilevamento, allineamento, coppie, divisione gallery/probe e metriche sono gli stessi.

Lo script scarica LFW in `data/lfw` se non c'è (foto e `pairs.txt`), crea il manifest e calcola i vettori delle foto necessarie. I vettori vengono salvati in una cache per metodo (`data/lfw/embeddings_arcface_center.npz` e `data/lfw/embeddings_lbp_center.npz`): se lo script si interrompe, al riavvio riparte da dove era arrivato, e le esecuzioni successive durano pochi minuti.

Calcolare i vettori di tutto LFW (13.233 foto) su CPU richiede circa 2 ore per ArcFace e circa 1 ora per LBP (in LBP la parte lenta è solo il rilevamento del volto), a seconda del computer. Un portatile a batteria o in risparmio energetico può essere anche 10-20 volte più lento.

Lo script fa tre valutazioni:

| Valutazione | Dati usati | Cosa misura |
|---|---|---|
| **Verifica 1:1** | le 6.000 coppie ufficiali di `pairs.txt` (10 gruppi) | EER, d′, distanze medie, accuratezza con il protocollo a 10 gruppi |
| **Identificazione closed-set** | tutto LFW: prima foto di ogni persona con almeno 2 foto in gallery, le altre come probe | Recognition Rate, CMS(5), CMS(10) |
| **Identificazione open-set** | come sopra, più le persone con una sola foto come sconosciuti | DIR al rango 1 con false alarm all'1 % e al 10 % |

Con `--quick` si usano solo il primo gruppo di coppie, le prime 100 persone e 100 sconosciuti, e l'accuratezza a 10 gruppi non viene calcolata. I risultati vengono salvati in `data/lfw/results_<metodo>.json` (oppure `results_<metodo>_quick.json`).

### Risultati (LFW completo, 24 novembre 2026)

Le due colonne usano le stesse foto, gli stessi volti rilevati (18 foto su 13.233 senza volto, 17 coppie escluse dalla verifica) e gli stessi ritagli allineati.

| | ArcFace (coseno) | LBP 7 × 7, uniformi (chi-quadro) |
|---|---|---|
| **Accuratezza di verifica (10 gruppi)** | **99,83 % (± 0,20)** | **66,00 % (± 1,48)** |
| EER | 0,30 % (soglia 0,818) | 34,54 % (soglia 30,968) |
| Distanza media genuini / impostori | 0,327 / 0,996 | 29,00 / 33,05 |
| d′ | 7,97 | 0,80 |
| Closed-set (1.677 persone, 7.467 probe): RR | 99,77 % | 7,67 % |
| CMS(5) / CMS(10) | 99,84 % / 99,87 % | 12,82 % / 16,37 % |
| Open-set (4.067 sconosciuti): DIR con false alarm 0,98 % | 99,64 % (soglia 0,675) | 1,11 % (soglia 18,35) |
| DIR con false alarm 9,98 % | 99,72 % (soglia 0,733) | 2,76 % (soglia 20,20) |

L'accuratezza di ArcFace coincide con quella dichiarata da insightface per lo stesso modello (`buffalo_l`, 99,83 %). LBP è nettamente peggiore: su LFW le foto hanno pose, luci ed espressioni molto diverse, e le feature scelte a mano ne soffrono molto più di quelle apprese da una rete. EER, CMS e DIR non hanno un valore ufficiale di confronto su LFW: la divisione gallery/probe è nostra.

## Affidabilità della risposta: SRR e margine su LFW

`src/secondlook/reliability/srr.py` calcola l'SRR (System Response Reliability) di una risposta dell'identificazione, a partire dalle distanze del probe da **ogni persona** della gallery (se una persona ha più template si tiene il più vicino):

- relative distance φ = (d₂ − d₁) / d_N, density ratio φ = 1 − |N_b| / |N| (nuvola < 2·d₁, primo escluso);
- valore critico φ̄, stimato dove si bilanciano risposte sbagliate giudicate affidabili e risposte giuste giudicate inaffidabili;
- SRR = (φ − φ̄) / S, in [−1, 1]; nel login, SRR basso fa scattare lo step-up.

`src/secondlook/reliability/margin.py` calcola i due concorrenti più semplici, che guardano solo i primi posti: il **margine top-1/top-2** d₂ − d₁ e la **distanza dalla soglia** t − d₁ (che ordina i probe come −d₁, qualunque sia t).

```bash
uv run python scripts/eval_srr.py                 # ArcFace (serve la cache di eval_lfw.py)
uv run python scripts/eval_srr.py --method lbp    # LBP
```

Lo script usa le matrici dell'identificazione open-set di `eval_lfw.py`. Una risposta è giusta se il probe è registrato e il primo candidato è la sua persona; è sbagliata se il primo candidato è un altro o se il probe è di uno sconosciuto. Le persone vengono divise a metà (seme fisso): il valore critico di ogni criterio si stima sulla metà di tuning, gli errori si misurano sull'altra metà. Risultati in `data/lfw/results_srr_<metodo>.json`.

Risultati sulla metà di test (27 novembre 2026), regola "affidabile se criterio > valore critico, altrimenti step-up". Circa 1/3 dei login di test sono sconosciuti, quindi il tasso di step-up atteso è intorno al 33 %.

| | SRR relative | SRR density | margine top-1/top-2 | distanza dalla soglia |
|---|---|---|---|---|
| ArcFace: valore critico | φ̄ = 0,139 | φ̄ = 0,000 | 0,170 | t = 0,610 |
| ArcFace: risposte sbagliate lasciate passare | 0,25 % | 0,25 % | 0,25 % | 0,30 % |
| ArcFace: step-up su risposte giuste | 0,69 % | 0,71 % | 0,76 % | 0,45 % |
| LBP: risposte sbagliate lasciate passare | 30,5 % | 38,5 % | 30,0 % | 36,2 % |
| LBP: step-up su risposte giuste | 31,1 % | 35,8 % | 31,8 % | 34,4 % |

Ogni colonna è un solo punto di lavoro. Il confronto a **tutte le soglie** usa le funzioni di `eval/metrics.py`: la curva errore contro rifiuto (quanti errori restano se si mandano allo step-up le risposte meno affidabili), il **PRR** che la riassume (1 = perfetto, 0 = come il caso) e l'**AUROC** (probabilità che una risposta giusta abbia un valore più alto di una sbagliata), calcolato anche per tipo di errore: scambio di persona e sconosciuto.

| | PRR | AUROC | AUROC scambi | AUROC sconosciuti |
|---|---|---|---|---|
| ArcFace SRR relative | 0,997 | 0,9986 | 0,90 | 0,9990 |
| ArcFace SRR density | 0,991 | 0,9954 | 0,91 | 0,9958 |
| ArcFace margine | 0,997 | 0,9986 | 0,91 | 0,9990 |
| ArcFace distanza dalla soglia | 0,997 | 0,9985 | 0,90 | 0,9990 |
| LBP SRR relative | 0,532 | 0,766 | 0,77 | 0,77 |
| LBP SRR density | 0,364 | 0,682 | 0,68 | 0,69 |
| LBP margine | 0,528 | 0,764 | 0,76 | 0,77 |
| LBP distanza dalla soglia | 0,404 | 0,702 | 0,69 | 0,72 |

Con ArcFace i criteri sono quasi perfetti e praticamente pari (nella metà test gli scambi di persona sono solo 9, quindi il loro AUROC è poco stabile). Con LBP la relative distance e il margine sono pari e nettamente migliori del density ratio e della distanza dalla soglia.

Con ArcFace e distanze coseno, per gli sconosciuti la nuvola < 2·d₁ contiene quasi sempre tutta la gallery: il density ratio diventa quasi un sì/no e φ̄ vale 0.

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
    face/embed.py         rilevamento (SCRFD), scelta del volto, allineamento, embedding ArcFace e ritaglio per LBP
    face/lbp.py           LBP a griglia (uniformi, 7 x 7) e distanza chi-quadro
    eval/datasets.py      manifest nel formato comune, download di LFW, coppie di pairs.txt, divisione gallery/probe
    eval/matrices.py      cache degli embedding (.npz) e matrice delle distanze probe x gallery
    eval/metrics.py       FAR, FRR, ROC, DET, EER, d', CMS/CMC, DIR e false alarm open-set; curva errore-rifiuto, PRR, AUROC
    reliability/srr.py    SRR: relative distance, density ratio, valore critico, normalizzazione
    reliability/margin.py margine top-1/top-2 e distanza dalla soglia
    reliability/          altri stimatori: Weibull (da fare)
    keystroke/            feature di digitazione e matcher (da fare)
    fusion/               normalizzazione, step-up, aggiornamento template (da fare)
    app/                  app web di acquisizione e demo (da fare)
  scripts/
    compare_faces.py      distanza tra due foto
    eval_lfw.py           valutazione completa su LFW, con ArcFace o LBP (--method)
    eval_srr.py           valutazione di SRR e margine su LFW (valore critico sul tuning; errori, PRR e AUROC sul test)
  tools/
    timestamp_test.html   test di precisione dei timestamp del browser
  tests/                  test con pytest (67 test)
  data/                   dati locali, esclusi da git (LFW, cache, risultati, raccolte)
```

Ogni file del codice ha commenti in italiano che uniscono teoria e implementazione. Per ogni funzione indicano cosa riceve, cosa restituisce, un esempio numerico, quali funzioni chiama e da chi è chiamata.

## Dati

Tutto ciò che sta in `data/` è escluso da git:

| Cosa | Dove |
|---|---|
| foto di LFW e `pairs.txt` | `data/lfw/lfw/`, `data/lfw/pairs.txt` |
| manifest di LFW | `data/lfw/manifest.csv` |
| cache dei vettori | `data/lfw/embeddings_arcface_center.npz`, `data/lfw/embeddings_lbp_center.npz` |
| risultati della valutazione | `data/lfw/results_arcface.json`, `data/lfw/results_lbp.json`, `data/lfw/results_srr_<metodo>.json` |
| raccolte del gruppo (volto, keystroke, JSON dei timestamp) | sotto `data/` |

I dati biometrici, le foto e i JSON con il ritmo di digitazione delle persone **non devono finire nel repository**.

## Se serve un requirements.txt

Per ambienti senza uv (ad esempio Colab o Kaggle) si può generare dal lock, invece di scriverlo a mano:

```bash
uv export --format requirements-txt --no-hashes > requirements.txt
```

Va rigenerato ogni volta che si aggiunge una dipendenza.
