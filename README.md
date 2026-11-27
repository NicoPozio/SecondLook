# SecondLook

Login multibiometrico a step-up: il sistema riconosce l'utente dal volto e chiede un secondo tratto (il ritmo di digitazione di una frase) solo quando la risposta del volto non è affidabile. L'affidabilità è misurata con l'SRR (System Response Reliability) su embedding facciali ArcFace.

Progetto d'esame di Biometric Systems, Sapienza Università di Roma.

## Stato del progetto

| Fase | Contenuto | Stato |
|---|---|---|
| 0 | test di precisione dei timestamp del browser | fatto |
| 1 | embedding del volto con insightface (`face/embed.py`) | fatto |
| 2 | metriche, caricamento di LFW, matrici delle distanze, valutazione di ArcFace e LBP su LFW | fatto: ArcFace 99,83 %, LBP 66,00 % di accuratezza su LFW |
| 5 | stimatori di affidabilità: SRR (`reliability/srr.py`) valutato su LFW | fatta (27 novembre): SRR, margine, Weibull, qualità (SP, SI, SY), incertezza appresa (MagFace), PRR/AUROC, Viola-Jones, impostore con foto ottima |
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

Devono passare tutti i 79 test. I test sul modello scaricano `buffalo_l` se non è già presente, quindi la prima esecuzione richiede qualche minuto.

| File di test | Cosa controlla |
|---|---|
| `tests/test_embed.py` | distanza coseno, rilevamento dei volti, stesso volto più vicino degli altri, errore se non c'è nessun volto |
| `tests/test_face_selection.py` | scelta del volto quando in una foto ce n'è più di uno (`largest` e `center`); il ritaglio per LBP è identico a quello che riceve ArcFace |
| `tests/test_lbp.py` | codici LBP calcolati a mano, pattern uniformi, istogrammi a griglia, distanza chi-quadro |
| `tests/test_srr.py` | relative distance, density ratio, valore critico e SRR sugli esempi a mano degli appunti e della guida |
| `tests/test_margin.py` | margine top-1/top-2 (casi A e B della guida), una distanza per persona, distanza dalla soglia |
| `tests/test_quality.py` | indici di qualità SP, SI, SY su esempi a mano; stesso ritaglio di ArcFace e LBP |
| `tests/test_metarecognition.py` | meta-recognition di Weibull: casi A e B, uso della sola coda, una distanza per persona |
| `tests/test_reliability_metrics.py` | curva errore contro rifiuto, PRR e AUROC sugli esempi a mano, pareggi |
| `tests/test_detect.py` | IoU e ritaglio quadrato a mano, detection rate, Viola-Jones sulla foto di gruppo, `embed_crop` uguale a `embed` (salta se OpenCV non ha Viola-Jones) |
| `tests/test_magface.py` | preparazione dell'input a mano, messaggio se manca il modello, foto sfocata più corta di quella nitida (salta se manca il modello) |
| `tests/test_hard_impostors.py` | soglie della "foto ottima" sul tuning e quota lasciata passare, su esempi a mano |
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

## Affidabilità della risposta: SRR, margine, Weibull e qualità su LFW

`src/secondlook/reliability/srr.py` calcola l'SRR (System Response Reliability) di una risposta dell'identificazione, a partire dalle distanze del probe da **ogni persona** della gallery (se una persona ha più template si tiene il più vicino):

- relative distance φ = (d₂ − d₁) / d_N, density ratio φ = 1 − |N_b| / |N| (nuvola < 2·d₁, primo escluso);
- valore critico φ̄, stimato dove si bilanciano risposte sbagliate giudicate affidabili e risposte giuste giudicate inaffidabili;
- SRR = (φ − φ̄) / S, in [−1, 1]; nel login, SRR basso fa scattare lo step-up.

`src/secondlook/reliability/margin.py` calcola i due concorrenti più semplici, che guardano solo i primi posti: il **margine top-1/top-2** d₂ − d₁ e la **distanza dalla soglia** t − d₁ (che ordina i probe come −d₁, qualunque sia t).

`src/secondlook/face/quality.py` calcola gli **indici di qualità** degli appunti, sulla foto e prima del confronto: **SP** (posa: roll, yaw e pitch dai 5 punti del detector e dal mento dei 106 punti di `2d106det.onnx`), **SI** (illuminazione: uniformità del grigio medio di 5 regioni del ritaglio allineato) e **SY** (simmetria di 48 coppie di punti). Gli appunti non danno pesi, normalizzazioni, regioni né la funzione F: le scelte fatte (α = β = γ = 1/3, F(x) = x/127,5, r₀ stimato sul tuning, SY come media) sono descritte nel file. Le misure si calcolano con `FaceEmbedder(landmarks=True).quality_inputs()`, con lo stesso rilevamento e lo stesso ritaglio di ArcFace e LBP, e si salvano in `data/lfw/quality_center.npz` (circa 50 minuti su CPU per tutto LFW).

`src/secondlook/reliability/metarecognition.py` calcola la **meta-recognition di Weibull** (Scheirer et al., 2011): adatta una Weibull (`scipy.stats.weibull_min`) alla coda delle 10 distanze dopo il primo, cioè ai candidati sbagliati, e misura quanto è improbabile d₁ secondo quella distribuzione, con il punteggio −log₁₀ P(non-corrispondente ≤ d₁).

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
| ArcFace Weibull | 0,996 | 0,9980 | 0,92 | 0,9983 |
| ArcFace qualità SP / SI / SY | 0,00 / 0,02 / −0,00 | 0,50 / 0,51 / 0,50 | 0,44 / 0,60 / 0,63 | 0,50 / 0,51 / 0,50 |
| ArcFace incertezza appresa (MagFace) | 0,13 | 0,56 | 0,40 | 0,57 |
| LBP SRR relative | 0,532 | 0,766 | 0,77 | 0,77 |
| LBP SRR density | 0,364 | 0,682 | 0,68 | 0,69 |
| LBP margine | 0,528 | 0,764 | 0,76 | 0,77 |
| LBP distanza dalla soglia | 0,404 | 0,702 | 0,69 | 0,72 |
| LBP Weibull | 0,494 | 0,747 | 0,75 | 0,75 |
| LBP qualità SP / SI / SY | 0,06 / 0,08 / 0,18 | 0,53 / 0,54 / 0,59 | 0,53 / 0,54 / 0,59 | 0,53 / 0,54 / 0,58 |
| LBP incertezza appresa (MagFace) | 0,06 | 0,53 | 0,51 | 0,57 |

Con ArcFace i criteri sono quasi perfetti e praticamente pari (nella metà test gli scambi di persona sono solo 9, quindi il loro AUROC è poco stabile). Con LBP la relative distance e il margine sono pari e i migliori; Weibull è subito dietro, il density ratio e la distanza dalla soglia sono nettamente peggiori. Gli indici di qualità della foto non prevedono gli errori (PRR vicino a 0, cioè come il caso): non distinguono uno sconosciuto da un registrato, perché le loro foto sono belle uguali. Solo la simmetria aiuta un po' con LBP (PRR 0,18), che soffre i volti girati.

Con ArcFace e distanze coseno, per gli sconosciuti la nuvola < 2·d₁ contiene quasi sempre tutta la gallery: il density ratio diventa quasi un sì/no e φ̄ vale 0.

## Incertezza appresa: MagFace

`src/secondlook/face/magface.py` calcola il quarto candidato della Domanda 1, l'**incertezza appresa**: la lunghezza del vettore di **MagFace** (Meng e colleghi, CVPR 2021), una rete addestrata in modo che la lunghezza cresca con la qualità del volto (tra 10 e 110). Usa i pesi ufficiali iResNet100 / MS1MV2 e lo stesso ritaglio allineato di ArcFace, LBP e qualità; come SP, SI e SY guarda solo la foto.

Il modello non è nel repository. Per prepararlo, una volta sola:

1. scaricare `magface_epoch_00025.pth` (iResNet100, MS1MV2) dal README di [MagFace](https://github.com/IrvingMeng/MagFace) e metterlo in `data/models/`;
2. convertirlo in ONNX: `python tools/magface_to_onnx.py --repo <cartella del codice di MagFace>`. Serve PyTorch solo per questo passo (`pip install torch onnx`, fuori dal progetto); lo script controlla che ONNX e PyTorch diano gli stessi vettori. Il risultato è `data/models/magface_iresnet100.onnx`, che il progetto esegue con `onnxruntime`.

```bash
uv run python scripts/eval_srr.py     # se c'è data/lfw/magface_center.npz, MagFace entra tra i criteri
```

Le lunghezze di tutto LFW si calcolano con `embed_rows(..., method="magface")` (circa 3 ore su CPU) e si salvano in `data/lfw/magface_center.npz`. Risultati sulla metà test: con ArcFace PRR 0,129 e AUROC 0,565, il migliore dei criteri sulla foto (SP, SI, SY circa 0) ma lontanissimo da quelli sulle distanze (0,997); con LBP PRR 0,064. Anche una qualità appresa non sa se la persona è registrata. Il piccolo segnale sugli sconosciuti è probabilmente un effetto di LFW: le persone con una sola foto hanno vettori un po' più corti.

## Impostore molto simile, foto ottima

`scripts/eval_hard_impostors.py` mostra sul caso più insidioso che una foto bella non garantisce una decisione affidabile. Usa gli stessi punteggi e la stessa divisione tuning/test di `eval_srr.py` (funzione `prepare`). Una foto è **ottima** se SP, SI e SY sono tutti nel quarto migliore (soglie stimate sul tuning); tra le risposte sbagliate del test con foto ottima si conta quante ne lascia passare ogni criterio, al suo valore critico.

```bash
uv run python scripts/eval_hard_impostors.py                 # ArcFace
uv run python scripts/eval_hard_impostors.py --method lbp
```

| | sbagliate con foto ottima | lasciate passare dalla qualità | dalla sola soglia | dall'SRR (relative distance) | step-up inutili SRR (giuste, foto ottima) |
|---|---|---|---|---|---|
| ArcFace | 85 (tutti sconosciuti) | 100 % | 0 % | 0 % | 0,6 % |
| LBP | 249 (85 sconosciuti, 164 scambi) | 100 % | 71,1 % | 24,5 % | 43,8 % |

Un indice calcolato solo sulla foto non può sapere se la persona è registrata: davanti a una foto perfetta di un impostore dice sempre "affidabile". Con ArcFace l'SRR ferma tutti questi impostori, ma li fermerebbe anche la soglia (il più simile ha d₁ 0,664, sopra t = 0,61). Con LBP una foto ottima rende la distanza più piccola e l'impostore più pericoloso per la soglia; l'SRR ne ferma tre su quattro, al prezzo di molti step-up inutili.

## Viola-Jones contro SCRFD: quanto conta il detector

`src/secondlook/face/detect.py` aggiunge **Viola-Jones** (cascata di Haar di OpenCV, `haarcascade_frontalface_default.xml`) per rispondere alla domanda della pipeline: la qualità del detector influenza il riconoscimento e l'affidabilità? Viola-Jones dà solo il riquadro, senza i 5 punti, quindi non si può allineare. Per separare l'effetto del detector da quello dell'allineamento si confrontano tre varianti, tutte con ArcFace: `arcface` (SCRFD con allineamento), `arcface_noalign` (riquadro SCRFD senza allineamento, il controllo) e `arcface_vj` (riquadro Viola-Jones senza allineamento). Le due varianti senza allineamento usano lo stesso ritaglio quadrato (`square_crop`, lato 1,2 · √area del riquadro, 112 × 112).

```bash
uv run python scripts/eval_detectors.py                      # ~3 ore su CPU: riquadri, detection rate, cache
uv run python scripts/eval_lfw.py --method arcface_vj        # poi come gli altri metodi (anche arcface_noalign)
uv run python scripts/eval_srr.py --method arcface_vj
```

**OpenCV 5.** In OpenCV 5 `cv2.CascadeClassifier` è nel pacchetto `opencv-contrib-python` e il file `.xml` non è più incluso. `haar_cascade()` scarica il file da GitHub in `data/models/` e ne controlla l'md5. Per questo `pyproject.toml` fissa `opencv-python>=4.9,<5` (dal 27 novembre: OpenCV 4.14, che ha ancora `CascadeClassifier`). Con OpenCV 5 senza il pacchetto contrib i test di Viola-Jones vengono saltati.

Detection rate su LFW (13.233 foto, riferimento = volto di SCRFD): SCRFD 99,86 %, Viola-Jones 99,27 %, di cui 98,53 % sullo stesso volto di SCRFD (IoU ≥ 0,5, IoU mediano 0,72). Viola-Jones perde soprattutto i volti girati.

Risultati sulle **stesse 13.038 foto** (quelle in cui i due detector trovano lo stesso volto), PRR sulla metà test:

| variante | accuratezza 10-fold | EER | RR | DIR @ FA 1 % | PRR relative | PRR margine | PRR soglia | PRR Weibull |
|---|---|---|---|---|---|---|---|---|
| SCRFD + allineamento | 99,85 % | 0,28 % | 99,79 % | 99,66 % | 0,994 | 0,994 | 0,993 | 0,993 |
| SCRFD senza allineamento | 99,26 % | 0,62 % | 98,19 % | 95,29 % | 0,983 | 0,983 | 0,988 | 0,970 |
| Viola-Jones senza allineamento | 99,55 % | 0,41 % | 99,40 % | 98,70 % | 0,991 | 0,991 | 0,993 | 0,988 |

Conta soprattutto l'allineamento: senza, gli errori di identificazione sono circa 9 volte di più. A parità di ritaglio Viola-Jones va meglio di SCRFD, perché il suo riquadro tiene gli occhi più vicini e più stabili rispetto al template di ArcFace (scarto mediano 5,9 px contro 9,4 px). I criteri di affidabilità restano buoni in tutte e tre le varianti e seguono lo stesso ordine del riconoscimento.

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
    face/quality.py       indici di qualità SP (posa), SI (illuminazione), SY (simmetria)
    face/magface.py       incertezza appresa: lunghezza del vettore MagFace (modello ONNX)
    face/detect.py        Viola-Jones, IoU, ritaglio quadrato senza allineamento, confronto con SCRFD
    eval/datasets.py      manifest nel formato comune, download di LFW, coppie di pairs.txt, divisione gallery/probe
    eval/matrices.py      cache degli embedding (.npz) e matrice delle distanze probe x gallery
    eval/metrics.py       FAR, FRR, ROC, DET, EER, d', CMS/CMC, DIR e false alarm open-set; curva errore-rifiuto, PRR, AUROC
    reliability/srr.py    SRR: relative distance, density ratio, valore critico, normalizzazione
    reliability/margin.py margine top-1/top-2 e distanza dalla soglia
    reliability/metarecognition.py  meta-recognition di Weibull sulla coda delle distanze
    keystroke/            feature di digitazione e matcher (da fare)
    fusion/               normalizzazione, step-up, aggiornamento template (da fare)
    app/                  app web di acquisizione e demo (da fare)
  scripts/
    compare_faces.py      distanza tra due foto
    eval_lfw.py           valutazione completa su LFW, con ArcFace o LBP (--method)
    eval_detectors.py     Viola-Jones contro SCRFD: detection rate e cache delle varianti senza allineamento
    eval_hard_impostors.py  caso "impostore molto simile, foto ottima": qualità contro SRR
    eval_srr.py           valutazione di SRR, margine, Weibull e qualità su LFW (valore critico sul tuning; errori, PRR e AUROC sul test)
  tools/
    timestamp_test.html   test di precisione dei timestamp del browser
    magface_to_onnx.py    conversione una tantum dei pesi di MagFace da PyTorch a ONNX
  tests/                  test con pytest (89 test)
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
| cache delle misure di qualità | `data/lfw/quality_center.npz` |
| confronto tra detector | `data/lfw/detectors_center.npz`, `data/lfw/embeddings_arcface_noalign_center.npz`, `data/lfw/embeddings_arcface_vj_center.npz`, `data/lfw/results_detectors.json` |
| impostori con foto ottima | `data/lfw/results_hard_impostors_<metodo>.json` |
| lunghezze MagFace | `data/lfw/magface_center.npz` |
| modello MagFace (pesi originali e ONNX) | `data/models/magface_epoch_00025.pth`, `data/models/magface_iresnet100.onnx` |
| file della cascata di Viola-Jones | `data/models/haarcascade_frontalface_default.xml` (scaricato da solo) |
| risultati della valutazione | `data/lfw/results_arcface.json`, `data/lfw/results_lbp.json`, `data/lfw/results_srr_<metodo>.json` |
| raccolte del gruppo (volto, keystroke, JSON dei timestamp) | sotto `data/` |

I dati biometrici, le foto e i JSON con il ritmo di digitazione delle persone **non devono finire nel repository**.

## Se serve un requirements.txt

Per ambienti senza uv (ad esempio Colab o Kaggle) si può generare dal lock, invece di scriverlo a mano:

```bash
uv export --format requirements-txt --no-hashes > requirements.txt
```

Va rigenerato ogni volta che si aggiunge una dipendenza.
