# r8: kleinhirn-WASM gegen ORT-wasm

Stand 2026-10-08, Commit b6cd38b. Quelle `bench/results/r8-table.json`, erzeugt von `KBENCH_TAG=r8 node bench/run-kbench.mjs wasm-summary`. Median in ms je Aufruf; kleinhirn mit 8 Thread(s), ORT in der Einstellung aus `data/kbench/ort-best.json` (alle Kerne) und mit `numThreads` 1; Faktor ORT durch kleinhirn; Speicher-Spitze in MiB nur Chromium.

Gate R8: Chromium kleinhirn vor ORT best in 18 von 18 Zellen; Faktor ORT best durch kleinhirn im geometrischen Mittel WebKit 1,33, Safari 1,48, Firefox 4,14 (berichtet), Chromium 3,28. Erfüllt.

## chromium

| Modell | Länge | kleinhirn | p95 | MiB | ORT best | ORT 1 Thread | Faktor best | Faktor 1 Thread | Parität kleinhirn |
|---|---|---|---|---|---|---|---|---|---|
| MiniLM | L128 voll | 8,39 | 8,55 |  | 33,11 | fehlt: nicht gelaufen | 3,94 |  | ok |
| MiniLM | L512 voll | 46,59 | 47,83 |  | 130,84 | fehlt: nicht gelaufen | 2,81 |  | ok |
| MiniLM | echte Länge | 2,76 | 5,37 |  | 10,14 | fehlt: nicht gelaufen | 3,68 |  | ok |
| RoBERTa-base | L128 voll | 54,49 | 55,56 |  | 196,18 | fehlt: nicht gelaufen | 3,60 |  | ok |
| RoBERTa-base | L512 voll | 245,85 | 249,80 |  | 649,14 | fehlt: nicht gelaufen | 2,64 |  | ok |
| RoBERTa-base | echte Länge | 18,15 | 248,94 |  | 62,83 | fehlt: nicht gelaufen | 3,46 |  | ok |
| mMiniLM | L128 voll | 17,91 | 18,72 |  | 69,32 | fehlt: nicht gelaufen | 3,87 |  | ok |
| mMiniLM | L512 voll | 94,50 | 95,88 |  | 259,47 | fehlt: nicht gelaufen | 2,75 |  | ok |
| mMiniLM | echte Länge | 8,55 | 13,84 |  | 30,73 | fehlt: nicht gelaufen | 3,60 |  | ok |
| DistilBERT | L128 voll | 27,10 | 28,72 |  | 99,03 | fehlt: nicht gelaufen | 3,65 |  | ok |
| DistilBERT | L512 voll | 123,20 | 124,87 |  | 325,44 | fehlt: nicht gelaufen | 2,64 |  | ok |
| DistilBERT | echte Länge | 9,62 | 122,86 |  | 32,51 | fehlt: nicht gelaufen | 3,38 |  | ok |
| DeBERTa-v3-base | L128 voll | 63,05 | 70,19 |  | 221,54 | fehlt: nicht gelaufen | 3,51 |  | ok |
| DeBERTa-v3-base | L512 voll | 326,99 | 330,28 |  | 769,31 | fehlt: nicht gelaufen | 2,35 |  | ok |
| DeBERTa-v3-base | echte Länge | 19,03 | 325,66 |  | 75,73 | fehlt: nicht gelaufen | 3,98 |  | ok |
| granite | L128 voll | 22,24 | 23,36 |  | 82,23 | fehlt: nicht gelaufen | 3,70 |  | ok |
| granite | L512 voll | 107,82 | 110,13 |  | 295,03 | fehlt: nicht gelaufen | 2,74 |  | ok |
| granite | echte Länge | 6,48 | 11,66 |  | 22,05 | fehlt: nicht gelaufen | 3,40 |  | ok |

## webkit

| Modell | Länge | kleinhirn | p95 | MiB | ORT best | ORT 1 Thread | Faktor best | Faktor 1 Thread | Parität kleinhirn |
|---|---|---|---|---|---|---|---|---|---|
| MiniLM | L128 voll | 12,61 | 13,58 |  | 16,75 | fehlt: nicht gelaufen | 1,33 |  | ok |
| MiniLM | L512 voll | 59,72 | 70,04 |  | 62,57 | fehlt: nicht gelaufen | 1,05 |  | ok |
| MiniLM | echte Länge | 3,88 | 8,22 |  | 10,14 | fehlt: nicht gelaufen | 2,61 |  | ok |
| RoBERTa-base | L128 voll | 79,43 | 84,86 |  | 93,14 | fehlt: nicht gelaufen | 1,17 |  | ok |
| RoBERTa-base | L512 voll | 351,90 | 363,20 |  | 383,26 | fehlt: nicht gelaufen | 1,09 |  | ok |
| RoBERTa-base | echte Länge | 23,52 | 351,14 |  | 33,90 | fehlt: nicht gelaufen | 1,44 |  | ok |
| mMiniLM | L128 voll | 24,53 | 25,54 |  | 31,72 | fehlt: nicht gelaufen | 1,29 |  | ok |
| mMiniLM | L512 voll | 118,16 | 121,16 |  | 120,60 | fehlt: nicht gelaufen | 1,02 |  | ok |
| mMiniLM | echte Länge | 11,30 | 18,76 |  | 29,59 | fehlt: nicht gelaufen | 2,62 |  | ok |
| DistilBERT | L128 voll | 40,75 | 42,38 |  | 47,56 | fehlt: nicht gelaufen | 1,17 |  | ok |
| DistilBERT | L512 voll | 175,07 | 181,36 |  | 196,07 | fehlt: nicht gelaufen | 1,12 |  | ok |
| DistilBERT | echte Länge | 14,70 | 172,18 |  | 25,40 | fehlt: nicht gelaufen | 1,73 |  | ok |
| DeBERTa-v3-base | L128 voll | 89,08 | 102,78 |  | 105,27 | fehlt: nicht gelaufen | 1,18 |  | ok |
| DeBERTa-v3-base | L512 voll | 436,12 | 441,40 |  | 449,04 | fehlt: nicht gelaufen | 1,03 |  | ok |
| DeBERTa-v3-base | echte Länge | 22,96 | 434,34 |  | 40,58 | fehlt: nicht gelaufen | 1,77 |  | ok |
| granite | L128 voll | 30,37 | 35,66 |  | 37,10 | fehlt: nicht gelaufen | 1,22 |  | ok |
| granite | L512 voll | 144,55 | 150,32 |  | 143,16 | fehlt: nicht gelaufen | 0,99 |  | ok |
| granite | echte Länge | 10,42 | 20,24 |  | 13,32 | fehlt: nicht gelaufen | 1,28 |  | ok |

## firefox

| Modell | Länge | kleinhirn | p95 | MiB | ORT best | ORT 1 Thread | Faktor best | Faktor 1 Thread | Parität kleinhirn |
|---|---|---|---|---|---|---|---|---|---|
| MiniLM | L128 voll | fehlt: nicht gelaufen |  |  | fehlt: nicht gelaufen | fehlt: nicht gelaufen |  |  |  |
| MiniLM | L512 voll | fehlt: nicht gelaufen |  |  | fehlt: nicht gelaufen | fehlt: nicht gelaufen |  |  |  |
| MiniLM | echte Länge | fehlt: nicht gelaufen |  |  | fehlt: nicht gelaufen | fehlt: nicht gelaufen |  |  |  |
| RoBERTa-base | L128 voll | fehlt: nicht gelaufen |  |  | fehlt: nicht gelaufen | fehlt: nicht gelaufen |  |  |  |
| RoBERTa-base | L512 voll | fehlt: nicht gelaufen |  |  | fehlt: nicht gelaufen | fehlt: nicht gelaufen |  |  |  |
| RoBERTa-base | echte Länge | fehlt: nicht gelaufen |  |  | fehlt: nicht gelaufen | fehlt: nicht gelaufen |  |  |  |
| mMiniLM | L128 voll | fehlt: nicht gelaufen |  |  | fehlt: nicht gelaufen | fehlt: nicht gelaufen |  |  |  |
| mMiniLM | L512 voll | fehlt: nicht gelaufen |  |  | fehlt: nicht gelaufen | fehlt: nicht gelaufen |  |  |  |
| mMiniLM | echte Länge | fehlt: nicht gelaufen |  |  | fehlt: nicht gelaufen | fehlt: nicht gelaufen |  |  |  |
| DistilBERT | L128 voll | fehlt: nicht gelaufen |  |  | fehlt: nicht gelaufen | fehlt: nicht gelaufen |  |  |  |
| DistilBERT | L512 voll | fehlt: nicht gelaufen |  |  | fehlt: nicht gelaufen | fehlt: nicht gelaufen |  |  |  |
| DistilBERT | echte Länge | fehlt: nicht gelaufen |  |  | fehlt: nicht gelaufen | fehlt: nicht gelaufen |  |  |  |
| DeBERTa-v3-base | L128 voll | fehlt: nicht gelaufen |  |  | fehlt: nicht gelaufen | fehlt: nicht gelaufen |  |  |  |
| DeBERTa-v3-base | L512 voll | fehlt: nicht gelaufen |  |  | fehlt: nicht gelaufen | fehlt: nicht gelaufen |  |  |  |
| DeBERTa-v3-base | echte Länge | fehlt: nicht gelaufen |  |  | fehlt: nicht gelaufen | fehlt: nicht gelaufen |  |  |  |
| granite | L128 voll | fehlt: nicht gelaufen |  |  | fehlt: nicht gelaufen | fehlt: nicht gelaufen |  |  |  |
| granite | L512 voll | fehlt: nicht gelaufen |  |  | fehlt: nicht gelaufen | fehlt: nicht gelaufen |  |  |  |
| granite | echte Länge | fehlt: nicht gelaufen |  |  | fehlt: nicht gelaufen | fehlt: nicht gelaufen |  |  |  |

## safari

| Modell | Länge | kleinhirn | p95 | MiB | ORT best | ORT 1 Thread | Faktor best | Faktor 1 Thread | Parität kleinhirn |
|---|---|---|---|---|---|---|---|---|---|
| MiniLM | L128 voll | 15,63 | 21,16 |  | 44,38 | fehlt: nicht gelaufen | 2,84 |  | ok |
| MiniLM | L512 voll | 61,67 | 65,04 |  | 61,93 | fehlt: nicht gelaufen | 1,00 |  | ok |
| MiniLM | echte Länge | 4,20 | 7,88 |  | 10,09 | fehlt: nicht gelaufen | 2,40 |  | ok |
| RoBERTa-base | L128 voll | 78,48 | 85,28 |  | 92,50 | fehlt: nicht gelaufen | 1,18 |  | ok |
| RoBERTa-base | L512 voll | 344,35 | 350,58 |  | 372,55 | fehlt: nicht gelaufen | 1,08 |  | ok |
| RoBERTa-base | echte Länge | 23,39 | 341,22 |  | 32,02 | fehlt: nicht gelaufen | 1,37 |  | ok |
| mMiniLM | L128 voll | 23,08 | 34,46 |  | 50,95 | fehlt: nicht gelaufen | 2,21 |  | ok |
| mMiniLM | L512 voll | 116,58 | 121,86 |  | 116,20 | fehlt: nicht gelaufen | 1,00 |  | ok |
| mMiniLM | echte Länge | 11,04 | 17,72 |  | 28,71 | fehlt: nicht gelaufen | 2,60 |  | ok |
| DistilBERT | L128 voll | 40,07 | 42,14 |  | 48,47 | fehlt: nicht gelaufen | 1,21 |  | ok |
| DistilBERT | L512 voll | 176,17 | 232,12 |  | 195,44 | fehlt: nicht gelaufen | 1,11 |  | ok |
| DistilBERT | echte Länge | 12,18 | 171,88 |  | 21,88 | fehlt: nicht gelaufen | 1,80 |  | ok |
| DeBERTa-v3-base | L128 voll | 88,61 | 95,68 |  | 131,95 | fehlt: nicht gelaufen | 1,49 |  | ok |
| DeBERTa-v3-base | L512 voll | 436,57 | 441,92 |  | 564,69 | fehlt: nicht gelaufen | 1,29 |  | ok |
| DeBERTa-v3-base | echte Länge | 23,68 | 434,16 |  | 40,92 | fehlt: nicht gelaufen | 1,73 |  | ok |
| granite | L128 voll | 30,19 | 32,36 |  | 37,40 | fehlt: nicht gelaufen | 1,24 |  | ok |
| granite | L512 voll | 145,15 | 155,42 |  | 148,16 | fehlt: nicht gelaufen | 1,02 |  | ok |
| granite | echte Länge | 8,50 | 16,26 |  | 14,38 | fehlt: nicht gelaufen | 1,69 |  | ok |

## firefox-reg

| Modell | Länge | kleinhirn | p95 | MiB | ORT best | ORT 1 Thread | Faktor best | Faktor 1 Thread | Parität kleinhirn |
|---|---|---|---|---|---|---|---|---|---|
| MiniLM | L128 voll | 11,83 | 20,74 |  | 52,16 | fehlt: nicht gelaufen | 4,41 |  | ok |
| MiniLM | L512 voll | 66,84 | 69,28 |  | 187,48 | fehlt: nicht gelaufen | 2,80 |  | ok |
| MiniLM | echte Länge | 3,88 | 9,64 |  | 16,19 | fehlt: nicht gelaufen | 4,17 |  | ok |
| RoBERTa-base | L128 voll | 58,27 | 63,16 |  | 306,22 | fehlt: nicht gelaufen | 5,26 |  | ok |
| RoBERTa-base | L512 voll | 286,01 | 311,88 |  | 937,52 | fehlt: nicht gelaufen | 3,28 |  | ok |
| RoBERTa-base | echte Länge | 18,62 | 280,56 |  | 91,78 | fehlt: nicht gelaufen | 4,93 |  | ok |
| mMiniLM | L128 voll | 21,65 | 23,84 |  | 97,71 | fehlt: nicht gelaufen | 4,51 |  | ok |
| mMiniLM | L512 voll | 132,05 | 146,30 |  | 398,39 | fehlt: nicht gelaufen | 3,02 |  | ok |
| mMiniLM | echte Länge | 9,75 | 15,92 |  | 43,91 | fehlt: nicht gelaufen | 4,50 |  | ok |
| DistilBERT | L128 voll | 30,88 | 34,88 |  | 149,31 | fehlt: nicht gelaufen | 4,84 |  | ok |
| DistilBERT | L512 voll | 142,33 | 155,48 |  | 475,79 | fehlt: nicht gelaufen | 3,34 |  | ok |
| DistilBERT | echte Länge | 10,06 | 140,84 |  | 48,74 | fehlt: nicht gelaufen | 4,84 |  | ok |
| DeBERTa-v3-base | L128 voll | 70,51 | 74,54 |  | 348,30 | fehlt: nicht gelaufen | 4,94 |  | ok |
| DeBERTa-v3-base | L512 voll | 412,07 | 475,06 |  | 1141,87 | fehlt: nicht gelaufen | 2,77 |  | ok |
| DeBERTa-v3-base | echte Länge | 18,82 | 411,10 |  | 129,49 | fehlt: nicht gelaufen | 6,88 |  | ok |
| granite | L128 voll | 27,11 | 31,80 |  | 127,15 | fehlt: nicht gelaufen | 4,69 |  | ok |
| granite | L512 voll | 140,35 | 155,70 |  | 433,53 | fehlt: nicht gelaufen | 3,09 |  | ok |
| granite | echte Länge | 8,06 | 17,06 |  | 35,67 | fehlt: nicht gelaufen | 4,43 |  | ok |
