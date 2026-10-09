# r8b: kleinhirn-WASM gegen ORT-wasm

Stand 2026-10-09, Commit d6593f8. Quelle `bench/results/r8b-table.json`, erzeugt von `KBENCH_TAG=r8b node bench/run-kbench.mjs wasm-summary`. Median in ms je Aufruf; kleinhirn mit 8 Thread(s), ORT in der Einstellung aus `data/kbench/ort-best.json` (alle Kerne) und mit `numThreads` 1; Faktor ORT durch kleinhirn; Speicher-Spitze in MiB nur Chromium.

Gate R8: Chromium kleinhirn vor ORT best in 18 von 18 Zellen; Faktor ORT best durch kleinhirn im geometrischen Mittel WebKit 1,53, Safari 1,34, Firefox 4,37 (berichtet), Chromium 4,20. Erfüllt.

## chromium

| Modell | Länge | kleinhirn | p95 | MiB | ORT best | ORT 1 Thread | Faktor best | Faktor 1 Thread | Parität kleinhirn |
|---|---|---|---|---|---|---|---|---|---|
| MiniLM | L128 voll | 8,20 | 9,35 | 401 | 55,11 | fehlt: nicht gelaufen | 6,72 |  | ok |
| MiniLM | L512 voll | 32,44 | 37,82 | 257 | 137,07 | fehlt: nicht gelaufen | 4,23 |  | ok |
| MiniLM | echte Länge | 2,34 | 4,30 | 352 | 8,91 | fehlt: nicht gelaufen | 3,81 |  | ok |
| RoBERTa-base | L128 voll | 46,65 | 48,13 | 1319 | 209,03 | fehlt: nicht gelaufen | 4,48 |  | ok |
| RoBERTa-base | L512 voll | 201,36 | 205,08 | 1374 | 732,68 | fehlt: nicht gelaufen | 3,64 |  | ok |
| RoBERTa-base | echte Länge | 15,39 | 201,13 | 1373 | 60,29 | fehlt: nicht gelaufen | 3,92 |  | ok |
| mMiniLM | L128 voll | 13,34 | 13,51 | 1541 | 63,75 | fehlt: nicht gelaufen | 4,78 |  | ok |
| mMiniLM | L512 voll | 63,37 | 64,14 | 1436 | 271,53 | fehlt: nicht gelaufen | 4,29 |  | ok |
| mMiniLM | echte Länge | 6,73 | 10,32 | 1507 | 28,54 | fehlt: nicht gelaufen | 4,24 |  | ok |
| DistilBERT | L128 voll | 23,22 | 23,50 | 800 | 103,87 | fehlt: nicht gelaufen | 4,47 |  | ok |
| DistilBERT | L512 voll | 100,98 | 102,19 | 779 | 376,37 | fehlt: nicht gelaufen | 3,73 |  | ok |
| DistilBERT | echte Länge | 8,27 | 100,83 | 796 | 31,45 | fehlt: nicht gelaufen | 3,80 |  | ok |
| DeBERTa-v3-base | L128 voll | 51,08 | 56,87 | 2080 | 231,18 | fehlt: nicht gelaufen | 4,53 |  | ok |
| DeBERTa-v3-base | L512 voll | 238,53 | 239,92 | 2140 | 873,25 | fehlt: nicht gelaufen | 3,66 |  | ok |
| DeBERTa-v3-base | echte Länge | 15,84 | 237,59 | 2210 | 72,72 | fehlt: nicht gelaufen | 4,59 |  | ok |
| granite | L128 voll | 18,57 | 20,86 | 663 | 80,26 | fehlt: nicht gelaufen | 4,32 |  | ok |
| granite | L512 voll | 89,70 | 91,75 | 648 | 322,28 | fehlt: nicht gelaufen | 3,59 |  | ok |
| granite | echte Länge | 5,39 | 9,69 | 642 | 19,91 | fehlt: nicht gelaufen | 3,70 |  | ok |

## webkit

| Modell | Länge | kleinhirn | p95 | MiB | ORT best | ORT 1 Thread | Faktor best | Faktor 1 Thread | Parität kleinhirn |
|---|---|---|---|---|---|---|---|---|---|
| MiniLM | L128 voll | 10,55 | 11,40 |  | 36,55 | fehlt: nicht gelaufen | 3,46 |  | ok |
| MiniLM | L512 voll | 48,65 | 53,58 |  | 54,42 | fehlt: nicht gelaufen | 1,12 |  | ok |
| MiniLM | echte Länge | 3,37 | 7,02 |  | 9,65 | fehlt: nicht gelaufen | 2,86 |  | ok |
| RoBERTa-base | L128 voll | 72,62 | 76,56 |  | 81,22 | fehlt: nicht gelaufen | 1,12 |  | ok |
| RoBERTa-base | L512 voll | 314,90 | 336,40 |  | 341,94 | fehlt: nicht gelaufen | 1,09 |  | ok |
| RoBERTa-base | echte Länge | 21,14 | 315,40 |  | 31,44 | fehlt: nicht gelaufen | 1,49 |  | ok |
| mMiniLM | L128 voll | 21,08 | 23,12 |  | 55,86 | fehlt: nicht gelaufen | 2,65 |  | ok |
| mMiniLM | L512 voll | 96,29 | 98,10 |  | 105,49 | fehlt: nicht gelaufen | 1,10 |  | ok |
| mMiniLM | echte Länge | 9,99 | 16,18 |  | 25,41 | fehlt: nicht gelaufen | 2,54 |  | ok |
| DistilBERT | L128 voll | 36,67 | 37,64 |  | 41,25 | fehlt: nicht gelaufen | 1,12 |  | ok |
| DistilBERT | L512 voll | 157,71 | 159,20 |  | 178,82 | fehlt: nicht gelaufen | 1,13 |  | ok |
| DistilBERT | echte Länge | 11,57 | 157,52 |  | 18,27 | fehlt: nicht gelaufen | 1,58 |  | ok |
| DeBERTa-v3-base | L128 voll | 79,49 | 83,68 |  | 96,45 | fehlt: nicht gelaufen | 1,21 |  | ok |
| DeBERTa-v3-base | L512 voll | 370,60 | 373,12 |  | 419,01 | fehlt: nicht gelaufen | 1,13 |  | ok |
| DeBERTa-v3-base | echte Länge | 21,97 | 371,92 |  | 38,71 | fehlt: nicht gelaufen | 1,76 |  | ok |
| granite | L128 voll | 27,90 | 28,54 |  | 37,52 | fehlt: nicht gelaufen | 1,34 |  | ok |
| granite | L512 voll | 131,58 | 132,60 |  | 133,53 | fehlt: nicht gelaufen | 1,01 |  | ok |
| granite | echte Länge | 7,41 | 15,22 |  | 16,12 | fehlt: nicht gelaufen | 2,18 |  | ok |

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
| MiniLM | L128 voll | 22,03 | 121,58 |  | 33,77 | fehlt: nicht gelaufen | 1,53 |  | ok |
| MiniLM | L512 voll | 48,39 | 51,86 |  | 52,34 | fehlt: nicht gelaufen | 1,08 |  | ok |
| MiniLM | echte Länge | 3,24 | 6,94 |  | 10,18 | fehlt: nicht gelaufen | 3,14 |  | ok |
| RoBERTa-base | L128 voll | 73,40 | 100,68 |  | 79,69 | fehlt: nicht gelaufen | 1,09 |  | ok |
| RoBERTa-base | L512 voll | 314,04 | 323,12 |  | 338,44 | fehlt: nicht gelaufen | 1,08 |  | ok |
| RoBERTa-base | echte Länge | 22,01 | 313,42 |  | 28,71 | fehlt: nicht gelaufen | 1,30 |  | ok |
| mMiniLM | L128 voll | 25,64 | 32,16 |  | 26,57 | fehlt: nicht gelaufen | 1,04 |  | ok |
| mMiniLM | L512 voll | 106,61 | 121,58 |  | 104,19 | fehlt: nicht gelaufen | 0,98 |  | ok |
| mMiniLM | echte Länge | 10,05 | 23,32 |  | 25,14 | fehlt: nicht gelaufen | 2,50 |  | ok |
| DistilBERT | L128 voll | 36,54 | 41,04 |  | 40,09 | fehlt: nicht gelaufen | 1,10 |  | ok |
| DistilBERT | L512 voll | 157,64 | 163,08 |  | 174,78 | fehlt: nicht gelaufen | 1,11 |  | ok |
| DistilBERT | echte Länge | 11,72 | 157,64 |  | 18,35 | fehlt: nicht gelaufen | 1,57 |  | ok |
| DeBERTa-v3-base | L128 voll | 79,25 | 87,82 |  | 94,80 | fehlt: nicht gelaufen | 1,20 |  | ok |
| DeBERTa-v3-base | L512 voll | 369,02 | 371,58 |  | 410,29 | fehlt: nicht gelaufen | 1,11 |  | ok |
| DeBERTa-v3-base | echte Länge | 22,72 | 369,14 |  | 43,63 | fehlt: nicht gelaufen | 1,92 |  | ok |
| granite | L128 voll | 27,69 | 33,38 |  | 37,13 | fehlt: nicht gelaufen | 1,34 |  | ok |
| granite | L512 voll | 131,00 | 150,10 |  | 130,67 | fehlt: nicht gelaufen | 1,00 |  | ok |
| granite | echte Länge | 7,62 | 17,64 |  | 12,05 | fehlt: nicht gelaufen | 1,58 |  | ok |

## firefox-reg

| Modell | Länge | kleinhirn | p95 | MiB | ORT best | ORT 1 Thread | Faktor best | Faktor 1 Thread | Parität kleinhirn |
|---|---|---|---|---|---|---|---|---|---|
| MiniLM | L128 voll | 10,11 | 15,30 |  | 42,41 | fehlt: nicht gelaufen | 4,19 |  | ok |
| MiniLM | L512 voll | 35,93 | 39,76 |  | 150,33 | fehlt: nicht gelaufen | 4,18 |  | ok |
| MiniLM | echte Länge | 3,57 | 10,14 |  | 13,69 | fehlt: nicht gelaufen | 3,83 |  | ok |
| RoBERTa-base | L128 voll | 49,91 | 54,58 |  | 241,19 | fehlt: nicht gelaufen | 4,83 |  | ok |
| RoBERTa-base | L512 voll | 212,92 | 231,34 |  | 842,97 | fehlt: nicht gelaufen | 3,96 |  | ok |
| RoBERTa-base | echte Länge | 16,76 | 209,28 |  | 70,79 | fehlt: nicht gelaufen | 4,22 |  | ok |
| mMiniLM | L128 voll | 15,35 | 16,36 |  | 80,22 | fehlt: nicht gelaufen | 5,23 |  | ok |
| mMiniLM | L512 voll | 67,96 | 76,62 |  | 289,36 | fehlt: nicht gelaufen | 4,26 |  | ok |
| mMiniLM | echte Länge | 7,90 | 12,18 |  | 34,62 | fehlt: nicht gelaufen | 4,38 |  | ok |
| DistilBERT | L128 voll | 25,85 | 27,88 |  | 124,15 | fehlt: nicht gelaufen | 4,80 |  | ok |
| DistilBERT | L512 voll | 106,60 | 116,20 |  | 429,32 | fehlt: nicht gelaufen | 4,03 |  | ok |
| DistilBERT | echte Länge | 9,45 | 103,56 |  | 38,55 | fehlt: nicht gelaufen | 4,08 |  | ok |
| DeBERTa-v3-base | L128 voll | 52,95 | 58,20 |  | 269,81 | fehlt: nicht gelaufen | 5,10 |  | ok |
| DeBERTa-v3-base | L512 voll | 252,07 | 261,84 |  | 1015,83 | fehlt: nicht gelaufen | 4,03 |  | ok |
| DeBERTa-v3-base | echte Länge | 16,53 | 252,12 |  | 97,96 | fehlt: nicht gelaufen | 5,93 |  | ok |
| granite | L128 voll | 22,56 | 27,86 |  | 97,55 | fehlt: nicht gelaufen | 4,32 |  | ok |
| granite | L512 voll | 104,04 | 111,10 |  | 361,77 | fehlt: nicht gelaufen | 3,48 |  | ok |
| granite | echte Länge | 6,78 | 14,96 |  | 29,99 | fehlt: nicht gelaufen | 4,42 |  | ok |
