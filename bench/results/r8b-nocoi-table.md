# r8b-nocoi: kleinhirn-WASM gegen ORT-wasm

Stand 2026-10-09, Commit d6593f8. Quelle `bench/results/r8b-nocoi-table.json`, erzeugt von `KBENCH_TAG=r8b-nocoi node bench/run-kbench.mjs wasm-summary`. Median in ms je Aufruf; kleinhirn mit 1 Thread(s), ORT in der Einstellung aus `data/kbench/ort-best.json` (alle Kerne) und mit `numThreads` 1; Faktor ORT durch kleinhirn; Speicher-Spitze in MiB nur Chromium.

Ohne Cross-Origin-Isolation (KBENCH_NO_COI=1): kleinhirn und ORT können keine Threads starten und rechnen mit einem Thread. Faktor ORT durch kleinhirn im geometrischen Mittel Chromium 1,74, Safari 1,09.

## chromium

| Modell | Länge | kleinhirn | p95 | MiB | ORT best | ORT 1 Thread | Faktor best | Faktor 1 Thread | Parität kleinhirn |
|---|---|---|---|---|---|---|---|---|---|
| MiniLM | L128 voll | 43,10 | 44,00 |  | 77,50 | fehlt: nicht gelaufen | 1,80 |  | ok |
| MiniLM | L512 voll | 215,60 | 219,30 |  | 359,70 | fehlt: nicht gelaufen | 1,67 |  | ok |
| MiniLM | echte Länge | 10,10 | 28,90 |  | 17,75 | fehlt: nicht gelaufen | 1,76 |  | ok |
| RoBERTa-base | L128 voll | 308,25 | 311,90 |  | 565,60 | fehlt: nicht gelaufen | 1,83 |  | ok |
| RoBERTa-base | L512 voll | 1372,40 | 1378,90 |  | 2399,20 | fehlt: nicht gelaufen | 1,75 |  | ok |
| RoBERTa-base | echte Länge | 85,85 | 1374,90 |  | 143,60 | fehlt: nicht gelaufen | 1,67 |  | ok |
| mMiniLM | L128 voll | 84,50 | 85,50 |  | 149,80 | fehlt: nicht gelaufen | 1,77 |  | ok |
| mMiniLM | L512 voll | 430,10 | 448,80 |  | 726,35 | fehlt: nicht gelaufen | 1,69 |  | ok |
| mMiniLM | echte Länge | 36,60 | 63,50 |  | 66,25 | fehlt: nicht gelaufen | 1,81 |  | ok |
| DistilBERT | L128 voll | 155,20 | 156,60 |  | 276,60 | fehlt: nicht gelaufen | 1,78 |  | ok |
| DistilBERT | L512 voll | 710,25 | 776,30 |  | 1238,10 | fehlt: nicht gelaufen | 1,74 |  | ok |
| DistilBERT | echte Länge | 47,00 | 689,90 |  | 75,25 | fehlt: nicht gelaufen | 1,60 |  | ok |
| DeBERTa-v3-base | L128 voll | 335,20 | 348,80 |  | 627,15 | fehlt: nicht gelaufen | 1,87 |  | ok |
| DeBERTa-v3-base | L512 voll | 1584,70 | 1619,80 |  | 2802,30 | fehlt: nicht gelaufen | 1,77 |  | ok |
| DeBERTa-v3-base | echte Länge | 88,55 | 1576,50 |  | 159,75 | fehlt: nicht gelaufen | 1,80 |  | ok |
| granite | L128 voll | 113,85 | 115,80 |  | 198,60 | fehlt: nicht gelaufen | 1,74 |  | ok |
| granite | L512 voll | 577,15 | 582,20 |  | 890,70 | fehlt: nicht gelaufen | 1,54 |  | ok |
| granite | echte Länge | 23,60 | 54,40 |  | 40,70 | fehlt: nicht gelaufen | 1,72 |  | ok |

## webkit

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
| MiniLM | L128 voll | 71,50 | 71,60 |  | 73,60 | fehlt: nicht gelaufen | 1,03 |  | ok |
| MiniLM | L512 voll | 338,60 | 339,00 |  | 345,50 | fehlt: nicht gelaufen | 1,02 |  | ok |
| MiniLM | echte Länge | 19,05 | 32,00 |  | 20,95 | fehlt: nicht gelaufen | 1,10 |  | ok |
| RoBERTa-base | L128 voll | 530,40 | 531,40 |  | 541,60 | fehlt: nicht gelaufen | 1,02 |  | ok |
| RoBERTa-base | L512 voll | 2322,10 | 2505,60 |  | 2326,30 | fehlt: nicht gelaufen | 1,00 |  | ok |
| RoBERTa-base | echte Länge | 143,40 | 2135,10 |  | 150,20 | fehlt: nicht gelaufen | 1,05 |  | ok |
| mMiniLM | L128 voll | 139,90 | 140,10 |  | 144,10 | fehlt: nicht gelaufen | 1,03 |  | ok |
| mMiniLM | L512 voll | 674,40 | 674,80 |  | 684,00 | fehlt: nicht gelaufen | 1,01 |  | ok |
| mMiniLM | echte Länge | 58,20 | 96,50 |  | 145,25 | fehlt: nicht gelaufen | 2,50 |  | ok |
| DistilBERT | L128 voll | 261,00 | 262,30 |  | 271,30 | fehlt: nicht gelaufen | 1,04 |  | ok |
| DistilBERT | L512 voll | 1137,30 | 1143,10 |  | 1184,70 | fehlt: nicht gelaufen | 1,04 |  | ok |
| DistilBERT | echte Länge | 74,15 | 1077,20 |  | 77,40 | fehlt: nicht gelaufen | 1,04 |  | ok |
| DeBERTa-v3-base | L128 voll | 565,30 | 571,50 |  | 607,70 | fehlt: nicht gelaufen | 1,08 |  | ok |
| DeBERTa-v3-base | L512 voll | 2664,70 | 2674,90 |  | 2685,40 | fehlt: nicht gelaufen | 1,01 |  | ok |
| DeBERTa-v3-base | echte Länge | 144,05 | 2326,60 |  | 166,35 | fehlt: nicht gelaufen | 1,15 |  | ok |
| granite | L128 voll | 184,60 | 185,10 |  | 186,10 | fehlt: nicht gelaufen | 1,01 |  | ok |
| granite | L512 voll | 890,20 | 892,60 |  | 849,60 | fehlt: nicht gelaufen | 0,95 |  | ok |
| granite | echte Länge | 42,65 | 80,90 |  | 46,85 | fehlt: nicht gelaufen | 1,10 |  | ok |
