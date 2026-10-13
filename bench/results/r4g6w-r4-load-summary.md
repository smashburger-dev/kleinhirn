# R4 load bench, tag r4g6w

| model | mode | profile | reps | ttfr ms | load ms | first call ms | call median ms | call max ms | warm median ms | MB models | MB engine | tokenizer ms | words (mode, rows, requests, bytes) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DeBERTa-v3-base | full | local | 3 | 684 | 668 | 16 | 8 | 11 | 7.90 | 388.37 | 0.86 | - | full, 128100, -, - |
| DeBERTa-v3-base | progressive | local | 3 | 427 | 401 | 30 | 16 | 153 | 7.90 | 211.91 | 0.91 | - | progressive, 128100, 150, 254100 |
| DeBERTa-v3-base | rows | local | 3 | 429 | 399 | 30 | 16 | 27 | 7.90 | 191.46 | 0.91 | - | rows, 288, 268, 443520 |
| DistilBERT | full | local | 3 | 273 | 259 | 16 | 4 | 11 | 4.00 | 134.17 | 0.86 | - | full, 30522, -, - |
| DistilBERT | progressive | local | 3 | 247 | 221 | 27 | 4 | 52 | 4.00 | 88.85 | 0.91 | - | progressive, 30522, 66, 144760 |
| DistilBERT | rows | local | 3 | 248 | 219 | 27 | 12 | 24 | 4.00 | 87.56 | 0.91 | - | rows, 279, 245, 429660 |
| granite | full | local | 3 | 230 | 217 | 14 | 4 | 10 | 4.40 | 95.93 | 0.86 | - | full, 50368, -, - |
| granite | progressive | local | 3 | 215 | 188 | 28 | 5 | 58 | 4.40 | 57.19 | 0.91 | - | progressive, 50368, 68, 64848 |
| granite | rows | local | 3 | 213 | 185 | 28 | 14 | 26 | 4.40 | 57.26 | 0.91 | - | rows, 344, 317, 265568 |
| MiniLM | full | local | 3 | 127 | 120 | 7 | 2 | 8 | 2.20 | 45.39 | 0.86 | - | full, 30522, -, - |
| MiniLM | progressive | local | 3 | 111 | 93 | 19 | 3 | 48 | 2.30 | 22.66 | 0.91 | - | progressive, 30522, 52, 57900 |
| MiniLM | rows | local | 3 | 116 | 95 | 20 | 11 | 24 | 2.20 | 22.05 | 0.91 | - | rows, 279, 245, 215388 |
| mMiniLM | full | local | 3 | 502 | 489 | 14 | 4 | 7 | 4.10 | 238.30 | 0.86 | - | full, 250002, -, - |
| mMiniLM | progressive | local | 3 | 220 | 190 | 32 | 16 | 188 | 4.10 | 45.50 | 0.91 | - | progressive, 250002, 218, 176788 |
| mMiniLM | rows | local | 3 | 224 | 191 | 34 | 12 | 41 | 4.10 | 45.50 | 0.91 | - | rows, 332, 319, 256304 |
| RoBERTa-base | full | local | 3 | 428 | 410 | 20 | 7 | 9 | 7.30 | 249.92 | 0.86 | - | full, 50265, -, - |
| RoBERTa-base | progressive | local | 3 | 382 | 351 | 32 | 8 | 80 | 7.40 | 205.73 | 0.91 | - | progressive, 50265, 95, 160160 |
| RoBERTa-base | rows | local | 3 | 369 | 337 | 31 | 16 | 28 | 7.40 | 172.87 | 0.91 | - | rows, 329, 319, 506660 |

ttfr, load, first call: median of reps. Call median and max: pooled over all reps of the cell. MB: decimal, from Resource Timing.
