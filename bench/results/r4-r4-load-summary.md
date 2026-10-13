# R4 load bench, tag r4

| model | mode | profile | reps | ttfr ms | load ms | first call ms | call median ms | call max ms | MB models | MB engine | tokenizer ms | words (mode, rows, requests, bytes) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DeBERTa-v3-base | full | 50m | 1 | 62521 | 62500 | 21 | 8 | 11 | 388.37 | 0.86 | - | full, 128100, -, - |
| DeBERTa-v3-base | pre | 50m | 1 | 62488 | 62470 | 18 | 8 | 11 | 387.85 | 0.76 | - | -, -, -, - |
| DeBERTa-v3-base | progressive | 50m | 1 | 31423 | 30958 | 465 | 240 | 658 | 191.46 | 0.91 | - | progressive, 288, 269, 443520 |
| DeBERTa-v3-base | rows | 50m | 1 | 31414 | 30954 | 460 | 351 | 966 | 191.46 | 0.91 | - | rows, 288, 268, 443520 |
| DeBERTa-v3-base | full | local | 3 | 704 | 683 | 19 | 8 | 11 | 388.37 | 0.86 | - | full, 128100, -, - |
| DeBERTa-v3-base | pre | local | 3 | 880 | 859 | 17 | 8 | 15 | 387.85 | 0.76 | - | -, -, -, - |
| DeBERTa-v3-base | progressive | local | 3 | 436 | 408 | 28 | 14 | 177 | 211.89 | 0.91 | - | progressive, 128100, 150, 254100 |
| DeBERTa-v3-base | rows | local | 3 | 434 | 407 | 27 | 16 | 31 | 191.46 | 0.91 | - | rows, 288, 268, 443520 |
| DistilBERT | full | 50m | 1 | 21754 | 21736 | 18 | 4 | 7 | 134.17 | 0.86 | - | full, 30522, -, - |
| DistilBERT | pre | 50m | 1 | 21851 | 21837 | 14 | 4 | 9 | 134.05 | 0.76 | - | -, -, -, - |
| DistilBERT | progressive | 50m | 1 | 14809 | 14317 | 492 | 220 | 549 | 87.56 | 0.91 | - | progressive, 279, 246, 429660 |
| DistilBERT | rows | 50m | 1 | 14821 | 14317 | 504 | 320 | 792 | 87.56 | 0.91 | - | rows, 279, 245, 429660 |
| DistilBERT | full | local | 3 | 320 | 306 | 14 | 4 | 11 | 134.17 | 0.86 | - | full, 30522, -, - |
| DistilBERT | pre | local | 3 | 372 | 360 | 13 | 4 | 11 | 134.05 | 0.76 | - | -, -, -, - |
| DistilBERT | progressive | local | 3 | 311 | 283 | 30 | 5 | 67 | 88.85 | 0.91 | - | progressive, 30522, 66, 144760 |
| DistilBERT | rows | local | 3 | 292 | 263 | 29 | 12 | 25 | 87.56 | 0.91 | - | rows, 279, 245, 429660 |
| granite | full | 10m | 1 | 78288 | 78276 | 12 | 5 | 10 | 95.93 | 0.86 | - | full, 50368, -, - |
| granite | pre | 10m | 1 | 80768 | 80759 | 9 | 5 | 7 | 95.73 | 0.76 | - | -, -, -, - |
| granite | progressive | 10m | 1 | 51052 | 48811 | 2242 | 1426 | 3375 | 57.26 | 0.91 | - | progressive, 344, 318, 265568 |
| granite | rows | 10m | 1 | 51043 | 48777 | 2266 | 1532 | 3620 | 57.26 | 0.91 | - | rows, 344, 317, 265568 |
| granite | full | 50m | 1 | 15584 | 15573 | 11 | 4 | 7 | 95.93 | 0.86 | - | full, 50368, -, - |
| granite | pre | 50m | 1 | 15674 | 15666 | 8 | 4 | 7 | 95.73 | 0.76 | - | -, -, -, - |
| granite | progressive | 50m | 1 | 10076 | 9461 | 615 | 291 | 673 | 57.26 | 0.91 | - | progressive, 344, 318, 265568 |
| granite | rows | 50m | 1 | 10061 | 9461 | 600 | 416 | 958 | 57.26 | 0.91 | - | rows, 344, 317, 265568 |
| granite | full | local | 3 | 237 | 225 | 12 | 4 | 12 | 95.93 | 0.86 | - | full, 50368, -, - |
| granite | pre | local | 3 | 347 | 336 | 10 | 4 | 8 | 95.73 | 0.76 | - | -, -, -, - |
| granite | progressive | local | 3 | 218 | 190 | 28 | 5 | 64 | 57.19 | 0.91 | - | progressive, 50368, 68, 64848 |
| granite | rows | local | 3 | 220 | 190 | 29 | 14 | 33 | 57.26 | 0.91 | - | rows, 344, 317, 265568 |
| MiniLM | full | 10m | 1 | 37223 | 37217 | 7 | 3 | 6 | 45.39 | 0.86 | - | full, 30522, -, - |
| MiniLM | pre | 10m | 1 | 38832 | 38825 | 7 | 3 | 9 | 45.27 | 0.76 | - | -, -, -, - |
| MiniLM | progressive | 10m | 1 | 21208 | 19403 | 1805 | 901 | 2733 | 22.05 | 0.91 | - | progressive, 30522, 228, 201492 |
| MiniLM | rows | 10m | 1 | 21195 | 19375 | 1820 | 1179 | 2931 | 22.05 | 0.91 | - | rows, 279, 245, 215388 |
| MiniLM | full | 50m | 1 | 7468 | 7461 | 7 | 2 | 5 | 45.39 | 0.86 | - | full, 30522, -, - |
| MiniLM | pre | 50m | 1 | 7591 | 7585 | 7 | 3 | 8 | 45.27 | 0.76 | - | -, -, -, - |
| MiniLM | progressive | 50m | 1 | 4281 | 3794 | 487 | 190 | 613 | 22.05 | 0.91 | - | progressive, 30522, 228, 201492 |
| MiniLM | rows | 50m | 1 | 4281 | 3801 | 480 | 325 | 775 | 22.05 | 0.91 | - | rows, 279, 245, 215388 |
| MiniLM | full | local | 3 | 166 | 158 | 9 | 3 | 7 | 45.39 | 0.86 | - | full, 30522, -, - |
| MiniLM | pre | local | 3 | 204 | 196 | 9 | 2 | 7 | 45.27 | 0.76 | - | -, -, -, - |
| MiniLM | progressive | local | 3 | 143 | 121 | 24 | 3 | 52 | 22.68 | 0.91 | - | progressive, 30522, 66, 72568 |
| MiniLM | rows | local | 3 | 154 | 133 | 24 | 14 | 30 | 22.05 | 0.91 | - | rows, 279, 245, 215388 |
| mMiniLM | full | 10m | 1 | 196788 | 196771 | 17 | 5 | 8 | 238.30 | 0.86 | - | full, 250002, -, - |
| mMiniLM | pre | 10m | 1 | 191052 | 191033 | 19 | 5 | 8 | 237.30 | 0.76 | - | -, -, -, - |
| mMiniLM | progressive | 10m | 1 | 41507 | 38446 | 3061 | 1175 | 5220 | 45.50 | 0.91 | - | progressive, 332, 320, 256304 |
| mMiniLM | rows | 10m | 1 | 41467 | 38426 | 3042 | 1253 | 5635 | 45.50 | 0.91 | - | rows, 332, 319, 256304 |
| mMiniLM | full | 50m | 1 | 38471 | 38457 | 14 | 4 | 8 | 238.30 | 0.86 | - | full, 250002, -, - |
| mMiniLM | pre | 50m | 1 | 38396 | 38380 | 15 | 4 | 13 | 237.30 | 0.76 | - | -, -, -, - |
| mMiniLM | progressive | 50m | 1 | 8298 | 7478 | 820 | 262 | 1053 | 45.50 | 0.91 | - | progressive, 332, 320, 256304 |
| mMiniLM | rows | 50m | 1 | 8302 | 7480 | 822 | 343 | 1522 | 45.50 | 0.91 | - | rows, 332, 319, 256304 |
| mMiniLM | full | local | 3 | 587 | 573 | 14 | 4 | 8 | 238.30 | 0.86 | - | full, 250002, -, - |
| mMiniLM | pre | local | 3 | 759 | 744 | 14 | 4 | 9 | 237.30 | 0.76 | - | -, -, -, - |
| mMiniLM | progressive | local | 3 | 264 | 225 | 39 | 17 | 146 | 45.50 | 0.91 | - | progressive, 250002, 221, 179104 |
| mMiniLM | rows | local | 3 | 264 | 227 | 37 | 13 | 47 | 45.50 | 0.91 | - | rows, 332, 319, 256304 |
| RoBERTa-base | full | 50m | 1 | 40301 | 40278 | 24 | 7 | 9 | 249.92 | 0.86 | - | full, 50265, -, - |
| RoBERTa-base | pre | 50m | 1 | 40403 | 40384 | 19 | 7 | 10 | 249.72 | 0.76 | - | -, -, -, - |
| RoBERTa-base | progressive | 50m | 1 | 28521 | 27968 | 552 | 276 | 709 | 172.87 | 0.91 | - | progressive, 329, 320, 506660 |
| RoBERTa-base | rows | 50m | 1 | 28513 | 27966 | 547 | 400 | 1005 | 172.87 | 0.91 | - | rows, 329, 319, 506660 |
| RoBERTa-base | full | local | 3 | 520 | 499 | 21 | 8 | 11 | 249.92 | 0.86 | - | full, 50265, -, - |
| RoBERTa-base | pre | local | 3 | 594 | 571 | 23 | 8 | 12 | 249.72 | 0.76 | - | -, -, -, - |
| RoBERTa-base | progressive | local | 3 | 446 | 411 | 36 | 8 | 103 | 205.73 | 0.91 | - | progressive, 50265, 95, 160160 |
| RoBERTa-base | rows | local | 3 | 444 | 412 | 34 | 18 | 35 | 172.87 | 0.91 | - | rows, 329, 319, 506660 |

ttfr, load, first call: median of reps. Call median and max: pooled over all reps of the cell. MB: decimal, from Resource Timing.
