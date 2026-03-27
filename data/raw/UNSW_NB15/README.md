# UNSW-NB15 Dataset

Place the UNSW-NB15 CSV files here.

## Download

**Option A — Official UNSW research page (free):**
```
https://research.unsw.edu.au/projects/unsw-nb15-dataset
```
Request access on the page and download the CSV files.

**Option B — Kaggle mirror (faster):**
```
https://www.kaggle.com/datasets/dhoogla/unswnb15
```

**Option C — AARNet CloudStor (Australia-based CDN):**
```
https://cloudstor.aarnet.edu.au/plus/index.php/s/2DhnLGDdEECo4ys
```

## Expected files

The loader supports two layouts. Use whichever you have:

**Layout A (pre-split — preferred):**
```
UNSW_NB15/
├── UNSW_NB15_training-set.csv    (~175k rows)
└── UNSW_NB15_testing-set.csv     (~82k rows)
```

**Layout B (raw partitions — all 4 parts):**
```
UNSW_NB15/
├── UNSW-NB15_1.csv
├── UNSW-NB15_2.csv
├── UNSW-NB15_3.csv
└── UNSW-NB15_4.csv
```

## Dataset details

| Property | Value |
|----------|-------|
| Total records | ~2.5 million |
| Features | 49 features |
| Classes | 10 (Normal + 9 attack categories) |
| Format | CSV |

## Attack categories

`Normal`, `Fuzzers`, `Analysis`, `Backdoors`, `DoS`, `Exploits`, `Generic`, `Reconnaissance`, `Shellcode`, `Worms`

## Reference

> Moustafa, N. and Slay, J., "UNSW-NB15: A Comprehensive Data Set for Network Intrusion Detection
> Systems," MilCIS 2015.
