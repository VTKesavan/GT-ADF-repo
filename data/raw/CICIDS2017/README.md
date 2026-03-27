# CICIDS2017-SafeML Dataset

Place the **MachineLearningCSV** files from the CICIDS2017 dataset here.

## Download

**Option A — Official CIC server (free, no account needed):**
```
http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/MachineLearningCSV.zip
```
Extract the zip. You should have 8 CSV files. Place them directly in this folder.

**Option B — Kaggle mirror (faster, requires free Kaggle account):**
```
https://www.kaggle.com/datasets/cicdataset/cicids2017
```

## Expected files after extraction

```
CICIDS2017/
├── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
├── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
├── Friday-WorkingHours-Morning.pcap_ISCX.csv
├── Monday-WorkingHours.pcap_ISCX.csv
├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
├── Tuesday-WorkingHours.pcap_ISCX.csv
└── Wednesday-workingHours.pcap_ISCX.csv
```

## Dataset details

| Property | Value |
|----------|-------|
| Total records | ~2.8 million |
| Features | 78 network flow features |
| Classes | 15 (BENIGN + 14 attack types) |
| Format | CSV (CICFlowMeter-generated) |

## Reference

> Iman Sharafaldin, Arash Habibi Lashkari, and Ali A. Ghorbani,
> "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization",
> ICISSP 2018.

## Notes

- The CIC server (205.174.165.80) has intermittent availability — use the Kaggle mirror as a backup.
- Column names contain leading/trailing spaces (e.g. `" Label"`) — the dataset loader handles this automatically.
- The `Heartbleed` and `Infiltration` classes are very rare (<10 samples each). This is handled by stratified splitting in the preprocessor.
