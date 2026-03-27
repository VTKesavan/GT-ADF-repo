# Dataset Directory

Place downloaded datasets here with the following structure:

```
data/
├── raw/
│   ├── CICIDS2017/         ← CIC-IDS-2017 CSV files
│   │   ├── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
│   │   ├── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
│   │   ├── Monday-WorkingHours.pcap_ISCX.csv
│   │   ├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
│   │   ├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
│   │   ├── Tuesday-WorkingHours.pcap_ISCX.csv
│   │   └── Wednesday-workingHours.pcap_ISCX.csv
│   ├── UNSW_NB15/          ← UNSW-NB15 CSV files
│   │   ├── UNSW_NB15_training-set.csv
│   │   └── UNSW_NB15_testing-set.csv
│   └── ToN_IoT/            ← ToN-IoT CSV files
│       └── NF-ToN-IoT.csv  (or Train_Test_Network.csv)
└── processed/              ← Auto-generated PyG graphs (after preprocessing)
```

## Download Links

| Dataset | Official Source | Kaggle Mirror |
|---------|----------------|---------------|
| CICIDS2017 | http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/MachineLearningCSV.zip | https://www.kaggle.com/datasets/cicdataset/cicids2017 |
| UNSW-NB15 | https://research.unsw.edu.au/projects/unsw-nb15-dataset | https://www.kaggle.com/datasets/dhoogla/unswnb15 |
| ToN-IoT | https://research.unsw.edu.au/projects/toniot-datasets | https://www.kaggle.com/datasets/arnobbhowmik/ton-iot-network-dataset |

These files are excluded from the repository via `.gitignore`.
