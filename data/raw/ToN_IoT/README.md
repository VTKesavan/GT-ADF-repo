# ToN-IoT Dataset

Place the ToN-IoT network traffic CSV files here.

## Download

**Option A — Official UNSW research page (free):**
```
https://research.unsw.edu.au/projects/toniot-datasets
```

**Option B — Kaggle mirror:**
```
https://www.kaggle.com/datasets/arnobbhowmik/ton-iot-network-dataset
```

## Expected files

The loader accepts any CSV file(s) in this folder. Common filenames include:

```
ToN_IoT/
├── NF-ToN-IoT.csv              (NetFlow version, ~16M rows)
└── Train_Test_Network.csv      (pre-split network traffic subset)
```

You can also place both files and the loader will concatenate them.

## Dataset details

| Property | Value |
|----------|-------|
| Records (network) | ~16 million (NetFlow version) |
| Features | ~44 network flow features |
| Classes | 10 (normal + 9 attack types) |
| Format | CSV |

## Attack categories

`normal`, `backdoor`, `ddos`, `dos`, `injection`, `mitm`, `password`, `ransomware`, `scanning`, `xss`

## Which subset to use

For this paper, only the **network traffic** subset is used. The ToN-IoT collection also includes
OS-level (Windows/Linux) datasets — those are not needed here.

## Reference

> Alsaedi, A., Moustafa, N., Tari, Z., Mahmood, A., and Anwar, A.,
> "TON_IoT Telemetry Dataset: A New Generation Dataset of IoT and IIoT for
> Data-Driven Intrusion Detection Systems," IEEE Access, 2020.
