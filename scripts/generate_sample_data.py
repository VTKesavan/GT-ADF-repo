#!/usr/bin/env python3
"""
Generate small synthetic CSV files that mimic the structure of each
benchmark dataset. These are NOT real network traffic — they are
randomly generated with realistic column names and label distributions
so that the full preprocessing -> graph-building -> training pipeline
can be exercised without downloading the original datasets.

Usage:
    python scripts/generate_sample_data.py

Outputs:
    sample_data/CICIDS2017_sample.csv     (2000 rows)
    sample_data/UNSW_NB15_sample.csv      (2000 rows)
    sample_data/ToN_IoT_sample.csv        (2000 rows)
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RNG = np.random.default_rng(42)
N = 2000


def make_probs(n_classes, dominant_share=0.83):
    """
    Return a probability array of length n_classes that sums exactly to 1.0.
    The first class gets dominant_share; the rest share equally.
    """
    each = (1.0 - dominant_share) / (n_classes - 1)
    probs = [dominant_share] + [each] * (n_classes - 1)
    # Fix floating-point drift by adjusting last element
    probs[-1] = round(1.0 - sum(probs[:-1]), 10)
    # Ensure no negative value from rounding edge cases
    probs[-1] = max(probs[-1], 0.0)
    arr = np.array(probs, dtype=np.float64)
    arr = arr / arr.sum()   # normalise to guarantee sum == 1.0 exactly
    return arr


# ── CICIDS2017 ─────────────────────────────────────────────────────────────

CICIDS_NUMERIC_COLS = [
    "Destination Port", "Flow Duration", "Total Fwd Packets",
    "Total Backward Packets", "Total Length of Fwd Packets",
    "Total Length of Bwd Packets", "Fwd Packet Length Max",
    "Fwd Packet Length Min", "Fwd Packet Length Mean",
    "Fwd Packet Length Std", "Bwd Packet Length Max",
    "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max",
    "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std",
    "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags",
    "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length",
    "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s",
    "Min Packet Length", "Max Packet Length", "Packet Length Mean",
    "Packet Length Std", "Packet Length Variance", "FIN Flag Count",
    "SYN Flag Count", "RST Flag Count", "PSH Flag Count",
    "ACK Flag Count", "URG Flag Count", "CWE Flag Count",
    "ECE Flag Count", "Down/Up Ratio", "Average Packet Size",
    "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "Fwd Header Length.1", "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate", "Subflow Fwd Packets",
    "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward",
    "act_data_pkt_fwd", "min_seg_size_forward",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
]

CICIDS_LABELS = [
    "BENIGN", "DDoS", "DoS Hulk", "DoS GoldenEye",
    "DoS slowloris", "DoS Slowhttptest", "PortScan",
    "FTP-Patator", "SSH-Patator", "Bot",
    "Web Attack - Brute Force", "Web Attack - XSS",
    "Web Attack - Sql Injection", "Infiltration", "Heartbleed",
]
CICIDS_PROBS = make_probs(len(CICIDS_LABELS), dominant_share=0.83)

data_cic = {col: RNG.exponential(scale=100, size=N).clip(0, 1e6)
            for col in CICIDS_NUMERIC_COLS}
data_cic[" Label"] = RNG.choice(CICIDS_LABELS, size=N, p=CICIDS_PROBS)

df_cic = pd.DataFrame(data_cic)
out_cic = os.path.join(OUTPUT_DIR, "CICIDS2017_sample.csv")
df_cic.to_csv(out_cic, index=False)
print(f"CICIDS2017 sample: {out_cic}  ({N} rows, {len(df_cic.columns)} cols)")
print(f"  Label distribution:\n{df_cic[' Label'].value_counts().to_string()}\n")


# ── UNSW-NB15 ──────────────────────────────────────────────────────────────

UNSW_COLS = [
    "dur", "proto", "service", "state", "spkts", "dpkts",
    "sbytes", "dbytes", "rate", "sttl", "dttl", "sload", "dload",
    "sloss", "dloss", "sinpkt", "dinpkt", "sjit", "djit",
    "swin", "stcpb", "dtcpb", "dwin", "tcprtt", "synack", "ackdat",
    "smean", "dmean", "trans_depth", "response_body_len",
    "ct_srv_src", "ct_state_ttl", "ct_dst_ltm", "ct_src_dport_ltm",
    "ct_dst_sport_ltm", "ct_dst_src_ltm", "is_ftp_login",
    "ct_ftp_cmd", "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst",
    "is_sm_ips_ports",
]
UNSW_CAT = {
    "proto":   ["tcp", "udp", "icmp", "arp"],
    "service": ["-", "http", "ftp", "smtp", "dns", "ssh"],
    "state":   ["FIN", "INT", "CON", "REQ", "RST"],
}

UNSW_ATTACK_CATS = [
    "Normal", "Fuzzers", "Analysis", "Backdoors", "DoS",
    "Exploits", "Generic", "Reconnaissance", "Shellcode", "Worms",
]
UNSW_PROBS = make_probs(len(UNSW_ATTACK_CATS), dominant_share=0.50)

data_unsw = {}
for col in UNSW_COLS:
    if col in UNSW_CAT:
        data_unsw[col] = RNG.choice(UNSW_CAT[col], size=N)
    else:
        data_unsw[col] = RNG.exponential(scale=50, size=N).clip(0, 1e5)

attack_cat = RNG.choice(UNSW_ATTACK_CATS, size=N, p=UNSW_PROBS)
data_unsw["attack_cat"] = attack_cat
data_unsw["label"] = (attack_cat != "Normal").astype(int)

df_unsw = pd.DataFrame(data_unsw)
out_unsw = os.path.join(OUTPUT_DIR, "UNSW_NB15_sample.csv")
df_unsw.to_csv(out_unsw, index=False)
print(f"UNSW-NB15 sample:  {out_unsw}  ({N} rows, {len(df_unsw.columns)} cols)")
print(f"  Attack cat distribution:\n{df_unsw['attack_cat'].value_counts().to_string()}\n")


# ── ToN-IoT ────────────────────────────────────────────────────────────────

TON_COLS = [
    "proto", "duration", "src_bytes", "dst_bytes", "src_pkts", "dst_pkts",
    "conn_state", "src_ip_bytes", "dst_ip_bytes", "dns_query",
    "dns_qclass", "dns_qtype", "dns_rcode", "dns_AA", "dns_RD",
    "dns_RA", "dns_rejected", "ssl_version", "ssl_cipher",
    "ssl_resumed", "ssl_established", "http_method", "http_version",
    "http_request_body_len", "http_response_body_len",
    "http_status_code", "weird_name", "weird_addl", "weird_notice",
    "missed_bytes", "src_port", "dst_port",
]
TON_CAT = {
    "proto":      ["tcp", "udp", "icmp"],
    "conn_state": ["S0", "S1", "SF", "REJ", "S2", "S3", "RSTO", "RSTR"],
}

TON_TYPES = [
    "normal", "backdoor", "ddos", "dos", "injection",
    "mitm", "password", "ransomware", "scanning", "xss",
]
TON_PROBS = make_probs(len(TON_TYPES), dominant_share=0.50)

data_ton = {}
for col in TON_COLS:
    if col in TON_CAT:
        data_ton[col] = RNG.choice(TON_CAT[col], size=N)
    elif col in ("dns_query", "weird_name", "weird_addl"):
        data_ton[col] = RNG.choice(["", "example.com", "malware.io"], size=N)
    else:
        data_ton[col] = RNG.exponential(scale=200, size=N).clip(0, 1e6)

attack_type = RNG.choice(TON_TYPES, size=N, p=TON_PROBS)
data_ton["type"] = attack_type
data_ton["label"] = (attack_type != "normal").astype(int)

df_ton = pd.DataFrame(data_ton)
out_ton = os.path.join(OUTPUT_DIR, "ToN_IoT_sample.csv")
df_ton.to_csv(out_ton, index=False)
print(f"ToN-IoT sample:    {out_ton}  ({N} rows, {len(df_ton.columns)} cols)")
print(f"  Type distribution:\n{df_ton['type'].value_counts().to_string()}\n")

print("=" * 60)
print("All sample datasets generated in sample_data/")
print("Next step:")
print("  python scripts\\preprocess.py --dataset cicids2017 ^")
print("      --data_dir sample_data --output_dir data\\processed --binary")
