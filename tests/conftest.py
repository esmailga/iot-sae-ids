"""Shared fixtures: small synthetic raw datasets written in each dataset's real schema."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ids.utils.config import load_config


def _synthetic_flows(n: int, rng: np.random.Generator, attack_frac: float = 0.4) -> pd.DataFrame:
    is_attack = rng.random(n) < attack_frac
    dur = np.where(is_attack, rng.exponential(0.2, n), rng.exponential(2.0, n))
    sbytes = np.where(is_attack, rng.integers(0, 200, n), rng.integers(100, 5000, n))
    return pd.DataFrame({
        "duration": dur,
        "src_bytes": sbytes,
        "dst_bytes": sbytes * 2 + rng.integers(0, 10, n),  # highly correlated with src_bytes
        "src_pkts": rng.integers(1, 50, n),
        "proto": rng.choice(["tcp", "udp", "icmp"], n),
        "service": rng.choice(["-", "http", "dns", "ssl", "ftp", "smtp", "mqtt", "ssh"], n),
        "http_trans_depth": np.where(rng.random(n) < 0.9, "-", rng.integers(0, 3, n).astype(str)),
        "is_attack": is_attack,
    })


@pytest.fixture()
def raw_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Write synthetic TON_IoT / UNSW-NB15 / RT-IoT2022 raw files and point config at them."""
    rng = np.random.default_rng(0)
    raw = tmp_path / "data" / "raw"

    # TON_IoT schema
    df = _synthetic_flows(600, rng)
    attack_types = np.array(["scanning", "ddos", "xss", "password"])
    df["type"] = np.where(df.is_attack, rng.choice(attack_types, len(df)), "normal")
    df["label"] = df.is_attack.astype(int)
    df["ts"] = np.arange(len(df))
    df["src_ip"] = "192.168.1." + pd.Series(rng.integers(1, 255, len(df))).astype(str)
    df["dst_ip"] = "10.0.0.1"
    df["src_port"] = rng.integers(1024, 65535, len(df))
    df["dst_port"] = rng.choice([80, 443, 53], len(df))
    df["dns_query"] = "-"
    (raw / "ton_iot").mkdir(parents=True)
    df.drop(columns="is_attack").to_csv(raw / "ton_iot" / "Train_Test_Network.csv", index=False)

    # UNSW-NB15 schema (official train + test files)
    (raw / "unsw_nb15").mkdir(parents=True)
    for part, n in (("training", 500), ("testing", 300)):
        d = _synthetic_flows(n, rng)
        d.insert(0, "id", np.arange(1, n + 1))
        d["attack_cat"] = np.where(d.is_attack, rng.choice(["Exploits", "Generic", "DoS"], n),
                                   "Normal")
        d["label"] = d.is_attack.astype(int)
        d.drop(columns="is_attack").to_csv(raw / "unsw_nb15" / f"UNSW_NB15_{part}-set.csv",
                                           index=False)

    # RT-IoT2022 schema (local CSV fallback)
    d = _synthetic_flows(500, rng, attack_frac=0.7)
    d["id.orig_p"] = rng.integers(1024, 65535, len(d))
    d["id.resp_p"] = 1883
    d["Attack_type"] = np.where(
        d.is_attack, rng.choice(["DOS_SYN_Hping", "NMAP_UDP_SCAN", "ARP_poisioning"], len(d)),
        rng.choice(["MQTT_Publish", "Thing_Speak", "Wipro_bulb"], len(d)),
    )
    (raw / "rt_iot2022").mkdir(parents=True)
    d.drop(columns="is_attack").to_csv(raw / "rt_iot2022" / "RT_IOT2022.csv", index=False)

    monkeypatch.setenv("IDS_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("IDS_RESULTS_ROOT", str(tmp_path / "results"))
    return tmp_path


RT_CONFIRMED = [
    "dataset.normal_classes=[MQTT_Publish, Thing_Speak, Wipro_bulb]",
    "dataset.label_mapping_confirmed=true",
]


def cfg_for(name: str, extra: list[str] | None = None) -> dict:
    """Config for ``name`` with RT-IoT2022 mapping confirmed (test-only)."""
    ov = list(extra or [])
    if name == "rt_iot2022":
        ov += RT_CONFIRMED
    return load_config(name, ov)
