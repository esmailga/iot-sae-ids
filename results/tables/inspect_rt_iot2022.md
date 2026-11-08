## rt_iot2022

Files: `RT_IOT2022.csv` (found)

### all

- Rows: **123,117**; feature columns (excluding labels): **83**
- Numeric columns: 81; categorical/object columns: 2
- Exact duplicate rows (features + multiclass label): 5,195 (4.22%)
- Memory: 45.2 MB

#### all — binary

| label | count | pct |
|---|---|---|
| 0 | 12507 | 10.16 |
| 1 | 110610 | 89.84 |

#### all — multiclass

| class | binary | count | pct |
|---|---|---|---|
| DOS_SYN_Hping | 1 | 94659 | 76.885 |
| Thing_Speak | 0 | 8108 | 6.586 |
| ARP_poisioning | 1 | 7750 | 6.295 |
| MQTT_Publish | 0 | 4146 | 3.368 |
| NMAP_UDP_SCAN | 1 | 2590 | 2.104 |
| NMAP_XMAS_TREE_SCAN | 1 | 2010 | 1.633 |
| NMAP_OS_DETECTION | 1 | 2000 | 1.624 |
| NMAP_TCP_scan | 1 | 1002 | 0.814 |
| DDOS_Slowloris | 1 | 534 | 0.434 |
| Wipro_bulb | 0 | 253 | 0.205 |
| Metasploit_Brute_Force_SSH | 1 | 37 | 0.03 |
| NMAP_FIN_SCAN | 1 | 28 | 0.023 |

### Planned column drops (leakage control)

- Present and dropped (2): ['id.orig_p', 'id.resp_p']
- Configured but not in file: ['Unnamed: 0']
- Feature columns after drop: **81**

### Columns

| column | dtype | missing_% | dash_% | n_unique | negatives | example |
|---|---|---|---|---|---|---|
| id.orig_p | int32 | 0.0 | 0.0 | 65478 | 0 |  |
| id.resp_p | int32 | 0.0 | 0.0 | 1809 | 0 |  |
| proto | object | 0.0 | 0.0 | 3 |  | ['tcp'] |
| service | object | 0.0 | 83.547 | 10 |  | ['mqtt'] |
| flow_duration | float32 | 0.0 | 0.0 | 16364 | 0 |  |
| fwd_pkts_tot | int16 | 0.0 | 0.0 | 163 | 0 |  |
| bwd_pkts_tot | int16 | 0.0 | 0.0 | 172 | 0 |  |
| fwd_data_pkts_tot | int16 | 0.0 | 0.0 | 116 | 0 |  |
| bwd_data_pkts_tot | int16 | 0.0 | 0.0 | 154 | 0 |  |
| fwd_pkts_per_sec | float32 | 0.0 | 0.0 | 15891 | 0 |  |
| bwd_pkts_per_sec | float32 | 0.0 | 0.0 | 15426 | 0 |  |
| flow_pkts_per_sec | float32 | 0.0 | 0.0 | 16111 | 0 |  |
| down_up_ratio | float32 | 0.0 | 0.0 | 468 | 0 |  |
| fwd_header_size_tot | int32 | 0.0 | 0.0 | 353 | 0 |  |
| fwd_header_size_min | int8 | 0.0 | 0.0 | 7 | 0 |  |
| fwd_header_size_max | int8 | 0.0 | 0.0 | 8 | 0 |  |
| bwd_header_size_tot | int32 | 0.0 | 0.0 | 352 | 0 |  |
| bwd_header_size_min | int8 | 0.0 | 0.0 | 7 | 0 |  |
| bwd_header_size_max | int8 | 0.0 | 0.0 | 9 | 0 |  |
| flow_FIN_flag_count | int8 | 0.0 | 0.0 | 11 | 0 |  |
| flow_SYN_flag_count | int8 | 0.0 | 0.0 | 8 | 0 |  |
| flow_RST_flag_count | int8 | 0.0 | 0.0 | 9 | 0 |  |
| fwd_PSH_flag_count | int16 | 0.0 | 0.0 | 70 | 0 |  |
| bwd_PSH_flag_count | int16 | 0.0 | 0.0 | 98 | 0 |  |
| flow_ACK_flag_count | int16 | 0.0 | 0.0 | 210 | 0 |  |
| fwd_URG_flag_count | int8 | 0.0 | 0.0 | 2 | 0 |  |
| bwd_URG_flag_count | int8 | 0.0 | 0.0 | 1 | 0 |  |
| flow_CWR_flag_count | int8 | 0.0 | 0.0 | 5 | 0 |  |
| flow_ECE_flag_count | int8 | 0.0 | 0.0 | 5 | 0 |  |
| fwd_pkts_payload.min | int16 | 0.0 | 0.0 | 95 | 0 |  |
| fwd_pkts_payload.max | int16 | 0.0 | 0.0 | 407 | 0 |  |
| fwd_pkts_payload.tot | int32 | 0.0 | 0.0 | 1372 | 0 |  |
| fwd_pkts_payload.avg | float32 | 0.0 | 0.0 | 1700 | 0 |  |
| fwd_pkts_payload.std | float32 | 0.0 | 0.0 | 1739 | 0 |  |
| bwd_pkts_payload.min | int16 | 0.0 | 0.0 | 192 | 0 |  |
| bwd_pkts_payload.max | int16 | 0.0 | 0.0 | 499 | 0 |  |
| bwd_pkts_payload.tot | int32 | 0.0 | 0.0 | 1879 | 0 |  |
| bwd_pkts_payload.avg | float32 | 0.0 | 0.0 | 2363 | 0 |  |
| bwd_pkts_payload.std | float32 | 0.0 | 0.0 | 2580 | 0 |  |
| flow_pkts_payload.min | int16 | 0.0 | 0.0 | 98 | 0 |  |
| flow_pkts_payload.max | int16 | 0.0 | 0.0 | 527 | 0 |  |
| flow_pkts_payload.tot | int32 | 0.0 | 0.0 | 2201 | 0 |  |
| flow_pkts_payload.avg | float32 | 0.0 | 0.0 | 2774 | 0 |  |
| flow_pkts_payload.std | float32 | 0.0 | 0.0 | 3289 | 0 |  |
| fwd_iat.min | float32 | 0.0 | 0.0 | 2902 | 0 |  |
| fwd_iat.max | float32 | 0.0 | 0.0 | 12539 | 0 |  |
| fwd_iat.tot | float32 | 0.0 | 0.0 | 12659 | 0 |  |
| fwd_iat.avg | float32 | 0.0 | 0.0 | 12700 | 0 |  |
| fwd_iat.std | float32 | 0.0 | 0.0 | 10819 | 0 |  |
| bwd_iat.min | float32 | 0.0 | 0.0 | 6631 | 0 |  |
| bwd_iat.max | float32 | 0.0 | 0.0 | 14762 | 0 |  |
| bwd_iat.tot | float32 | 0.0 | 0.0 | 14852 | 0 |  |
| bwd_iat.avg | float32 | 0.0 | 0.0 | 14869 | 0 |  |
| bwd_iat.std | float32 | 0.0 | 0.0 | 10429 | 0 |  |
| flow_iat.min | float32 | 0.0 | 0.0 | 1846 | 0 |  |
| flow_iat.max | float32 | 0.0 | 0.0 | 16462 | 0 |  |
| flow_iat.tot | float32 | 0.0 | 0.0 | 16743 | 0 |  |
| flow_iat.avg | float32 | 0.0 | 0.0 | 16886 | 0 |  |
| flow_iat.std | float32 | 0.0 | 0.0 | 17148 | 0 |  |
| payload_bytes_per_second | float32 | 0.0 | 0.0 | 17234 | 0 |  |
| fwd_subflow_pkts | float32 | 0.0 | 0.0 | 355 | 0 |  |
| bwd_subflow_pkts | float32 | 0.0 | 0.0 | 375 | 0 |  |
| fwd_subflow_bytes | float32 | 0.0 | 0.0 | 1531 | 0 |  |
| bwd_subflow_bytes | float32 | 0.0 | 0.0 | 2038 | 0 |  |
| fwd_bulk_bytes | float32 | 0.0 | 0.0 | 220 | 0 |  |
| bwd_bulk_bytes | float32 | 0.0 | 0.0 | 459 | 0 |  |
| fwd_bulk_packets | float32 | 0.0 | 0.0 | 56 | 0 |  |
| bwd_bulk_packets | float32 | 0.0 | 0.0 | 137 | 0 |  |
| fwd_bulk_rate | float32 | 0.0 | 0.0 | 247 | 0 |  |
| bwd_bulk_rate | float32 | 0.0 | 0.0 | 926 | 0 |  |
| active.min | float32 | 0.0 | 0.0 | 16380 | 0 |  |
| active.max | float32 | 0.0 | 0.0 | 16722 | 0 |  |
| active.tot | float32 | 0.0 | 0.0 | 16724 | 0 |  |
| active.avg | float32 | 0.0 | 0.0 | 16729 | 0 |  |
| active.std | float32 | 0.0 | 0.0 | 1673 | 0 |  |
| idle.min | float32 | 0.0 | 0.0 | 5634 | 0 |  |
| idle.max | float32 | 0.0 | 0.0 | 5630 | 0 |  |
| idle.tot | float32 | 0.0 | 0.0 | 5635 | 0 |  |
| idle.avg | float32 | 0.0 | 0.0 | 5634 | 0 |  |
| idle.std | float32 | 0.0 | 0.0 | 1001 | 0 |  |
| fwd_init_window_size | int32 | 0.0 | 0.0 | 20 | 0 |  |
| bwd_init_window_size | int32 | 0.0 | 0.0 | 56 | 0 |  |
| fwd_last_window_size | int32 | 0.0 | 0.0 | 109 | 0 |  |

Constant columns (zero variance; dropped during preprocessing): ['bwd_URG_flag_count']

Columns with > 5% missing (NaN, would be dropped by preprocessing): []

### Impossible values

None detected by the length/duration/port rules.

### Proposed RT-IoT2022 normal-vs-attack mapping (NEEDS CONFIRMATION)

- Normal (benign device traffic), 12,507 rows: `MQTT_Publish` (4,146), `Thing_Speak` (8,108), `Wipro_bulb` (253)
- Attack, 110,610 rows: `ARP_poisioning` (7,750), `DDOS_Slowloris` (534), `DOS_SYN_Hping` (94,659), `Metasploit_Brute_Force_SSH` (37), `NMAP_FIN_SCAN` (28), `NMAP_OS_DETECTION` (2,000), `NMAP_TCP_scan` (1,002), `NMAP_UDP_SCAN` (2,590), `NMAP_XMAS_TREE_SCAN` (2,010)
- Confirmed in config: False

