## unsw_nb15

Files: `UNSW_NB15_training-set.csv` (found), `UNSW_NB15_testing-set.csv` (found)

### official train

- Rows: **175,341**; feature columns (excluding labels): **43**
- Numeric columns: 40; categorical/object columns: 3
- Exact duplicate rows (features + multiclass label): 0 (0.00%)
- Memory: 46.5 MB

#### official train — binary

| label | count | pct |
|---|---|---|
| 0 | 56000 | 31.94 |
| 1 | 119341 | 68.06 |

#### official train — multiclass

| class | binary | count | pct |
|---|---|---|---|
| normal | 0 | 56000 | 31.938 |
| Generic | 1 | 40000 | 22.813 |
| Exploits | 1 | 33393 | 19.045 |
| Fuzzers | 1 | 18184 | 10.371 |
| DoS | 1 | 12264 | 6.994 |
| Reconnaissance | 1 | 10491 | 5.983 |
| Analysis | 1 | 2000 | 1.141 |
| Backdoor | 1 | 1746 | 0.996 |
| Shellcode | 1 | 1133 | 0.646 |
| Worms | 1 | 130 | 0.074 |

### official test

- Rows: **82,332**; feature columns (excluding labels): **43**
- Numeric columns: 40; categorical/object columns: 3
- Exact duplicate rows (features + multiclass label): 0 (0.00%)
- Memory: 21.8 MB

#### official test — binary

| label | count | pct |
|---|---|---|
| 0 | 37000 | 44.94 |
| 1 | 45332 | 55.06 |

#### official test — multiclass

| class | binary | count | pct |
|---|---|---|---|
| normal | 0 | 37000 | 44.94 |
| Generic | 1 | 18871 | 22.921 |
| Exploits | 1 | 11132 | 13.521 |
| Fuzzers | 1 | 6062 | 7.363 |
| DoS | 1 | 4089 | 4.966 |
| Reconnaissance | 1 | 3496 | 4.246 |
| Analysis | 1 | 677 | 0.822 |
| Backdoor | 1 | 583 | 0.708 |
| Shellcode | 1 | 378 | 0.459 |
| Worms | 1 | 44 | 0.053 |

### Planned column drops (leakage control)

- Present and dropped (1): ['id']
- Configured but not in file: []
- Feature columns after drop: **42**

### Columns

| column | dtype | missing_% | dash_% | n_unique | negatives | example |
|---|---|---|---|---|---|---|
| id | int32 | 0.0 | 0.0 | 175341 | 0 |  |
| dur | float32 | 0.0 | 0.0 | 74039 | 0 |  |
| proto | object | 0.0 | 0.0 | 133 |  | ['tcp'] |
| service | object | 0.0 | 53.706 | 13 |  | ['-'] |
| state | object | 0.0 | 0.0 | 9 |  | ['FIN'] |
| spkts | int16 | 0.0 | 0.0 | 480 | 0 |  |
| dpkts | int16 | 0.0 | 0.0 | 443 | 0 |  |
| sbytes | int32 | 0.0 | 0.0 | 7214 | 0 |  |
| dbytes | int32 | 0.0 | 0.0 | 6660 | 0 |  |
| rate | float32 | 0.0 | 0.0 | 76794 | 0 |  |
| sttl | int16 | 0.0 | 0.0 | 11 | 0 |  |
| dttl | int16 | 0.0 | 0.0 | 6 | 0 |  |
| sload | float32 | 0.0 | 0.0 | 80885 | 0 |  |
| dload | float32 | 0.0 | 0.0 | 77474 | 0 |  |
| sloss | int16 | 0.0 | 0.0 | 409 | 0 |  |
| dloss | int16 | 0.0 | 0.0 | 370 | 0 |  |
| sinpkt | float32 | 0.0 | 0.0 | 76050 | 0 |  |
| dinpkt | float32 | 0.0 | 0.0 | 74177 | 0 |  |
| sjit | float32 | 0.0 | 0.0 | 77390 | 0 |  |
| djit | float32 | 0.0 | 0.0 | 76743 | 0 |  |
| swin | int16 | 0.0 | 0.0 | 13 | 0 |  |
| stcpb | int64 | 0.0 | 0.0 | 75265 | 0 |  |
| dtcpb | int64 | 0.0 | 0.0 | 75089 | 0 |  |
| dwin | int16 | 0.0 | 0.0 | 7 | 0 |  |
| tcprtt | float32 | 0.0 | 0.0 | 43319 | 0 |  |
| synack | float32 | 0.0 | 0.0 | 40142 | 0 |  |
| ackdat | float32 | 0.0 | 0.0 | 37708 | 0 |  |
| smean | int16 | 0.0 | 0.0 | 1357 | 0 |  |
| dmean | int16 | 0.0 | 0.0 | 1328 | 0 |  |
| trans_depth | int16 | 0.0 | 0.0 | 11 | 0 |  |
| response_body_len | int32 | 0.0 | 0.0 | 2386 | 0 |  |
| ct_srv_src | int8 | 0.0 | 0.0 | 52 | 0 |  |
| ct_state_ttl | int8 | 0.0 | 0.0 | 5 | 0 |  |
| ct_dst_ltm | int8 | 0.0 | 0.0 | 50 | 0 |  |
| ct_src_dport_ltm | int8 | 0.0 | 0.0 | 47 | 0 |  |
| ct_dst_sport_ltm | int8 | 0.0 | 0.0 | 32 | 0 |  |
| ct_dst_src_ltm | int8 | 0.0 | 0.0 | 54 | 0 |  |
| is_ftp_login | int8 | 0.0 | 0.0 | 4 | 0 |  |
| ct_ftp_cmd | int8 | 0.0 | 0.0 | 4 | 0 |  |
| ct_flw_http_mthd | int8 | 0.0 | 0.0 | 11 | 0 |  |
| ct_src_ltm | int8 | 0.0 | 0.0 | 50 | 0 |  |
| ct_srv_dst | int8 | 0.0 | 0.0 | 52 | 0 |  |
| is_sm_ips_ports | int8 | 0.0 | 0.0 | 2 | 0 |  |

Constant columns (zero variance; dropped during preprocessing): []

Columns with > 5% missing (NaN, would be dropped by preprocessing): []

### Impossible values

None detected by the length/duration/port rules.

