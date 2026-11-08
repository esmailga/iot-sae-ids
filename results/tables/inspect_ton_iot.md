## ton_iot

Files: `train_test_network.csv` (found)

### all

- Rows: **211,043**; feature columns (excluding labels): **42**
- Numeric columns: 19; categorical/object columns: 23
- Exact duplicate rows (features + multiclass label): 20,569 (9.75%)
- Memory: 265.0 MB

#### all — binary

| label | count | pct |
|---|---|---|
| 0 | 50000 | 23.69 |
| 1 | 161043 | 76.31 |

#### all — multiclass

| class | binary | count | pct |
|---|---|---|---|
| normal | 0 | 50000 | 23.692 |
| backdoor | 1 | 20000 | 9.477 |
| ddos | 1 | 20000 | 9.477 |
| dos | 1 | 20000 | 9.477 |
| injection | 1 | 20000 | 9.477 |
| password | 1 | 20000 | 9.477 |
| scanning | 1 | 20000 | 9.477 |
| ransomware | 1 | 20000 | 9.477 |
| xss | 1 | 20000 | 9.477 |
| mitm | 1 | 1043 | 0.494 |

### Planned column drops (leakage control)

- Present and dropped (9): ['src_ip', 'dst_ip', 'src_port', 'dst_port', 'dns_query', 'ssl_subject', 'ssl_issuer', 'http_uri', 'http_user_agent']
- Configured but not in file: ['ts']
- Feature columns after drop: **33**

### Columns

| column | dtype | missing_% | dash_% | n_unique | negatives | example |
|---|---|---|---|---|---|---|
| src_ip | object | 0.0 | 0.0 | 51 |  | ['192.168.1.37'] |
| src_port | int32 | 0.0 | 0.0 | 26628 | 0 |  |
| dst_ip | object | 0.0 | 0.0 | 753 |  | ['192.168.1.193'] |
| dst_port | int32 | 0.0 | 0.0 | 2039 | 0 |  |
| proto | object | 0.0 | 0.0 | 3 |  | ['tcp'] |
| service | object | 0.0 | 62.562 | 9 |  | ['-'] |
| duration | float32 | 0.0 | 0.0 | 68523 | 0 |  |
| src_bytes | int64 | 0.0 | 0.0 | 2199 | 0 |  |
| dst_bytes | int64 | 0.0 | 0.0 | 2338 | 0 |  |
| conn_state | object | 0.0 | 0.0 | 13 |  | ['OTH'] |
| missed_bytes | int32 | 0.0 | 0.0 | 694 | 0 |  |
| src_pkts | int16 | 0.0 | 0.0 | 274 | 0 |  |
| src_ip_bytes | int32 | 0.0 | 0.0 | 3648 | 0 |  |
| dst_pkts | int32 | 0.0 | 0.0 | 203 | 0 |  |
| dst_ip_bytes | int32 | 0.0 | 0.0 | 3304 | 0 |  |
| dns_query | object | 0.0 | 83.489 | 726 |  | ['-'] |
| dns_qclass | int32 | 0.0 | 0.0 | 3 | 0 |  |
| dns_qtype | int16 | 0.0 | 0.0 | 12 | 0 |  |
| dns_rcode | int8 | 0.0 | 0.0 | 4 | 0 |  |
| dns_AA | object | 0.0 | 83.41 | 3 |  | ['-'] |
| dns_RD | object | 0.0 | 83.41 | 3 |  | ['-'] |
| dns_RA | object | 0.0 | 83.41 | 3 |  | ['-'] |
| dns_rejected | object | 0.0 | 83.41 | 3 |  | ['-'] |
| ssl_version | object | 0.0 | 99.81 | 4 |  | ['-'] |
| ssl_cipher | object | 0.0 | 99.81 | 6 |  | ['-'] |
| ssl_resumed | object | 0.0 | 99.81 | 3 |  | ['-'] |
| ssl_established | object | 0.0 | 99.81 | 3 |  | ['-'] |
| ssl_subject | object | 0.0 | 99.995 | 6 |  | ['-'] |
| ssl_issuer | object | 0.0 | 99.995 | 5 |  | ['-'] |
| http_trans_depth | float32 | 99.856 | 0.0 | 10 | 0 |  |
| http_method | object | 0.0 | 99.864 | 4 |  | ['-'] |
| http_uri | object | 0.0 | 99.864 | 86 |  | ['-'] |
| http_version | float32 | 99.859 | 0.0 | 1 | 0 |  |
| http_request_body_len | int16 | 0.0 | 0.0 | 6 | 0 |  |
| http_response_body_len | int32 | 0.0 | 0.0 | 75 | 0 |  |
| http_status_code | int16 | 0.0 | 0.0 | 8 | 0 |  |
| http_user_agent | object | 0.0 | 99.864 | 35 |  | ['-'] |
| http_orig_mime_types | object | 0.0 | 99.992 | 3 |  | ['-'] |
| http_resp_mime_types | object | 0.0 | 99.903 | 10 |  | ['-'] |
| weird_name | object | 0.0 | 99.831 | 11 |  | ['-'] |
| weird_addl | float32 | 99.926 | 0.0 | 2 | 0 |  |
| weird_notice | object | 0.0 | 99.831 | 2 |  | ['-'] |

Constant columns (zero variance; dropped during preprocessing): []

Columns with > 5% missing (NaN, would be dropped by preprocessing): ['http_trans_depth', 'http_version', 'weird_addl']

### Impossible values

None detected by the length/duration/port rules.

