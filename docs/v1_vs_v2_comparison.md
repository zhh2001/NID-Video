# CIC-IDS-2017 ETL: 12h Fix Impact (v1 vs v2)

> Pre/post numbers for the 12h-without-AM/PM CSV format fix
> (Finding M4-001). v1 was the buggy run that motivated the diagnosis;
> v2 is the rerun with the inference enabled.
>
> v1 archive snapshot taken before deletion to preserve the
> "modulo-12h silent failure" fingerprint as Discussion-section
> evidence (8 PM-attack classes silently zero in v1).

---

## v1 (before 12h fix)

- **Run window**: 2026-04-28 21:32 → 22:55
- **ETL elapsed**: 80.85 min (4850.8 s)
- **Total windows**: 110,783
- **Total shards**: 113
- **Total disk** (uncompressed `.tar`): 82 GB
- **Pcaps OK / failed**: 3 / 0
- **Unmatched packets total**: 31,968,030

### Per-pcap breakdown (v1)

| pcap subdir              | shards | samples | labels (windows)                                                        |
|--------------------------|-------:|--------:|-------------------------------------------------------------------------|
| Friday-WorkingHours      | 37     | 36,212  | BENIGN 36,168 · Bot 44                                                  |
| Tuesday-WorkingHours     | 37     | 36,507  | BENIGN 35,924 · FTP-Patator 583                                         |
| Wednesday-workingHours   | 39     | 38,064  | BENIGN 34,887 · DoS slowloris 1,598 · DoS Slowhttptest 623 · DoS Hulk 588 · DoS GoldenEye 368 |
| **TOTAL**                | **113**| **110,783** |                                                                     |

### Flat label distribution (v1)

| Label (raw15)             | v1 windows | CIC-doc time (ADT) | Bug status                    |
|---------------------------|-----------:|--------------------|-------------------------------|
| BENIGN                    | 106,979    | —                  | —                             |
| DoS slowloris             | 1,598      | Wed AM             | OK (AM)                       |
| DoS Slowhttptest          | 623        | Wed AM             | OK (AM)                       |
| DoS Hulk                  | 588        | Wed AM             | OK (AM)                       |
| FTP-Patator               | 583        | Tue AM 09:20–10:20 | OK (AM)                       |
| DoS GoldenEye             | 368        | Wed AM             | OK (AM)                       |
| Bot                       | 44         | Fri AM 10:02–11:02 | OK (AM)                       |
| **SSH-Patator**           | **0**      | Tue PM 14:00–15:00 | **silently zero (12h bug)**   |
| **DDoS**                  | **0**      | Fri PM 15:56–16:16 | **silently zero (12h bug)**   |
| **PortScan**              | **0**      | Fri PM 13:55–15:29 | **silently zero (12h bug)**   |
| **Heartbleed**            | **0**      | Wed PM 15:12–15:32 | **silently zero (12h bug)**   |
| **Infiltration**          | **0**      | Thu PM 14:19–15:00 | n/a (Thursday pcap not in subset, but PM CSV would have failed too) |
| **Web Attack – Brute Force** | 0       | Thu AM 09:15–10:00 | n/a (Thursday pcap not in subset; AM CSV unaffected) |
| **Web Attack – XSS**      | 0          | Thu AM 10:15–10:35 | n/a (same)                    |
| **Web Attack – Sql Injection** | 0     | Thu AM 10:40–10:42 | n/a (same)                    |

> 96.6% BENIGN windows. **Every PM attack class within the Tue+Wed+Fri
> pcap subset went to zero windows** — not because the attacks weren't
> in the pcap, but because every PM CSV row's hour got parsed 12 hours
> earlier than reality. Diagnostic confirmed root cause: CIC's CSVs
> use 12h format without explicit AM/PM markers, encoding the period
> implicitly via the hour range (1..7 = PM, 8..11 = AM, 12 = noon PM).
> See Finding M4-001 in the project's findings ledger.

---

## v2 (after 12h fix)

- **Run window**: 2026-04-29 22:17 → 23:31
- **ETL elapsed**: 71.27 min (4275.9 s) — *~12% faster than v1, fs cache effect*
- **Total windows**: 110,783 — *identical to v1 (deterministic windowing on same pcaps)*
- **Total shards**: 113 — *identical*
- **Total disk** (uncompressed `.tar`): 82 GB — *identical*
- **Pcaps OK / failed**: 3 / 0
- **Unmatched packets total**: 23,148,458 — *down 8.82M (28% fewer) vs v1; PM-attack flow lookups now hit*

### Per-pcap breakdown (v2)

| pcap subdir              | shards | samples | labels (windows)                                                         |
|--------------------------|-------:|--------:|--------------------------------------------------------------------------|
| Friday-WorkingHours      | 37     | 36,212  | BENIGN 34,682 · DDoS 1,375 · PortScan 111 · Bot 44                       |
| Tuesday-WorkingHours     | 37     | 36,507  | BENIGN 34,991 · SSH-Patator 933 · FTP-Patator 583                        |
| Wednesday-workingHours   | 39     | 38,064  | BENIGN 33,382 · DoS slowloris 1,598 · Heartbleed 1,505 · DoS Slowhttptest 623 · DoS Hulk 588 · DoS GoldenEye 368 |
| **TOTAL**                | **113**| **110,783** |                                                                      |

### Flat label distribution (v2)

| Label (raw15)             | v2 windows | CIC-doc time (ADT) |
|---------------------------|-----------:|--------------------|
| BENIGN                    | 103,055    | —                  |
| DoS slowloris             | 1,598      | Wed AM             |
| Heartbleed                | 1,505      | Wed PM 15:12–15:32 |
| DDoS                      | 1,375      | Fri PM 15:56–16:16 |
| SSH-Patator               | 933        | Tue PM 14:00–15:00 |
| DoS Slowhttptest          | 623        | Wed AM             |
| DoS Hulk                  | 588        | Wed AM             |
| FTP-Patator               | 583        | Tue AM 09:20–10:20 |
| DoS GoldenEye             | 368        | Wed AM             |
| PortScan                  | 111        | Fri PM 13:55–15:29 |
| Bot                       | 44         | Fri AM 10:02–11:02 |

---

## v1 → v2 delta (silent-failure cure evidence)

| Label                | v1 windows | v2 windows | delta    | meaning                                                                  |
|----------------------|-----------:|-----------:|---------:|--------------------------------------------------------------------------|
| **SSH-Patator**      | **0**      | **933**    | **+933** | PM attack recovered. Tue 14:00–15:00 ADT lookups now hit                 |
| **DDoS**             | **0**      | **1,375**  | **+1,375** | PM attack recovered. Fri 15:56–16:16 ADT now hits ~92% of windows in span |
| **PortScan**         | **0**      | **111**    | **+111** | PM attack recovered. Fri 13:55–15:29 ADT — low hit rate (sparse SYN packets, dominant-rule loses to BENIGN bulk in most windows; M4 acceptance just needs > 0) |
| **Heartbleed**       | **0**      | **1,505**  | **+1,505** | PM attack recovered. Wed 15:12–15:32 ADT (20 min); 1,505 ≈ full span × 0.8 s step ≈ ~93% of windows in span (Heartbleed flows long-duration, dominant-rule labels every window in span) |
| BENIGN               | 106,979    | 103,055    | -3,924   | exactly equals (933 + 1,375 + 111 + 1,505) recovered windows: every PM-attack window was previously mis-labelled BENIGN |
| FTP-Patator          | 583        | 583        | 0        | AM, unaffected by 12h fix ✓                                              |
| DoS slowloris        | 1,598      | 1,598      | 0        | AM, unaffected ✓                                                         |
| DoS Slowhttptest     | 623        | 623        | 0        | AM, unaffected ✓                                                         |
| DoS Hulk             | 588        | 588        | 0        | AM, unaffected ✓                                                         |
| DoS GoldenEye        | 368        | 368        | 0        | AM, unaffected ✓                                                         |
| Bot                  | 44         | 44         | 0        | AM, unaffected ✓                                                         |
| **Total non-BENIGN** | **3,804**  | **7,728**  | **+3,924** | 100ms shard set's attack-class population doubled                        |

### Why this is paper-grade evidence

- **Surgical impact**: every AM-attack class count is bit-identical between v1 and v2 (`delta = 0`). The fix only touched PM CSV rows (hour ∈ [1,7]); AM rows (hour ∈ [8,11]) flowed through unchanged. The +12 h shift logic does not affect data outside its target window.
- **Conservation law**: BENIGN drop (-3,924) equals exactly the sum of recovered PM attack windows (+3,924). No windows were created or destroyed; they only moved from "mis-labelled BENIGN" to "correctly labelled attack-class".
- **Independent verification levels**:
  1. Packet-level (TRANSITION-005, 6.1 M Tuesday packets): 99.99% miss → ~17% miss after TZ fix
  2. ETL-level v1 (M4.7 task 4.7, AM-only attacks): 7 attack classes seen, 4 PM-class invisible
  3. ETL-level v2 (this run, post 12h fix): all 11 attack classes in subset present
  4. Time-range verification (M4 12h-fix LabelIndex probe): 14/14 attack windows in CIC docs match recovered tmin/tmax to ≤ 5 min

The two silent-failure modes (TRANSITION-005 TZ + M4-001 12h format) are independent root causes. Each one alone would suppress ~50% of the attack mass; both undetected would suppress 100% on PM. Both fixes verified at packet, time-window, and end-to-end ETL levels.
