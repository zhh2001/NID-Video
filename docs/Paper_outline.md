# Paper Outline v1(ACSAC 2027 target,N+5+ drafting 起点)

> Target: ~10-12 pages + references(ACSAC 2026 CFP 标准,2027 CFP 出来后 verify)。
> Style: characterization paper,不 propose new method,Pareto-front design space exploration。

> **v0 → v1 (N+3 P9.5) changelog**:
> - 72 entries (post-N+3 P5.5 + P7) anchor mapped to §1-§8
> - §3 / §6 anchor lists expanded per-subsection (42 / 42 literal each)
> - §5 promotes aspirational anchors (literal 3 → with 7+ aspirational)
> - §7 / §8 populate previously-empty v0 anchor lists (10 + 3 literal)
> - 6 ambiguous "Implementation Notes" entries: 3 dropped (no paper exposure) + 3 mapped to §3.6 Reproducibility
> - §1 / §2 retain Idea.md-driven narrative pattern (no finding 🎯 tags those sections)
> - Density check table refreshed with v1 literal + aspirational mapping

## Title 候选

1. "Characterizing Video-Backbone Models for Network Intrusion Detection: A Cross-Paradigm Empirical Study"
2. "Video, 2D, 1D, or Flow Features? An Empirical Characterization of Representational Paradigms for NIDS"
3. "A Pareto-Front View of Video-Backbone Models on CIC-IDS-2017"

## Abstract(target ~250 words)

- 一句问题陈述(网络入侵检测的多范式表征 landscape)
- 一句方法 framing(empirical characterization across 4 paradigms × 12 cells)
- 主要 finding 列出 4-5 条(候选:M5-005 v3 三维耦合 / M5-007 + M5-008 K400 transfer / M5-013 SSv2 anti-evidence / M5-014 Pareto navigator / M6-004 + M6-008 paradigm hierarchy / silent failure detection chain)
- 一句 scope acknowledgment(single dataset + no SoTA 比 + characterization not propose)
- 一句 reproducibility(consumer-grade GPU + 488 MB peak)

## §1 Introduction(~1.5 pages)

- §1.1 Background:NID + 多范式 landscape(flow / 1D / 2D / video)
- §1.2 Gap:跨范式 per-class behavior characterization 缺位;已发表数字不可比 due to label space / eval policy 差异
- §1.3 This paper:empirical characterization under unified fairness contract
  - 4 paradigm × ~12 cells
  - 5 ablation dimensions(round 1 共 4 + M6 cross-paradigm,维度 3 deferred)
- §1.4 Contributions(7 条 from Idea.md §6 v2)
- §1.5 Scope acknowledgment:single dataset / no SoTA 比 / characterization not propose / consumer-grade reproducibility

**Narrative anchor**: Idea.md §1.1 problem diagnosis + §6 contributions list
**Cite-from anchors (for §1.4 contributions)**:
- Silent failure chain: TRANSITION-005 / M4-001 / M4-002 / M4-010a / M5-003 / M5-006 (all [HIGHEST] cluster)
- M5-005 v3 [HIGHEST] 三维耦合 (contribution #2 anchor)
- M5-007 [HIGHEST] cross-baseline architecture × pretraining (contribution #1)
- M5-008 [HIGHEST] K400 head start + M5-013 [HIGHEST] SSv2 anti-evidence (contribution #3)
- M5-014 [HIGH] Pareto navigator framing (contribution #4)
- M6-004 [HIGHEST] + M6-008 [HIGH] 4-paradigm hierarchy (contribution #5)

## §2 Related Work(~1 page)

- §2.1 flow-feature paradigm:Sharafaldin CIC-IDS-2017 + RF/XGBoost/DL surveys
- §2.2 1D byte-sequence paradigm:DeepLog / Kitsune / 1D Transformer
- §2.3 2D snapshot paradigm:FlowPic / image-based IDS
- §2.4 video / spatial-temporal paradigm:本工作首次系统对照
- §2.5 Cross-paradigm comparison gap

**Narrative anchor**: Idea.md §2
**Empirical anchors (for §2.5 cross-paradigm gap)**:
- M6-003 [HIGH]: per-flow vs per-window eval space discontinuity (literature framing)
- M6-004 [HIGHEST]: 11-cell paradigm hierarchy empirical anchor
- M6-008 [HIGH]: 12-cell hierarchy update with M6.1

## §3 Methods(~2 pages)

### §3.1 Representation design — (T=16, C=6, H=32, W=64) tube tensor + 13-class collapsed labeling
**Narrative anchor**: Idea.md §3.1

### §3.2 Channel encoding (6 channels) + Spatial layout — H-axis (k-means IP) + W-axis (port log buckets)
- M2-001 [HIGH]: 逐窗 k-means H-axis 防标签泄露
- M2-002 [MEDIUM]: TCP flags 6-bit 选择
- M2-003 [MEDIUM]: Ch 6 log-delta motion channel
- M2-004 [HIGH]: W-axis min(src,dst) 连接对称
- M2-005 [MEDIUM]: log 桶 1/3-octave 48-列填满
- Cross-link: TRANSITION-005 [HIGHEST] (dataset TZ correctness, also §3.6 + §6.1)

### §3.3 Spatial layout — semantic IP clustering (adopted, not contributing); ablation deferred per Idea.md §7.2

### §3.4 Multi-scale + scale token + Pretrain rationale (Pareto navigator framing, NOT contribution)
- M3-001 [HIGHEST]: VideoMAE channel preservation 5.28× norm ratio (项目核心实证)
- M3-002 [HIGHEST]: ignore_mismatched_sizes silent pitfall
- M3-005 [MEDIUM]: VideoMAE 位置编码 sinusoidal not Parameter
- M3-006 [LOW]: VideoMAE-Small 16-head attention 配置
- M5-008 [HIGHEST]: K400 5-epoch loss-level inductive head start
- M5-013 [HIGHEST]: SSv2 corpus zero transfer (anti-evidence)
- M5-014 [HIGH]: motion + scale token combined ≈ 0 — Pareto navigator framing
- M5-015 [MEDIUM]: K400 slow-stream 3.6× lift
- M5-016 [HIGH]: scale token Bot AUROC stabilizer (joint regime)
- M5-019 [LOW]: K400 stem feature drift +49% over 10 epochs (init-effect feature-level)
- M5-020 [LOW]: K400-transient Bot detection ep 0-1 (init-effect detection-level)
- M6-005 [HIGH]: ImageNet ResNet zero benefit (cross-paradigm pretrain comparison anchor)

### §3.5 Fairness contract — Path B head_lr × init status + noise-free no_cycle eval
- M5-005 [HIGHEST]: head_lr × pretrained-status 耦合 (Path B contract design)
- M5-005 v3 [HIGHEST]: 三维耦合 (head_lr ×5 ∧ multi-scale)
- M5-006 [HIGH]: trainer head matcher silent failure (multi-baseline)
- M5-003 [HIGHEST]: no_cycle eval methodology + cycling-induced inflation
- M5-004 [HIGH]: focal loss prevent vs correct + reweight saturation

### §3.6 Reproducibility — RTX 4060 Mobile / 8 GB / WSL2 / 488 MB peak GPU + Dataset preprocessing pitfalls
**Hardware + environment**:
- M1-001 [MEDIUM]: WSL2 + uv + CUDA 13 environment baseline
- M3-007 [HIGH]: memory assumption correction (Idea.md narrative refinement)
- M3-009 [HIGH]: throughput saturation b=128-256 (12× speed-up vs base.yaml)
- M3-010 [LOW]: WSL2 b=1024 hang footgun

**Computational efficiency (from ambiguous → §3.6)**:
- M2-009 [MEDIUM]: ETL 138k pps L2 ceiling + 18k pps end-to-end
- M4-009 [LOW]: tensor compressibility 0.3% gzip (real traffic asymmetry)
- TRANSITION-002 [LOW]: pcapng vs classic 55% throughput

**Dataset preprocessing pitfalls (community-service cluster)**:
- TRANSITION-001 [MEDIUM]: CIC pcap is pcapng (not classic)
- TRANSITION-003 [MEDIUM]: CIC CSV CP-1252 encoding
- TRANSITION-006 [LOW]: dpkt truncation returns raw bytes
- M2-007 [MEDIUM]: EN-DASH (U+2013) vs ASCII hyphen label match

## §4 Experimental Setup(~1 page)

### §4.1 Dataset CIC-IDS-2017,13-class collapsed,val_n=18,156 bit-identical across all cells
- M2-006 [HIGH]: 15 raw + 13 collapsed dual-track labeling
- M2-008 [HIGH]: Bot n=12 rare-class statistical edge disclosure
- TRANSITION-004 [LOW]: CSV trailing comma 288,602 rows
- TRANSITION-008 [MEDIUM]: dominant-attack labeling rule (low-intensity attack blind)

### §4.2 Models — 12-row table: 6 video backbones + M6.1 1D Transformer + M6.2 RF/XGBoost + M6.3 2D ResNet
- M5-007 [HIGHEST]: cross-baseline 6-row 总览 (aspirational anchor)
- M6-004 [HIGHEST]: 11-cell paradigm hierarchy
- M6-008 [HIGH]: 12-cell hierarchy update with M6.1 1D byte

### §4.3 Training contract — focal γ=2 + inverse_sqrt α + Path B head_lr,10 epochs,48,530 grad_steps
- M5-004 [HIGH]: focal loss design + α reweight saturation
- M5-005 v3 [HIGHEST]: Path B head_lr × init contract anchor (aspirational)
- M4-007 [MEDIUM]: total_steps × epoch_end_strategy bug + fix
- M5-001 [MEDIUM]: total_steps × epoch_end_strategy 策略级泛化

### §4.4 Evaluation — no_cycle noise-free,combined / fast / slow macro_f1 + per-class
- M5-003 [HIGHEST]: no_cycle eval strategy methodology anchor (aspirational)
- M6-003 [HIGH]: per-flow vs per-window eval space framing (cross-paradigm fair contract justification)

## §5 Results(~2 pages)

### §5.1 Cross-paradigm 12-row table (combined macro_f1 + per-class signature)
- M5-007 [HIGHEST]: 6-row video cross-baseline (aspirational, K400 vs random group)
- M6-001 [HIGHEST]: M6.2 BENIGN-bias paradigm pathology
- M6-004 [HIGHEST]: 11-cell paradigm hierarchy strict ordering
- M6-008 [HIGH]: 12-cell hierarchy update with M6.1 (video > 1D byte > 2D snap > flow)

### §5.2 13×12 per-class grand table
- M5-007 [HIGHEST]: per-class signature (aspirational anchor)
- M6-007 [MEDIUM]: Heartbleed paradigm-specific F1 case
- M6-010 [MEDIUM]: DoS slowloris in 1D byte
- M6-011 [MEDIUM]: DoS GoldenEye AUROC decay in 1D byte

### §5.3 K400 prior loss-level inductive evidence (M5-008 figure)
- M5-008 [HIGHEST]: K400 5-epoch loss head start (aspirational figure anchor)
- M5-019 [LOW]: K400 stem feature drift +49% (forensic)
- M5-020 [LOW]: K400-transient Bot detection 3-tier

### §5.4 Three-way coupling for Bot AUROC (M5-005 v3 figure)
- M5-005 [HIGHEST]: head_lr × pretrained 2D anchor
- M5-005 v3 [HIGHEST]: 三维耦合 (aspirational §5 figure anchor)

### §5.5 Negative findings — motion / scale token zero-contribution (M5-014 framing)
- M5-014 [HIGH]: motion + scale token combined ≈ 0 (aspirational)
- M5-016 [HIGH]: scale token Bot AUROC stabilization (secondary metric load-bearing)

### §5.6 Architecture × pretraining decomposition — within-group spread > group gap
- M5-007 [HIGHEST]: cross-arch evidence (aspirational)
- M5-012 [HIGH]: TimeSformer R1 双重 unique architectural property
- M5-013 [HIGHEST]: SSv2 anti-evidence (corpus-specific transfer)

## §6 Discussion(~2 pages)

### §6.1 Silent failure detection chain — **flagship methodology narrative**
- TRANSITION-005 [HIGHEST]: pcap UTC vs CSV ADT 3h offset (packet-level anchor)
- TRANSITION-007 [HIGHEST]: ETL dry-run TZ fix verification (端到端 closure)
- M4-001 [HIGHEST]: CIC CSV 12h-no-AM/PM offset (二独立 silent failure 根因)
- M4-002 [HIGHEST]: 三守恒律 algebraic proof
- M4-010a [HIGHEST]: M4.8 downstream F1 closure (training-level final验证)
- M5-003 [HIGHEST]: no_cycle eval methodology (evaluation-policy silent failure)
- M5-006 [HIGH]: trainer head matcher segment-match fix (multi-baseline)
- M3-002 [HIGHEST]: ignore_mismatched_sizes pitfall (silent failure sibling)
- M5-018 [LOW]: CC stop-and-report methodology infrastructure (optional Methodology / Acknowledgments mention)

### §6.2 K400 transfer mechanism — initialization-effect vs feature-preservation
- M5-007 [HIGHEST]: K400 helps at NID scale (group lift +0.037)
- M5-008 [HIGHEST]: K400 = 5-epoch loss-level inductive head start
- M5-013 [HIGHEST]: SSv2 corpus zero transfer (corpus-specific anti-evidence)
- M5-015 [MEDIUM]: K400 slow-stream specific lift (3.6× vs fast)
- M5-019 [LOW]: K400 stem feature drift +49% (feature-level)
- M5-020 [LOW]: K400-transient Bot detection ep 0-1 (detection-level)
- M6-005 [HIGH]: ImageNet ResNet zero benefit (cross-paradigm pretrain comparison)

### §6.3 Training dynamics universal patterns
- M5-002 [HIGHEST]: vanilla CE Bot AUROC step-collapse (epoch 1-3)
- M5-009 [MEDIUM]: DDoS final-epoch jump 18 cells universal
- M5-010 [HIGH]: dip count × magnitude 与 architecture × head_lr 4×2 matrix
- M5-011 [MEDIUM]: GoldenEye oscillation 18 cells universal
- M5-017 [MEDIUM]: dim 4 single-stream OOD asymmetry
- M6-009 [HIGH]: M6.1 epoch-0 argmax 2-class collapse → epoch-9 7-class recovery
- M6-011 [MEDIUM]: M6.1 GoldenEye AUROC decay 0.82→0.65

### §6.4 Pareto-front design space + Bot rare-class paradigm scope
- M5-005 v3 [HIGHEST]: 三维耦合 (head_lr × init × multi-scale)
- M5-012 [HIGH]: TimeSformer R1 双重 unique (sustained Bot AUROC + smooth DDoS)
- M5-014 [HIGH]: motion / scale token Pareto navigator framing
- M5-016 [HIGH]: scale token Bot AUROC stabilizer
- M6-006 [HIGH]: Bot AUROC video-paradigm scope (refined by M6-012)
- M6-012 [HIGH]: 1D byte Bot AUROC partial overturns video-only scope

### §6.5 Paradigm × decision-rule pathology + eval space framing + per-class signature
- M6-001 [HIGHEST]: M6.2 max-confidence BENIGN-bias aggregation
- M6-002 [HIGH]: M6.2 Heartbleed flow vulnerability
- M6-003 [HIGH]: per-flow vs per-window eval space discontinuity
- M6-007 [MEDIUM]: Heartbleed paradigm-specific signal-capture
- M6-010 [MEDIUM]: DoS slowloris 1D byte single-attack effective
- M3-001 [HIGHEST]: VideoMAE channel preservation (pretrained-feature retention narrative)
- M2-003 [MEDIUM]: Ch 6 log-delta (training dynamics implication)
- M2-006 [HIGH]: 13-class collapsed labeling (eval space)
- M3-008 [LOW]: GC vs FP16 surprise (footnote optional)
- M4-003 [MEDIUM]: dominant-rule asymmetry
- M4-007 [MEDIUM]: total_steps × epoch_end_strategy (also §8 future work)
- M4-010b [MEDIUM]: M4.8 baseline limits (also §7 limitations)
- M5-004 [HIGH]: focal loss saturation (n=12 implication)

## §7 Limitations(~0.75 pages)

### §7.1 Single dataset (CIC-IDS-2017),cross-dataset generalization future work
**Narrative anchor**: Idea.md §7.1

### §7.2 Spatial layout ablation (维度 3) deferred to future work
**Narrative anchor**: Idea.md §7.2

### §7.3 Bot rare-class n=12 statistical edge
- M2-008 [HIGH]: rare-class disclosure mechanism (proactive)
- M6-012 [HIGH]: Bot AUROC preservation + argmax F1=0 paradigm-uniform pathology

### §7.4 No SoTA direct comparison (scope choice + label space / eval setup incomparability)
- M5-013 [HIGHEST]: SSv2 anti-evidence (corpus specificity reinforces scope acknowledgment)
- TRANSITION-008 [MEDIUM]: dominant-attack labeling rule limitations

### §7.5 Single-stream cells (dim 4 C/D) training budget confound
- M5-017 [MEDIUM]: dim 4 C/D single-stream OOD asymmetry
- M4-007 [MEDIUM]: total_steps × epoch_end_strategy coupling
- M4-010b [MEDIUM]: M4.8 baseline F1 = 0 honest disclosure

### §7.6 Some attack class absolute F1 < 0.3 (GoldenEye / Patator)
- M3-003 [LOW]: VideoMAE q/v_bias loss (implementation-related)
- M4-003 [MEDIUM]: dominant-rule asymmetry impact on labeling
- TRANSITION-004 [LOW]: CSV trailing comma (dataset quality footnote)

## §8 Future Work + Conclusion(~0.5 pages)

### §8.1 Future Work
- Cross-dataset(UNSW-NB15 / TON_IoT / CIC-IDS-2018)
- 维度 3 spatial layout ablation
- SoTA direct comparison (if methodology gap closes)
- Bot-specific data augmentation
- M4-007 [MEDIUM]: total_steps × epoch_end_strategy (achieved via M5 multi-epoch)
- M4-010b [MEDIUM]: M4.8 baseline limits (resolved via M5+ multi-epoch closure)

### §8.2 Conclusion — distill 3-4 most actionable findings
**Methodology mention candidate**:
- M5-018 [LOW]: CC stop-and-report collaborative error correction infrastructure (optional acknowledgments / methodology mention)
**Conclusion distillation**: from §6 flagship — silent failure chain + 三维耦合 + K400 init-effect + Pareto navigator framing

## References

~30-50 refs estimated;待 paper draft 阶段填

---

## Section anchor finding density check (v1 refresh)

| Section | HIGHEST | HIGH | MEDIUM | LOW | Literal Total | Aspirational add | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| §1 | 0 | 0 | 0 | 0 | 0 | 6 (Idea.md + cite-from §3/§5/§6) | Idea.md-driven adequate |
| §2 | 0 | 0 | 0 | 0 | 0 | 3 (Idea.md + M6-003/004/008) | Idea.md-driven adequate |
| §3 | 13 | 11 | 11 | 7 | 42 | — | very strong |
| §4 | 0 | 2 | 0 | 0 | 2 | +4 aspirational | strong combined |
| §5 | 2 | 1 | 0 | 0 | 3 | +7 aspirational | strong combined |
| §6 | 14 | 14 | 11 | 3 | 42 | — | very strong |
| §7 | 1 | 2 | 5 | 2 | 10 | — | strong |
| §8 | 0 | 0 | 2 | 1 | 3 | +Idea.md future-work narrative | adequate |

**Ambiguous mapping resolved (N+3 P9.5)**:
- 3 dropped from paper anchors (M3-004, M4-006, M4-008 — Implementation Notes, no paper exposure)
- 3 mapped to §3.6 Reproducibility (M2-009, M4-009, TRANSITION-002 — Computational Efficiency)

每 section 有 anchor findings 锚定 narrative,**N+5 paper drafting 不会 anchor-less**。
