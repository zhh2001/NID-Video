# Findings.md — 项目实证发现档案

> 这份文档持续累积本项目从代码实现中产生的、对论文写作有真实价值的发现：实证数据、设计决策、工程陷阱、反直觉结果。
>
> 它不是 README、不是 changelog、不是 design doc。它是**论文素材库**。
>
> **位置**：项目 `prompts/` 目录下，与其他人机协作工具一起，本地存在但不入仓库。

---

## 如何使用这份文档

### 维护规则

- **事实层**（无 blockquote）：由 Claude Code 在每个里程碑结束时按模板追加。只记录 *what* 和 *how*。
- **论文价值层**（blockquote 标 🎯）：由 Claude conversation（即设计 / 评审 LLM 角色）在多个里程碑后批量标注。回答 *why this matters for the paper*。
- 任何人不得删除已有条目，只能追加 / 修订。
- 修订时保留原条目，新增 "**Updated**:" 段说明变更原因。

### 论文写作时的查询方式

```bash
# 找 Method 章节素材
grep -A 5 "Section: Methods" Findings.md

# 找 Discussion / Limitation 素材
grep -A 5 "Section: Discussion" Findings.md
grep -A 5 "Section: Limitation" Findings.md

# 找 figure 素材
grep -B 1 "Figure idea" Findings.md

# 找数据集预处理陷阱（Methods/Reproducibility 用）
grep -i "preprocessing\|pitfall\|footgun" Findings.md

# 列出所有发现编号
grep -E "^### Finding [A-Z0-9-]+" Findings.md

# 按 Priority 级别筛选
grep -A 1 "Priority\*\*: HIGHEST" Findings.md
grep -A 1 "Priority\*\*: HIGH$" Findings.md
```

### 模板（Claude Code 追加事实层时用）

```markdown
### Finding <ID>: <短标题>
- **Context**: <哪个里程碑/任务/文件触发的>
- **Discovery**: <事实陈述 + 关键数字>
- **Evidence**: <测试名 / 日志位置 / commit hash>
- **Decision rationale**: <为什么这么做，引用提示词或讨论位置>
- **Status**: <persisted in CI / documented only / pending followup>
```

ID 规则：`<里程碑>-<3位序号>`，例如 `M2-001`、`M3-005`、`TRANSITION-002`。

### 论文价值层标注模板（Claude conversation 追加 🎯 段时用）

```markdown
> 🎯 **论文价值标注**
> - **Section**: <对应论文章节，如 "Methods §3.X" / "Discussion" / "Limitations" / "Reproducibility">
> - **Use**: <这条 finding 在该章节怎么用，1-2 句具体说明>
> - **Figure idea**: <如适用，描述潜在 figure 内容；不适用则省略此字段>
> - **Quote candidate**: <如适用，1-2 句可直接用作论文段落的句子>
> - **Risk if missed**: <不写这条 reviewer 可能怎么质疑>
> - **Cross-link**: <与其他 findings 的关联引用>
> - **Priority**: <HIGHEST / HIGH / MEDIUM / LOW>
```

**Priority 标注规则**：
- HIGHEST: 论文核心 figure / 核心 quote 候选、不放就缺关键 evidence
- HIGH: Methods/Discussion 章节必引、对论文可信度有显著贡献
- MEDIUM: 对应章节有用素材、可选放
- LOW: 实现细节、不预期论文出口、维护笔记

### Cross-link convention

> **Cross-link convention**: Cross-link fields use two modes by design:
>
> - **Formal list**: leading list of finding IDs forms a directional anchor graph (downstream → anchor, non-bidirectional). Separators " / " and " + " are both accepted by design (" / " for list-form references, " + " for additive / supersedes-link references). Version-suffixed references (e.g. "M5-005 v2") point to specific superseded entries and are semantically distinct from the current-version entry; they are NOT self-references. An anchor finding is referenced by multiple downstream findings without reciprocal back-links unless the anchor's own narrative materially depends on the downstream finding (hub-spoke pattern).
> - **Narrative tail**: prose after the formal list (often with CJK separators 、 ； → etc.) may inline-reference other findings as explanatory context. These mentions are intentional prose, not formal graph edges; they do not participate in the formal cross-link graph or its audit.

---

# M1 — 项目骨架阶段

（这个阶段主要是工程搭建，论文价值产出较少。完整记录见 git log。）

### Finding M1-001: WSL2 + uv + CUDA 13 环境的可行性
- **Context**: M1 任务 1.1-1.5
- **Discovery**: 在 RTX 4060 Laptop (8GB) + WSL2 Ubuntu + uv + CUDA 13.0 上完整跑通项目骨架，72 个测试 + smoke test 全绿。
- **Evidence**: tests/test_smoke.py 4 测试 + commit M1 (`uv-managed Python 3.10 project ...`)
- **Decision rationale**: prompts/00_master_prompt.md "硬件" 段约束
- **Status**: persisted in README

> 🎯 **论文价值标注**
> - **Section**: Reproducibility / Computational Considerations
> - **Use**: 论证贡献点 #6（消费级硬件可复现性）的环境准入证据——展示完整 pipeline 在 RTX 4060 Mobile 8GB + WSL2 + uv 上跑通，对应 Idea.md §4.4 "laptop-class GPU 即可完成全量训练 + real-time 推理"叙事。
> - **Quote candidate**: "The full pipeline—data ingestion, training, and evaluation—runs on a single consumer-grade laptop GPU (RTX 4060 Mobile, 8 GB VRAM) under WSL2, with peak training memory of 485 MB on the production configuration."
> - **Risk if missed**: Reviewer 质疑"消费级硬件可复现性"是否仅是事后 claim 还是设计阶段就贯彻——这条作为 M1 阶段就建立的环境基线提供时间线证据。
> - **Cross-link**: M3-007（显存假设修正实测）、M3-009（吞吐饱和点）、M3-010（WSL2 B=1024 hard limit）、M4-010a（M4.8 实测 485 MB peak）共同构成"消费级硬件可复现"完整证据链。
> - **Priority**: MEDIUM

---

# M2 — 数据 ETL 阶段

### Finding M2-001: 逐窗 k-means 而非全局 k-means（防标签泄露）
- **Context**: M2 任务 2.5（IP 聚类）
- **Discovery**: 选择"每个滑窗 1.6s 内独立做 k-means" 而非"用全数据集训一个全局 k-means"。原因是全局聚类会让攻击者 IP 因训练时见过而被分到固定行，泄露标签信息。
- **Evidence**: src/nid_video/data/ip_clustering.py 模块顶部 4 行 docstring + tests/test_ip_clustering.py::test_determinism
- **Decision rationale**: 在 M2 准备阶段由对话式评审拍板（与 Claude Code 的 RFC 讨论后）
- **Status**: persisted in code + CI

> 🎯 **论文价值标注**
> - **Section**: Methods §3.2 "Spatial Layout — H-axis Construction" + Reproducibility
> - **Use**: 支撑贡献点 #2（语义感知空间布局）。强调设计不仅工程合理，也主动规避了 NID 领域常见的 label leakage 陷阱——全局聚类会让攻击者 IP 因训练时见过而被分到固定行，泄露标签信息。
> - **Quote candidate**: "We perform k-means clustering independently within each 1.6-second window rather than globally across the dataset; the latter would cause attacker IPs—seen during training—to consistently occupy fixed rows at test time, leaking class identity through spatial position."
> - **Risk if missed**: Reviewer 或 reproducer 可能采用全局聚类得出虚高指标；不预先声明会被质疑"你们的 H 轴布局是否在测试时利用了训练阶段见过的 IP"。
> - **Cross-link**: M2-006（双轨标签策略）、M2-008（极稀疏类披露）共同构成"诚实数据处理"方法学簇；M2-004（W 轴 min(src,dst) 对称）配套构成完整空间布局设计。
> - **Priority**: HIGH

### Finding M2-002: TCP flags 6 位 vs 4 位的取舍
- **Context**: M2 任务 2.4 Q1
- **Discovery**: 通道 3 (TCP flags) 选 6 位（SYN/ACK/FIN/RST/PSH/URG）而非 4 位。理由：PSH 在数据渗漏检测有信号，URG 在某些 botnet 心跳里有信号。
- **Evidence**: src/nid_video/data/channels.py docstring "Decision: M2 task 2.4 Q1"
- **Decision rationale**: 与 Claude conversation 评审讨论后拍板
- **Status**: persisted in code

> 🎯 **论文价值标注**
> - **Section**: Methods §3.2 "Channel Encoder Design"
> - **Use**: Channel 3 (TCP flags) 6-bit 选择的具体决策依据，避免论文 reviewer 质疑"为什么不用 4-bit 节省维度"
> - **Quote candidate**: "We retain all 6 TCP flag bits (SYN/ACK/FIN/RST/PSH/URG) rather than the more common 4-bit subset, as PSH carries data exfiltration signal and URG appears in certain botnet heartbeat patterns."
> - **Risk if missed**: Methods 章节维度选择缺乏 rationale 解释
> - **Cross-link**: M2-003 (log-delta motion channel) / M2-004 (W-axis port symmetry) / M2-005 (1/3-octave bucket) — 共同构成"6 通道编码学合理性"论文素材
> - **Priority**: MEDIUM

### Finding M2-003: Ch 6 用 log-delta 而非 raw delta
- **Context**: M2 任务 2.4 Q2
- **Discovery**: 通道 6（帧间包数差分）用 `log(1+N(t)) - log(1+N(t-1))` 而非 `N(t) - N(t-1)`。raw delta 在 DDoS 突发场景下值会飙到 1000+，与其他通道的 [0,10] 量级失衡，**梯度被这个通道支配**。
- **Evidence**: src/nid_video/data/channels.py docstring + tests/test_channels.py 中针对 DDoS 突发的测试
- **Decision rationale**: 量纲分析驱动的设计决策
- **Status**: persisted in code + CI

> 🎯 **论文价值标注**
> - **Section**: Methods §3.2 "Channel Encoder" + Discussion (Training Dynamics)
> - **Use**: 对应 Idea.md §3.2 通道 6（帧间包数差分）的设计细化。raw delta 在 DDoS 突发场景下值会飙到 1000+ 与其他通道 [0,10] 量级失衡，**梯度被这个通道支配**——这是运动通道设计中的训练动力学考量。
> - **Figure idea**: Ablation 实验 figure（如做这个消融）：raw delta vs log delta 的训练 loss 曲线对比，预期 raw delta 早期 loss 被运动通道梯度劫持。
> - **Quote candidate**: "We apply log-scale to the inter-frame packet-count delta channel: raw differences during DDoS bursts can exceed 10³, dominating the gradient of all other channels which sit in the [0, 10] range."
> - **Risk if missed**: Reviewer 质疑运动通道是否真在协同其他通道训练（而非"一通道独大"）；不预先声明会被怀疑训练动力学未经审视。
> - **Cross-link**: M2-002（flags 6 位）、M2-004（min(src,dst)）同属 6 通道设计的逐项决策；与 Idea.md §3.2 通道表对应。
> - **Priority**: MEDIUM

### Finding M2-004: Port 列查表用 min(src,dst) 保连接对称
- **Context**: M2 任务 2.4 Q3
- **Discovery**: W 列查表用 `min(src_port, dst_port)` 而非 dst_port。原因：HTTP 通信里 client(54321)→server(80) 和 server(80)→client(54321) 都映射到第 0 列。**同一连接的所有包都落在同一 W 列**，DDoS / 扫描的视觉运动模式才能保持完整。
- **Evidence**: src/nid_video/data/channels.py `port_to_col` + tests
- **Decision rationale**: 视觉运动模式的语义保持
- **Status**: persisted in code

> 🎯 **论文价值标注**
> - **Section**: Methods §3.2 "Spatial Layout — W-axis Construction"
> - **Use**: 支撑贡献点 #1（视频化表征）的核心论证——空间布局必须保持视觉运动连续性。HTTP 通信里 client(54321)→server(80) 和 server(80)→client(54321) 都映射到第 0 列，**同一连接的所有包都落在同一 W 列**，DDoS / 扫描的视觉运动模式才能保持完整。这是与 1D/2D 表征本质差异的具体体现。
> - **Quote candidate**: "Each W-axis column maps via min(src_port, dst_port), ensuring all packets of a bidirectional connection occupy the same column; this preserves the visual continuity required for motion patterns—port-scan trajectories or DDoS column intensification—to be encoded coherently across frames."
> - **Risk if missed**: Reviewer 质疑"W 轴的语义是什么、为什么不直接用 dst_port"——这条提供 W 轴构造的语义依据，并直接呼应 Idea.md §1.2 "运动 = 行为变化率"立意。
> - **Cross-link**: M2-001（H 轴语义聚类）共同构成空间布局设计；与 Idea.md §1.2 立意核心呼应；M2-005（log 桶 1/3-octave）是 W 轴的子设计。
> - **Priority**: HIGH

### Finding M2-005: Port log 桶用 1/3-octave 细分（避免 31 列空置）
- **Context**: M2 任务 2.4 Q4
- **Discovery**: W 维 48 列 log 桶用 `int(log2(port+1) * 3)` 而非纯 `int(log2(port))`。后者只用 17/48 列，剩 31 列永远为 0；前者填满 48 列，分辨率更细。
- **Evidence**: src/nid_video/data/channels.py `port_to_log_bucket`
- **Status**: persisted in code

> 🎯 **论文价值标注**
> - **Section**: Methods §3.2 "Spatial Layout — W-axis Construction" (子细节)
> - **Use**: W 轴 log 桶的实现细节——`int(log2(port+1) * 3)` 填满 48 列而非 17 列，避免分辨率浪费。属于 implementation detail，不预期独立论文出口，但在 Methods 章节简短一句即可避免 reviewer 反问。
> - **Risk if missed**: Reviewer 关注空间分辨率利用率时可能反问"48 列是否真有效利用"——一句话即可回应。
> - **Cross-link**: M2-004（min(src,dst) 对称）的同章节子设计。
> - **Priority**: MEDIUM

### Finding M2-006: 15 类原始 + 13 类合并双轨标签策略
- **Context**: M2 任务 2.6
- **Discovery**: webdataset shard 存 15 类原始 ID，训练阶段提供 `label_mode='raw15'/'collapsed13'` 配置。collapsed13 把 Web Attack 三子类合并以兼容 CIC-IDS 2018，raw15 保留细粒度。
- **Evidence**: src/nid_video/data/labeling.py `LABEL_TO_ID_RAW` / `LABEL_TO_ID_COLLAPSED` / `collapse_to_13()`
- **Decision rationale**: 同一份 ETL 输出支持两条实验路径，**消除 raw15 vs collapsed13 对比中"是否预处理引入差异"的可能性**
- **Status**: persisted in code + CI

> 🎯 **论文价值标注**
> - **Section**: Experimental Setup + Methods §3.5 "Outputs" + Discussion
> - **Use**: 主实验报告 collapsed13（与 baseline 论文可比性），消融报告 raw15（验证细粒度区分能力反成加分点）。**消除 raw15 vs collapsed13 对比中"是否预处理引入差异"的可能性**——同一份 ETL 输出支持两条实验路径。
> - **Quote candidate**: "Our ETL output retains the 15-class raw labels; the 13-class collapsed scheme (merging Web Attack subtypes for compatibility with CIC-IDS 2018) is applied at training time via configuration, ensuring raw15 and collapsed13 results are derived from identical preprocessed tensors."
> - **Risk if missed**: Reviewer 质疑跨数据集泛化（CIC-2017 → CIC-2018）的标签兼容性是否引入预处理差异——这条提供 schema-level 兼容证据。
> - **Cross-link**: M2-008（极稀疏类披露）配套——collapsed13 不只是 2018 兼容，也是规避稀疏类（如 Web Attack 三子类合并后样本量恢复）拖低 macro-F1 的合理设计；M2-001（label leakage 防御）同属"诚实数据处理"方法学簇。
> - **Priority**: HIGH

### Finding M2-007: CIC-IDS 标签字段的 EN-DASH (U+2013) vs ASCII hyphen
- **Context**: M2 任务 2.6
- **Discovery**: CIC-IDS 标签字符串 "Web Attack – Brute Force" 里的破折号是 U+2013 EN DASH，不是 ASCII hyphen。直接用 ASCII 比对会全部匹配失败。
- **Evidence**: src/nid_video/data/labeling.py `normalize_label_name` + tests/test_labeling.py 多个测试
- **Status**: persisted in code + CI

> 🎯 **论文价值标注**
> - **Section**: Reproducibility / Dataset Preprocessing Pitfalls
> - **Use**: 实现陷阱清单的一项——CIC-IDS 标签字符串 "Web Attack – Brute Force" 里的破折号是 U+2013 EN DASH，按 ASCII hyphen 比对会全部匹配失败。
> - **Risk if missed**: 复现者（或 reviewer 自己复现 baseline）可能踩同一个坑得出"Web Attack 三子类全 0"的错误结果。
> - **Cross-link**: TRANSITION-003（cp1252 编码与 EN-DASH 字节同源 0x96）、TRANSITION-001（pcapng）、TRANSITION-004（CIC 工具 bug）共同构成 "CIC-IDS 2017 dataset preprocessing pitfalls" subsection。
> - **Priority**: MEDIUM

### Finding M2-008: 极稀疏攻击类的诚实披露机制
- **Context**: M2 任务 2.7
- **Discovery**: CIC-IDS 2017 的 Heartbleed 类全集只有约 11 个流，SQL Injection 21 行（修复 cp1252 编码后），切窗后实际可学样本极少。ETL 输出 stats 时会主动 warning "< 50 samples after windowing, statistically unlearnable"。
- **Evidence**: src/nid_video/data/labeling.py `warn_low_population_classes` + 测试
- **Status**: persisted in code

> 🎯 **论文价值标注**
> - **Section**: Limitations / Dataset Description
> - **Use**: 主动声明数据集稀疏性，避免审稿人质疑 macro-F1 计算合理性。CIC-IDS 2017 的 Heartbleed 类约 11 个流、SQL Injection 21 行（修复 cp1252 编码后）——切窗后实际可学样本极少，ETL 输出 stats 时主动 warning。
> - **Quote candidate**: "Heartbleed and SQL Injection in CIC-IDS 2017 contain approximately 11 and 21 flows respectively; we explicitly flag classes below the statistically learnable threshold (< 50 windows post-tube-patching) in the ETL output rather than silently include them in macro-F1."
> - **Risk if missed**: 主流论文常忽略稀疏类对 macro-F1 的扰动，导致数字虚高/虚低；预先披露反而是方法学严谨性加分点。
> - **Cross-link**: M2-006（双轨标签策略——collapsed13 缓解 Web Attack 子类稀疏性）、M2-001（label leakage 防御）共同构成"诚实数据处理"方法学簇；M4-010b（Bot/GoldenEye F1=0/AUROC>0.7 豁免规则）是稀疏类下游表现的具体体现。
> - **Priority**: HIGH

### Finding M2-009: pcap 解析速度 dpkt 上限 ~138k pps（pure Python 现实）
- **Context**: M2 任务 2.2
- **Discovery**: 用 dpkt 做 pure Python pcap 解析的实际上限是 ~138k pps（dpkt 内核解析的 ceiling）。我们的 PacketStream 封装达到 127k pps（92% 效率）。**M2 提示词原本写"≥ 50万 pps"是错误估算**，已校正到 ≥ 12 万 pps。
- **Evidence**: docs/etl_performance.md L1-L4' benchmark 表
- **Decision rationale**: 拒绝引入 C 扩展（pylibpcap / Rust）—— 5.5 min 的全 ETL 解析对一次性预处理可接受
- **Status**: persisted in docs

> 🎯 **论文价值标注**
> - **Section**: Implementation / Computational Efficiency
> - **Use**: ETL 时长报告的依据数字——L2 ceiling 138k pps、`PacketStream` 实测 127k pps（92% 效率）、全 ETL 端到端 18k pps / 22 win/s / 46 ms per window。在 efficiency table 报告 ETL 时长时引用此基线。
> - **Quote candidate**: "Our pure-Python pcap parser based on dpkt achieves 127k pps (92% of dpkt's L2 ceiling of 138k pps); end-to-end ETL throughput is 22 windows/s, processing the Tue+Wed+Fri subset in 71 minutes with three workers."
> - **Risk if missed**: Reviewer 质疑预处理时长是否合理（CIC-IDS 全集 ETL 71 分钟）——这条提供 stage-level 性能 breakdown。
> - **Cross-link**: TRANSITION-002（pcapng 慢 45%、71k pps vs 138k pps）共同构成 ETL 性能档案；M4-009（gzip 压缩率诊断）是磁盘 IO 侧的对应分析。
> - **Priority**: MEDIUM

---

# M3 — 主方法 v0 阶段

### Finding M3-001: VideoMAE 预训练通道保留 5.28× norm ratio（**项目核心实证**）
- **Context**: M3 任务 3.3 patch_embed 适配
- **Discovery**: patch_embed 重建后，前 3 通道（trilinear 下采样自预训练 RGB）权重 norm = 5.238，后 3 通道（Kaiming 新初始化的运动通道）norm = 27.665，**ratio 5.28×**。证明预训练权重确实保留下来，没被 `ignore_mismatched_sizes=True` 静默丢失。
- **Evidence**: tests/test_videomae_nid.py::test_real_pretrained_ch_0_3_norm_smaller_than_ch_3_6（断言 ratio > 1.5×，CI 持续守护）+ M3.3 完成报告日志
- **Decision rationale**: M3.3 探索阶段发现 `ignore_mismatched_sizes=True` 会让两半 norm 几乎相等（15.375 vs 15.371），主动绕开
- **Status**: persisted in CI test (regression-safe)

> 🎯 **论文价值标注**
> - **Section**: Methods §3.4 "Backbone Adaptation" + Discussion (Pretrained Transfer Verification)
> - **Use**: **整个项目立意（"VideoMAE 预训练运动语义可迁移到流量行为"）的代码层硬证据**。前 3 通道（trilinear 下采样自预训练 RGB）权重 norm = 5.238，后 3 通道（Kaiming 新初始化的运动通道）norm = 27.665，ratio = 5.28×。这不是间接推论（如下游 F1 提升），而是权重层面的直接验证——预训练权重确实保留下来，没被 silent killer 抹除。
> - **Figure idea**: 双柱状图：横轴是两种加载方式（"我们的方法 with trilinear downsampling preservation" vs "`ignore_mismatched_sizes=True`"），纵轴是 patch_embed 前 3 通道 vs 后 3 通道的权重 norm。我们这一侧 5.28× ratio 显著，对照侧 ratio≈1（15.375 vs 15.371）说明预训练完全丢失。这是一张读者一眼就能看懂"预训练真的迁移了"的 figure。
> - **Quote candidate**: "We empirically verify that pretrained appearance channels are preserved through patch-embedding adaptation: the L2 norm ratio between pretrained-derived channels (5.238) and Kaiming-initialized motion channels (27.665) is 5.28×, an order-of-magnitude separation that would not exist under fresh initialization."
> - **Risk if missed**: Reviewer 质疑"你们怎么证明 Kinetics 预训练真的迁移了，而不是 fine-tuning 期间任意 22M 参数都能拟合 CIC-IDS"——没有这条权重层面证据，立意核心论点只能依赖下游 F1 这种间接信号，论证强度大幅下降。
> - **Cross-link**: M3-002（`ignore_mismatched_sizes=True` 静默失败陷阱）形成"实证 + 陷阱"叙事对——M3-001 是正面证据、M3-002 是反例对照；M3-003（q/v_bias 丢失）、M3-004（HF 命名风格）、M3-005（pos enc 非 buffer）、M3-006（16 头不是 6 头）共同构成"VideoMAE 预训练适配的实现细节链路"，但 M3-001 是其中唯一具有论文 figure 出口的旗舰证据。
> - **Priority**: HIGHEST

### Finding M3-002: `ignore_mismatched_sizes=True` 静默丢失预训练权重（陷阱发现）
- **Context**: M3 任务 3.3 探索阶段
- **Discovery**: 用 `from_pretrained(num_channels=6, ignore_mismatched_sizes=True)` 加载 VideoMAE-S 后，patch_embed 前 3 通道与后 3 通道的 norm 几乎相等（15.375 vs 15.371）——证明前 3 通道的预训练权重**完全没被保留**，整个 patch_embed 都是 fresh init。
- **Evidence**: M3.3 探索报告（保存在对话历史，未持久化到文件）
- **Decision rationale**: 这是项目最大卖点（"VideoMAE 预训练迁移"）的潜在 silent killer
- **Status**: documented only（探索性代码已抛弃；防御措施在 M3-001 测试里）

> 🎯 **论文价值标注**
> - **Section**: Discussion (Implementation Footguns) + Reproducibility
> - **Use**: 警告社区——大部分基于 transformers 库的视频迁移工作可能在不知情中踩这个坑。`ignore_mismatched_sizes=True` 抑制 shape error 的代价是 **fresh-init 整个 patch_embed**（前 3 通道 norm 15.375 vs 后 3 通道 15.371，几乎相等说明预训练完全丢失），而这个失败模式在 inference 时不可见，仅通过权重 norm 检查可发现。
> - **Quote candidate**: "We caution that the convenient `ignore_mismatched_sizes=True` flag in the transformers library, while suppressing shape errors, silently discards all weights of the mismatched layer—including channels that could have been preserved via downsampling. This failure mode is invisible at inference time and only detectable through weight-norm inspection."
> - **Risk if missed**: 不写这条 → 论文里"我们做了 trilinear 下采样保留前 3 通道"读起来像无端的工程繁琐；写这条 → 同样的代码块变成"我们绕开了一个静默 footgun，并提供了可观察的验证手段（M3-001）"。这是工程严谨性叙事的关键一环。
> - **Cross-link**: M3-001（5.28× norm ratio 实证）形成"陷阱 + 防御"叙事对——M3-002 提出问题、M3-001 提供检测手段（CI-pinned regression test）。
> - **Priority**: HIGHEST

### Finding M3-003: VideoMAE q/v_bias 在 transformers 5.x 中丢失
- **Context**: M3 任务 3.3 探索 + 实现
- **Discovery**: VideoMAE 论文实现用 `q_bias` / `v_bias` / 无 `k_bias` 格式，transformers 5.x 用标准 `query.bias` / `key.bias` / `value.bias` 三件套。5.x 不做自动重映射，**导致 ~14k bias 参数（22M 总量的 0.06%）被 fresh-zero 而非加载预训练值**。
- **Evidence**: src/nid_video/models/videomae_nid.py docstring 留诊断线索 + LOAD REPORT 里的 MISSING 列表
- **Decision rationale**: 占比极小，决定接受不补救（M3 阶段）。如未来 M5 baseline 比对发现 Kinetics pretrained < ImageNet pretrained 这种异常信号，第一嫌疑就是这个
- **Status**: documented in code docstring

> 🎯 **论文价值标注**
> - **Section**: Limitations / Implementation Notes
> - **Use**: 实现透明度声明——VideoMAE 论文实现用 `q_bias` / `v_bias` 格式，transformers 5.x 用标准 `query.bias` / `key.bias` / `value.bias` 三件套，**5.x 不做自动重映射，~14k bias 参数（22M 总量的 0.06%）被 fresh-zero**。占比极小决定接受不补救。
> - **Risk if missed**: 如 M5 baseline 比对发现 Kinetics pretrained 表现意外低于 ImageNet pretrained（或随机初始化），这条是第一嫌疑。预先披露建立 audit trail。
> - **Cross-link**: M3-001（核心预训练验证）、M3-004（HF 命名风格）、M3-006（16 头 attention）共同构成 "VideoMAE 预训练加载的实现细节链路"; M4-008 (layernorm=None corner case) — 同属 HF transformers 5.x VideoMAE adaptation footgun cluster
> - **Priority**: LOW

### Finding M3-004: VideoMAE-Small 预训练权重命名是 HF 嵌套风格
- **Context**: M3 任务 3.3 探索
- **Discovery**: state_dict key 是 `embeddings.patch_embeddings.projection.weight`（HF 嵌套）而非 `patch_embed.proj.weight`（timm 风格）。按 timm 命名找属性会全部 KeyError。
- **Evidence**: M3.3 探索报告
- **Status**: documented only（已被 M3-001 实现间接守护）

> 🎯 **论文价值标注**
> - **Section**: Implementation Notes (no paper exposure expected)
> - **Use**: 维护笔记。state_dict key 是 `embeddings.patch_embeddings.projection.weight`（HF 嵌套）而非 `patch_embed.proj.weight`（timm 风格），按 timm 命名找属性会全部 KeyError。论文一般不会写到这种 framework-specific 细节，但代码层已被 M3-001 测试间接守护。
> - **Cross-link**: M3-003（q/v_bias）、M3-005（pos enc 非 buffer）、M3-006（16 头）共同构成 HF VideoMAE 适配陷阱簇; M4-008 (layernorm=None corner case) — 同属 HF transformers 5.x VideoMAE adaptation footgun cluster
> - **Priority**: LOW

### Finding M3-005: VideoMAE 位置编码是非可学正弦表（非 Parameter 也非 buffer）
- **Context**: M3 任务 3.3 探索
- **Discovery**: VideoMAE 的位置编码是 `get_sinusoid_encoding_table` 生成的非可学张量，**不是 Parameter 也不是 buffer**。所以从 14×14 网格切换到 4×8 网格时**不需要 trilinear 插值，直接用 transformers 内置函数重新计算更干净**（数学等价但实现更简洁）。
- **Evidence**: src/nid_video/models/videomae_nid.py `_adapt_position_embedding`
- **Status**: persisted in code

> 🎯 **论文价值标注**
> - **Section**: Methods §3.4 Backbone Adaptation (Implementation Notes)
> - **Use**: 维护笔记。VideoMAE pos enc 是 `get_sinusoid_encoding_table` 生成的非可学张量，从 14×14 网格切到 4×8 网格时**不需要 trilinear 插值，直接用 transformers 内置函数重新计算更干净**（数学等价）。属于实现简化决策，不预期论文出口。
> - **Cross-link**: M3-003 / M3-004 / M3-006 同属 HF VideoMAE 适配陷阱簇; M4-008 (HF transformers 5.x VideoMAE adaptation footgun cluster)
> - **Priority**: MEDIUM

### Finding M3-006: VideoMAE-Small 用 16 头 attention（不是论文的 6 头）
- **Context**: M3 任务 3.3 探索
- **Discovery**: 公开 `MCG-NJU/videomae-small-finetuned-kinetics` 的 config 用 `num_attention_heads=16`，head_dim=24。这与 VideoMAE 论文的 6 头 ViT-S 不同。如果 fallback 路径用 `VideoMAEConfig()` 默认值会拿到 768 hidden 的 Base config，必须显式构造 Small config 对齐 ckpt。
- **Evidence**: src/nid_video/models/videomae_nid.py `_videomae_small_config()` + 测试
- **Status**: persisted in code + CI

> 🎯 **论文价值标注**
> - **Section**: Implementation Notes / Reproducibility
> - **Use**: 维护笔记 + 复现陷阱。公开 `MCG-NJU/videomae-small-finetuned-kinetics` 的 config 用 `num_attention_heads=16`，head_dim=24（与 VideoMAE 论文 ViT-S 的 6 头不同）。fallback 路径用 `VideoMAEConfig()` 默认值会拿到 768-hidden 的 Base config，必须显式构造 Small config 对齐 ckpt。
> - **Risk if missed**: 复现者按论文 ViT-S 规格构造 config 加载 HF ckpt 会 fail；这条是 reproducibility footgun。
> - **Cross-link**: M3-003 / M3-004 / M3-005 同属 HF VideoMAE 适配陷阱簇; M4-008 (layernorm=None corner case) — 同属 HF transformers 5.x VideoMAE adaptation footgun cluster
> - **Priority**: LOW

### Finding M3-007: 显存假设根本错误（Idea.md 该改）
- **Context**: M3 任务 3.5 显存压测
- **Discovery**: Idea.md 之前写"为了挤进 8GB 选 batch=2 + grad_accumulation=16"。**实测 batch=2 峰值仅 290 MB，max-safe batch=512（4.5 GB）**。整个 base.yaml 的 batch 配置是基于错误估算。
- **Evidence**: docs/m3_perf.md Phase 1 batch sweep 表
- **Decision rationale**: 立意层面正确（视频范式 + 异构时空），但工程参数依据被推翻；**Idea.md §4 待 M4 真数据后做编辑性修订**
- **Status**: documented in m3_perf.md, deferred fix

> 🎯 **论文价值标注**
> - **Section**: Reproducibility / Computational Considerations + Idea.md editorial revision
> - **Use**: 论文 hardware section 叙事修订的关键依据——Idea.md 早期 "为了挤进 8GB 选 batch=2 + grad_accumulation=16" 的"内存优先"叙事被实测推翻：B=2 仅 290 MB，max-safe B=512（4.5 GB），B=128 仅 1.2 GB。论文不应写"我们受 8GB 显存约束"，而应写"在 8GB 卡上有大量余量、瓶颈是吞吐而非内存"。这条是 m3_perf.md 顶部 reframing（"the relevant question shifted from 'do we fit?' to 'what configuration maximizes throughput given memory is abundant?'"）的 finding-level 锚点。
> - **Quote candidate**: "Contrary to our initial design assumption, the input volume after tube-patch downsampling (256 tokens per sample versus VideoMAE-B's 1568) reduces activation memory by ~6×; on an 8 GB consumer GPU we measure peak training memory at 485 MB for the production configuration, with B=512 still leaving 3.5 GB headroom."
> - **Risk if missed**: 论文里把"8GB 显存约束"当成挑战叙事 → reviewer 验算后发现 memory 远未饱和 → "你们的硬件叙事不诚实"。修正后反过来强化贡献点 #6（消费级硬件可复现性）：不是"勉强能装下"，而是"laptop GPU 上有充裕余量、推动可复现性"。
> - **Cross-link**: M3-008（Gradient Checkpointing 是 4× 显存大头）、M3-009（吞吐饱和点 B=128-256）、M3-010（WSL2 B=1024 hard limit）共同构成 m3_perf.md "memory abundant, throughput is the constraint" reframing 三件套；M1-001（环境基线）的下游量化证据；M4-010a（M4.8 实测 485 MB peak）是 M3 估算到真数据训练的最终验证。
> - **Priority**: HIGH

### Finding M3-008: Gradient Checkpointing 是显存大头（4×），其他优化加起来仅 ~130 MB
- **Context**: M3 任务 3.5 ablation
- **Discovery**: B=32 时 GC ON vs OFF: 426 MB vs 1735 MB（**4×**）。FP16 vs FP32: 426 vs 554（128 MB 差）。8-bit AdamW vs 32-bit: 426 vs 553（127 MB 差，**24%** 而非教科书的 75%）。**模型越小 8-bit 收益越小**。
- **Evidence**: docs/m3_perf.md Phase 2 ablation 表
- **Status**: persisted in docs

> 🎯 **论文价值标注**
> - **Section**: Discussion (Counterintuitive Findings) + Implementation Notes
> - **Use**: 反直觉实证——B=32 时 GC ON vs OFF: 426 MB vs 1735 MB（**4.07×**）；FP16 vs FP32 仅差 128 MB；8-bit AdamW vs 32-bit 仅差 127 MB（22M 模型上是 24% 而非教科书的 75%，因为 8-bit 只压缩优化器状态不压缩参数本身）。值得讨论的"小模型 + 教科书优化策略"实证。
> - **Quote candidate**: "Gradient checkpointing dominates the memory budget on our 22M-parameter model: at B=32, disabling it raises peak memory 4.07× (426 → 1735 MB), whereas 8-bit AdamW saves only 127 MB (24% of optimizer state, far below the ~75% reported for billion-parameter LLMs, since 8-bit quantization compresses optimizer state but not parameters themselves)."
> - **Risk if missed**: Reviewer 质疑"为什么不去掉 GC 加速训练"——不写这条则缺乏 4× 显存代价的量化依据。
> - **Cross-link**: M3-007（显存假设修正）、M3-009（吞吐饱和点）共同构成显存/吞吐 reframing 三件套；与 Idea.md §4.1 显存节省组合拳表对应。
> - **Priority**: LOW

### Finding M3-009: 吞吐饱和点在 batch=128-256，base.yaml 配置慢 12×
- **Context**: M3 任务 3.5 throughput
- **Discovery**: 吞吐随 batch 增长：B=2 → 24 sps; B=32 → 203 sps; B=128 → 217 sps; B=256 → 222 sps（饱和）; B=512 → 211 sps（开始下降）。当前 base.yaml 的 B=2/accum=16 配置吞吐 24 sps，比最优配置慢 **12×**。
- **Evidence**: docs/m3_perf.md Phase 3 + Phase 4
- **Decision rationale**: 创建 configs/training_perf.yaml 承载新最优配置，base.yaml 保持稳定基线（CI/smoke 用）
- **Status**: persisted in configs

> 🎯 **论文价值标注**
> - **Section**: Reproducibility / Computational Cost
> - **Use**: 单 epoch 时长报告依据——B=128/workers=4 = 283 sps（最佳），base.yaml B=2/accum=16 = 24 sps（慢 12×）。500K-sample epoch wall time: 5.8h → 46min → 29min（取决于配置）。论文 efficiency table 报告训练时长时引用此基线。
> - **Quote candidate**: "Training throughput plateaus at B=128–256 (~220 samples/sec on GPU compute; 283 samples/sec end-to-end with `num_workers=4`); the original B=2 / accumulation=16 configuration—designed under a now-disproven memory-constrained assumption—runs 12× slower than necessary."
> - **Risk if missed**: 论文 efficiency table 的训练时长数字需要可解释依据；不预先建立吞吐档案则数字像"运气数字"。
> - **Cross-link**: M3-007（显存假设修正——base.yaml 慢 12× 的根因是错误显存约束）、M3-008（GC 是显存大头）共同构成显存/吞吐 reframing 三件套；M1-001 / M3-010 / M4-010a 共同构成"消费级硬件可复现"完整证据链。
> - **Priority**: HIGH

### Finding M3-010: WSL2 NVIDIA 驱动在 batch=1024 会 hang，需 kill -9 + 1min cleanup
- **Context**: M3 任务 3.5 batch sweep 边界探测
- **Discovery**: 实测 B=1024 让 WSL2 NVIDIA 驱动卡住 7.8GB / 100% util / 无进程归属，需要 kill 后等 1 分钟 + soft cleanup。**HARD LIMIT B ≤ 512** 即使 peak_mem 报告显示有余量。
- **Evidence**: docs/m3_perf.md 顶部 ⚠️ HARD LIMIT 警告框
- **Status**: persisted in docs

> 🎯 **论文价值标注**
> - **Section**: Implementation Notes / Reproducibility footguns
> - **Use**: 给在 WSL2 复现工作的读者的硬件 footgun 警告——B=1024 让 WSL2 NVIDIA 驱动卡住 7.8GB / 100% util / 无进程归属，需 kill -9 + 1min cleanup。HARD LIMIT B≤512 即使 peak_mem 报告显示有余量。"OOM transition is not graceful in WSL2 and a hung driver cancels your run rather than raising a recoverable Python exception."
> - **Risk if missed**: WSL2 复现者可能尝试更大 batch 训练崩溃后归因到我们方法，而不是平台限制。
> - **Cross-link**: M3-007 / M3-008 / M3-009 同属 m3_perf 性能档案；M1-001（WSL2 环境基线）的边界条件。
> - **Priority**: LOW

---

# Transition (M3→M4) — 真数据 ETL 适配阶段

### Finding TRANSITION-001: CIC-IDS 2017 pcap 是 pcapng 格式（非 classic libpcap）
- **Context**: 真实数据 dry-run 第一次跑撞到
- **Discovery**: CIC 官方 pcap 文件 magic bytes 是 `0a 0d 0d 0a`（pcapng）而非 `d4 c3 b2 a1` / `a1 b2 c3 d4`（classic libpcap）。`dpkt.pcap.Reader` 不支持 pcapng，全部三个 pcap 在 dry-run 阶段全部解析失败。
- **Evidence**: src/nid_video/data/pcap_parser.py `_open_pcap` magic 分发 + tests/test_pcap_parser.py::test_packet_stream_parses_pcapng_same_as_classic
- **Decision rationale**: 选 dpkt.pcapng.Reader 兼容（不引入 C 扩展，与立意一致）
- **Status**: persisted in code + CI

> 🎯 **论文价值标注**
> - **Section**: Reproducibility / Dataset Preprocessing Pitfalls
> - **Use**: CIC-IDS 2017 复现陷阱清单的入口项——CIC 官方 pcap magic bytes 是 `0a 0d 0d 0a`（pcapng）而非 `d4 c3 b2 a1`（classic libpcap），`dpkt.pcap.Reader` 不支持 pcapng，三个 pcap 在 dry-run 阶段全部解析失败。许多基于 dpkt 的 baseline 复现可能在第一步就失败。
> - **Risk if missed**: Reviewer 或 reproducer 按 dpkt classic reader 路径走会发现"全部 pcap 解析失败"，归因到我们 ETL 设计有误。
> - **Cross-link**: TRANSITION-003（cp1252 编码）、TRANSITION-004（CIC 工具 bug 288602 空行）、M2-007（EN-DASH U+2013）共同构成 "CIC-IDS 2017 dataset preprocessing pitfalls" 章节；TRANSITION-002（pcapng 慢 45%）是本条修复的性能后果。
> - **Priority**: MEDIUM

### Finding TRANSITION-002: pcapng 解析速度比 classic pcap 慢 45%（71k vs 138k pps）
- **Context**: TRANSITION-001 修复后实测
- **Discovery**: dpkt.pcapng.Reader 的真实数据吞吐 71k pps，是 classic pcap 138k pps 的 55%（慢 45%）。原因是 pcapng 格式更复杂（多种 block types、可变长度、增强时间戳精度）。
- **Evidence**: TRANSITION 阶段 1000-packet benchmark
- **Decision rationale**: 全 ETL 时长从 5.5 min 增至 ~10 min，仍在验收线内，可接受
- **Status**: documented (将记入 docs/etl_performance.md)

> 🎯 **论文价值标注**
> - **Section**: Implementation / Computational Efficiency
> - **Use**: ETL 性能档案的补充数字——`dpkt.pcapng.Reader` 真实数据吞吐 71k pps，是 classic pcap 138k pps 的 55%（慢 45%）。全 ETL 时长从 5.5 min 增至 ~10 min（合成 pcap 估算），仍在验收线内。论文 efficiency table 报告 ETL 时长时配合 M2-009 引用。
> - **Cross-link**: TRANSITION-001（pcapng 格式发现）的性能后果；M2-009（dpkt L2 ceiling 138k pps、PacketStream 127k pps）共同构成 ETL 性能档案。
> - **Priority**: LOW

### Finding TRANSITION-003: CIC CSV 用 CP-1252 编码（不是 UTF-8 或 latin-1）
- **Context**: 真数据 LabelIndex 构建警告
- **Discovery**: CIC-IDS 2017 标签 CSV 是 Windows 工具产出，**正确编码是 CP-1252 而非 latin-1**。0x96 在 CP-1252 是 EN DASH (U+2013)，在 latin-1 是控制字符 U+0096。原代码用 `encoding="latin-1"` 导致 "Web Attack \x96 Brute Force" 类标签字符串无法匹配 LABEL_TO_ID（key 是 EN DASH）。
- **Evidence**: src/nid_video/data/labeling.py `_load_label_csv` + tests/test_labeling.py::test_csv_with_cp1252_endash_byte_maps_to_web_attack_subtypes
- **Decision rationale**: cp1252 是 root-cause fix，不是字符串 hack。ASCII 范围 (0x00-0x7F) 与 latin-1 完全等价，所有非 Web Attack 标签零变化
- **Status**: persisted in code + CI

> 🎯 **论文价值标注**
> - **Section**: Reproducibility / Dataset Preprocessing Pitfalls
> - **Use**: 量化复现陷阱——CIC-IDS-2017 标签 CSV 是 Windows 工具产出，正确编码是 CP-1252 而非 latin-1。`encoding="latin-1"`（许多开源代码的默认值）会让 0x96 字节被误读为控制字符 U+0096 而非 EN DASH（U+2013），导致 "Web Attack \x96 Brute Force" 类标签字符串无法匹配 LABEL_TO_ID（key 是 EN DASH）——**Thursday 子集中 2,180 个 Web Attack 样本会全部 silent default 到 BENIGN**。
> - **Quote candidate**: "We identified that CIC-IDS-2017 label CSVs are CP-1252 encoded; using `latin-1` (the default in many open-source codebases) silently misinterprets the EN DASH separator in 'Web Attack – {Brute Force, XSS, SQL Injection}' labels, causing all 2,180 Web Attack samples in the Thursday subset to default to BENIGN—an invisible failure mode that produces self-consistent but degraded results."
> - **Risk if missed**: 许多基于 CIC-IDS 2017 的 baseline 论文可能在不知情中踩这个坑，得到"Web Attack 三子类全 0"或异常低 F1 但归因到方法不行。预先披露既是 community service 也是我们方法学严谨性证据。
> - **Cross-link**: M2-007（EN-DASH U+2013 字节同源）是字符串层面问题，本条是文件编码层面问题——TRANSITION-003 是 root-cause fix，M2-007 是字符串处理 fix；二者配套；TRANSITION-001 / TRANSITION-004 / M2-007 共同构成 "CIC-IDS 2017 dataset preprocessing pitfalls" 章节；与 TRANSITION-005（TZ）、M4-001（12h-without-AM/PM）同属 "silent failure mode" 模式族。
> - **Priority**: MEDIUM

### Finding TRANSITION-004: CIC 标注工具的产物 bug（288602 空行）
- **Context**: 28.8 万行 unparseable timestamp 诊断
- **Discovery**: Thursday-WorkingHours-Morning-WebAttacks.csv 末尾追加了 288,602 行纯逗号（86 字节，84 个逗号）。**288,602 = Thursday-Afternoon-Infiltration.csv 的总行数**——CIC 标注工具保存 WebAttacks 时把 Infiltration 的 row count 写进去了但没补内容。
- **Evidence**: scripts/diagnose_unparseable_timestamps.py + 修复 commit
- **Decision rationale**: 不是我们的解析问题。修法是 `df.dropna(how="all")` 在 timestamp 解析前去全空行 + 拆分 warning 文案区分 "fully-empty" vs "unparseable timestamp"
- **Status**: persisted in code + CI

> 🎯 **论文价值标注**
> - **Section**: Limitations / Dataset Preprocessing Pitfalls
> - **Use**: CIC-IDS 2017 数据集工程质量问题的进一步证据——Thursday-WorkingHours-Morning-WebAttacks.csv 末尾追加 288,602 行纯逗号（86 字节，84 个逗号），数量精确等于 Thursday-Afternoon-Infiltration.csv 总行数。CIC 标注工具保存 WebAttacks 时把 Infiltration 的 row count 写进去了但没补内容。修法：`df.dropna(how="all")` 在 timestamp 解析前去全空行 + 拆分 warning 文案区分 "fully-empty" vs "unparseable timestamp"。
> - **Risk if missed**: 复现者可能把 288,602 个 unparseable timestamp warning 当成自己代码的 bug 而非数据集问题。
> - **Cross-link**: TRANSITION-001 / TRANSITION-003 / M2-007 共同构成 "CIC-IDS 2017 dataset preprocessing pitfalls" 章节。
> - **Priority**: LOW

### Finding TRANSITION-005: pcap 是 UTC，CSV 是 ADT 本地时间（系统性 3 小时错位）
- **Context**: 真数据 ETL 时区交叉验证
- **Discovery**: CIC-IDS 2017 录制日（2017-07-04）pcap 文件用 UTC unix timestamp，但标签 CSV 时间戳字符串是 ADT 当地时间（UTC-3，夏令时）。**默认按 UTC 处理 CSV 会让 5-tuple lookup 系统性偏 3 小时**。
- **Evidence**:
  - 6.1M packet ground-truth lookup 对照表（Tuesday 09:20-10:20 ADT 窗口）：
    - **未修复**：99.99% packets unmatched，FTP-Patator 命中 **0** 次
    - **修复后**：5,030,982 BENIGN + 9,410 FTP-Patator 正确打标，17% unmatched 是预期（短流尾包）
  - tests/test_labeling.py::test_csv_tz_localizes_adt_summer_time_to_utc + test_csv_tz_localizes_ast_winter_time_to_utc
- **Decision rationale**: zoneinfo `tz_localize().tz_convert("UTC")` 替换 naive int64 转换，`csv_tz="America/Halifax"` 配置参数支持跨数据集
- **Status**: persisted in code + CI
- **Updated**: 2026-04-27 (CIC official full pcap)
  原 6.1M 包对照表是在过渡阶段下载的部分版本 Tuesday pcap (~3 GB) 上跑的，
  当时与 .md5 校验通过但只覆盖到 ~9:20 ADT 起点。现已替换为 CIC 官方完整版
  Tuesday-WorkingHours.pcap (11.05 GB, MD5 2d976fec... 与 .md5 一致)。
  - **修复正确性**：CSV 侧逻辑（zoneinfo/cp1252/empty rows）不依赖 pcap 内容，
    不变。
  - **新文件起点验证**：first packet ts=1499169224.398 = 2017-07-04 08:53:44 ADT
    （与原文件相同 first packet，说明 pcap 前缀字节相同；新文件只是后段更长）。
  - **官方完整版上的 6.1M 包对照表重跑延迟到 M4 任务 4.7（全 ETL）顺手做**，
    彼时数字会成为论文最终引用的 ground truth。当前 6.1M 数字保留为
    机制证明，论文写作时引用新数字。

> 🎯 **论文价值标注**
> - **Section**: Methods (Dataset Preprocessing) + Discussion (Silent Failure Modes — flagship evidence)
> - **Use**: **整个项目工程严谨性的旗舰证据**。CIC-IDS 2017 录制日 pcap 用 UTC，标签 CSV 用 ADT（UTC-3 夏令时），3 小时错位是系统性的。6.1M Tuesday packets ground-truth lookup 对照表是大数定律意义的硬证据：**未修复 99.99% packets unmatched / FTP-Patator 命中 0 次；修复后 5,030,982 BENIGN + 9,410 FTP-Patator 正确打标，17% unmatched 是预期（短流尾包）**。这条与 M4-001（12h-without-AM/PM）是两个独立的 silent failure 根因——单独存在 ~50% 抑制，同时存在 100% 抑制 PM 攻击；论文 Discussion 章节"silent failure detection chain"叙事的开端。
> - **Figure idea**: 对照柱状图——横轴是攻击类型（FTP-Patator / SSH-Patator / DDoS / Heartbleed / PortScan...），纵轴是 packets per class，并列两组柱（Buggy UTC vs Fixed ADT）。FTP-Patator 那一列 0 → 9,410 是绝佳视觉震撼；同时引导读者注意到这种"100% 抑制"的失败模式如何能产生 self-consistent 的训练结果而不被发现。
> - **Quote candidate**: "We discovered through ground-truth 5-tuple verification on 6.1M Tuesday packets that pcap timestamps are recorded in UTC while CSV labels use Atlantic Daylight Time (ADT, UTC-3); ignoring this 3-hour offset—as is the case in numerous CIC-IDS-2017 reproductions—causes 99.99% of packets to fail label matching and **all PM attack labels to silently default to BENIGN**, an invisible failure mode that yields self-consistent but meaningless training results."
> - **Risk if missed**: 这条是论文 Discussion 章节"silent failure detection chain"的旗舰开篇。不写则失去：(1) 对 NID 领域的 community service 价值（许多既有论文可能踩同一坑）、(2) 对我们方法学严谨性的最强证据、(3) 与 M4-001 共同构成"两个独立 silent failure 模式"叙事的一半。
> - **Cross-link**: TRANSITION-007（ETL-level 端到端验证）形成"packet-level + ETL-level 两层独立验证"叙事对——TRANSITION-005 是 6.1M 包大数据级 ground truth、TRANSITION-007 是 ETL 集成验证，两层证据强度乘法叠加；M4-001（12h-without-AM/PM）是第二个独立 silent failure 根因，与本条共同构成"silent failure detection chain"；M4-002（三守恒律）+ M4-010a（下游 F1 验证）是修复正确性的代数级与训练级证据；TRANSITION-001 / TRANSITION-003 / TRANSITION-004 / M2-007 共同构成 "CIC-IDS 2017 dataset preprocessing pitfalls"。
> - **Priority**: HIGHEST

### Finding TRANSITION-006: dpkt 在真数据上对截断包返回 raw bytes 而非 dpkt.tcp.TCP 对象
- **Context**: 6.1M 包手动验证时撞到 AttributeError
- **Discovery**: dpkt 解析 IP 层成功后，对一些截断/异常的包会把 L4 层 fallback 成 `bytes` 而非 `dpkt.tcp.TCP` / `dpkt.udp.UDP` 对象。Tuesday pcap 6.1M 包里有 32 个这种包，访问 `l4.sport` 时 AttributeError 整个 pcap 被标 failed。
- **Evidence**: src/nid_video/data/pcap_parser.py 加 `if not hasattr(l4, "sport") or not hasattr(l4, "dport")` guard + tests
- **Status**: persisted in code + CI

> 🎯 **论文价值标注**
> - **Section**: Reproducibility / Implementation Notes
> - **Use**: 又一项陷阱清单——dpkt 解析 IP 层成功后，对截断/异常的包会把 L4 层 fallback 成 `bytes` 而非 `dpkt.tcp.TCP` / `dpkt.udp.UDP` 对象。Tuesday pcap 6.1M 包里有 32 个这种包，访问 `l4.sport` 时 AttributeError 整个 pcap 被标 failed。"合成 pcap fixture 永远是 well-formed，真数据才暴露 bug"——是 transition 阶段（合成 → 真数据）系统性发现的一个具体实例。
> - **Cross-link**: TRANSITION-001 / TRANSITION-003 / TRANSITION-004 / M2-007 共同构成 "CIC-IDS 2017 dataset preprocessing pitfalls"；与 TRANSITION-005 配套作为"真数据暴露 vs 合成数据隐藏"系统性观察。
> - **Priority**: LOW

### Finding TRANSITION-007: ETL 端到端 dry-run 闭环验证 TZ 修复
- **Context**: 5 个 Finding 全部 patch 完成后的真数据集成验证
- **Discovery**: 在两轮 dry-run 上完整闭环了 TZ 修复的有效性：
  - 100 窗口 dry-run（Tuesday 9:00-9:01:20 ADT）：unmatched 1260/~8000 ≈ **16%**（pre-fix 99.99%）
  - 2000 窗口 dry-run（覆盖到 9:20+ FTP-Patator 时段）：**9 个 FTP-Patator 窗口命中**（pre-fix 0 个）
- **Evidence**: prompts/03_*/transition 阶段 dry-run 报告 + commit 02babba
- **Decision rationale**: 这是 packet-level (6.1M 对照表) → ETL-level (2000 窗口) 的两层独立验证
- **Status**: documented in finding ledger + verified by integration run
- **Updated**: 2026-04-27 (CIC official full pcap)
  在 CIC 官方完整版 Tuesday pcap (11.05 GB) 上重跑 2000 窗口 dry-run，标签分布
  与原 ~3 GB 部分版本**bit-identical**：
  - 1991 BENIGN + 9 FTP-Patator（与原 dry-run-2000 一致，delta=0）
  - 总耗时 68.8 s（原 101.3 s，加快源自文件系统缓存）
  - 覆盖范围：08:53:44 - 09:20:23 ADT（26.7 min，2000 窗口在 9:20 FTP-Patator
    刚开始时停止）
  - Unmatched packets: 633,710（与原 dry-run 一致）
  说明：(1) 修复仍在新 pcap 上正确产出标签；(2) 旧 pcap 与新 pcap 在前 26.7 min
  内字节等价（first packet ts 同为 1499169224.398），新文件只是后段更长。
  M4 任务 4.7 全 ETL 跑完后会有覆盖整个工作日的标签分布数字（含全部 FTP-Patator
  9:20-10:20、SSH-Patator 14:00-15:00 时段），届时该数字成为论文最终引用值。

> 🎯 **论文价值标注**
> - **Section**: Methods (Dataset Preprocessing) + Reproducibility (Multi-Level Verification)
> - **Use**: 与 TRANSITION-005 配套——TRANSITION-005 是 packet-level 6.1M 包 ground truth，本条是 ETL-level 2000-window 端到端验证（pre-fix 0 个 FTP-Patator 窗口 / post-fix 9 个 FTP-Patator 窗口）。**两层独立验证大大增强可信度**——一层独立证据可被质疑为"巧合"或"local fix"，两层独立验证（packet-level 大数定律 + ETL-level 端到端流程）的同向结果几乎无法被否定。CIC 官方完整版 pcap 重跑后两层数字 bit-identical 进一步固化此证据。
> - **Quote candidate**: "We verified the timezone fix at two independent levels: (1) packet-level ground-truth lookup against label CSV on 6.1M Tuesday packets, reducing unmatched rate from 99.99% to ~17%; and (2) end-to-end ETL run on the same time range producing 9 FTP-Patator windows where the pre-fix pipeline produced zero. Both verifications are bit-identical on the CIC official full-version pcap (11.05 GB)."
> - **Risk if missed**: 不写这条则 TRANSITION-005 是孤证，"6.1M 对照表"可能被质疑"为什么这个修复在 packet-level 看起来对，下游 ETL 真的也对吗"。本条提供 ETL-level 答案。
> - **Cross-link**: TRANSITION-005（packet-level ground truth）形成"两层独立验证"叙事对；TRANSITION-008（dominant-rule 不对称的初始观察——FTP-Patator 9 窗口而非更多，对应低 RPS 攻击的标注偏好）是本条 dry-run 中观察到的延伸现象；M4-001 / M4-002 / M4-010a silent failure detection chain 的下游环节。
> - **Priority**: HIGHEST

### Finding TRANSITION-008: Dominant-attack labeling rule 对低强度攻击不敏感
- **Context**: 2000 窗口 dry-run 观察到 FTP-Patator 命中 9 窗口（远少于直觉预期）
- **Discovery**: 我们的 `label_window` 用 dominant-attack rule（窗口内攻击包占比最高的类作为 label）。CIC FTP-Patator 是突发型 brute force，1.6s 窗口里多数情况下背景 BENIGN 流量包数仍占多数 → 整个窗口被标 BENIGN。**这不是 bug，但意味着低 RPS 攻击在我们标注下样本量减少**。
- **Evidence**: 2000 窗口 dry-run 标签分布（1991 BENIGN / 9 FTP-Patator，~0.5%）
- **Decision rationale**: M5 baseline 对比下保持公平统一标准（所有方法用相同的 dominant-attack rule）。是否引入 `attack_ratio_threshold` 参数（比如 0.1 即 10% 攻击包就标 attack）让 brute force 样本量增加，留给 M4 决定
- **Status**: documented for M4 consideration

> 🎯 **论文价值标注**
> - **Section**: Methods (Labeling Strategy) + Limitations
> - **Use**: 主动声明标注规则的偏好——`label_window` 用 dominant-attack rule（窗口内攻击包占比最高的类作为 label），CIC FTP-Patator 是突发型 brute force，1.6s 窗口里多数情况下背景 BENIGN 包数仍占多数 → 整个窗口被标 BENIGN。这不是 bug，是有意的"公平基线 + 与 baseline 论文可比"决策，但意味着低 RPS 攻击在我们标注下样本量减少。论文需主动披露这一标注偏好以避免 reviewer 质疑"FTP-Patator F1 偏低是因为方法不行还是样本量少"。
> - **Quote candidate**: "We adopt the dominant-attack labeling rule (a window inherits the class of the majority attack-type packet) for fair comparability with prior CIC-IDS baselines; this introduces an asymmetry against burst-type low-RPS attacks (e.g., FTP-Patator, PortScan), where background BENIGN traffic frequently dominates the per-window count even within attack time spans."
> - **Risk if missed**: Reviewer 看到稀有攻击类 F1 低 → 质疑"是方法不行还是数据问题"——不主动披露则把举证责任压给方法叙事。
> - **Cross-link**: M4-003（dominant-rule 双向不对称：长连接放大 / 短爆发压制——Heartbleed 93% windows hit vs PortScan 1.6%，60× 差距）是本条的全数据集级量化证据；TRANSITION-007 dry-run 中"FTP-Patator 9 窗口"是本条的初始观察起点；M4-005（split index-based 修法）的隐含前提是接受这种标注偏好；M2-008（极稀疏类披露）同属"诚实数据处理"方法学簇。
> - **Priority**: MEDIUM

---

# M4 — 真训练流程阶段

### Finding M4-001: CIC CSV 用 12h 无 AM/PM 标记格式（silent 12h offset）
- **Context**: M4 task 4.7 — 100ms 全量 ETL 完成后的标签分布异常诊断
- **Discovery**: CIC-IDS-2017 TrafficLabelling/*.csv 的 Timestamp 字段是 12h 格式但**不带 AM/PM 标记**，仅靠 hour ∈ [1,7] 范围隐式表达 PM。pd.to_datetime 默认按 24h 解析 → PM 攻击行错位 12h → 100ms 全集 110,783 windows 中：
  - AM 攻击全部命中：FTP-Patator 583, Bot 44, Wed-AM DoS 各 368-1598
  - PM 攻击全部 0 hit：SSH-Patator 0, DDoS 0, PortScan 0, Heartbleed 0, Wed-PM DoS 0
  hour 范围验证：Friday-Afternoon-DDos.csv hours=[3,4]（CIC docs 15:56-16:16 ADT），Friday-Afternoon-PortScan.csv hours=[1,2]（13:55-15:29 ADT）。修法：hour ∈ [1,7] +12h，[8,11] 不变，12 不变（noon=12 PM=12:00 24h），0 warn-but-unchanged。
- **Evidence**: src/nid_video/data/labeling.py `_absorb` 12h-shift 块（pd.to_datetime 之后、tz_localize 之前）+ tests/test_labeling.py 6 个新测试（test_hour_in_pm_range_gets_shifted_plus_twelve parametrized 1-7, test_hour_in_am_range_unchanged parametrized 8-11, test_hour_12_unchanged_as_noon, test_hour_0_warns_but_unchanged, test_inference_disabled_passes_through, test_real_csv_friday_afternoon_ddos_hour_range_after_fix）+ docs/v1_vs_v2_comparison.md
- **Decision rationale**: 选 hour-based 推断（不是 string-replace AM/PM）因为 CIC CSV 字面**没有** AM/PM 字符；所有攻击时段对照 CIC 文档（4 类精确到分钟、Heartbleed/DDoS/Infiltration/Web Attack 各子类）后采纳
- **Status**: persisted in code + CI

> 🎯 **论文价值标注**
> - **Section**: Methods (Dataset Preprocessing) + Discussion (Silent Failure Modes — flagship evidence)
> - **Use**: **silent failure detection chain 的第二个独立根因（与 TRANSITION-005 TZ 错位平级）**。CIC-IDS-2017 标签 CSV 的 Timestamp 字段是 12h 格式但**不带 AM/PM 标记**，仅靠 hour ∈ [1,7] 隐式表达 PM。`pd.to_datetime` 默认按 24h 解析 → PM 攻击行错位 12h → 100ms 全集 110,783 windows 中 **AM 攻击 6 类全部命中（1598/623/588/583/368/44），PM 攻击 4 类全部 0 hit（SSH-Patator/DDoS/PortScan/Heartbleed）**。修法是 hour-based 推断（不是 string-replace AM/PM，因为 CSV 字面没有 AM/PM 字符）：hour ∈ [1,7] +12h，[8,11] 不变，12 不变（noon），0 warn-but-unchanged。
> - **Figure idea**: 配合 v1_vs_v2_comparison.md 的 v1→v2 delta 表做柱状图——横轴 11 类 raw15 攻击，纵轴 windows count，并列 v1（pre-fix）vs v2（post-fix）。AM 6 类两组柱完全等高（delta=0 视觉直接证明守恒律 #1），PM 4 类 v1 全 0 / v2 933/1375/111/1505 形成强烈对比，BENIGN 那一柱减量 -3,924 = PM 增量之和的视觉守恒律。
> - **Quote candidate**: "CIC-IDS-2017 label CSVs encode timestamps in 12-hour format without explicit AM/PM markers, relying solely on the hour value's range to indicate the period (1–7 for PM, 8–11 for AM, 12 for noon). Default parsers treating these as 24-hour time silently shift all PM attack rows by 12 hours, causing every PM attack class in the Tue+Wed+Fri subset (SSH-Patator, DDoS, PortScan, Heartbleed) to register zero windows—not because the attacks are absent from the pcap, but because the label lookup falls outside the attack time span."
> - **Risk if missed**: 这条与 TRANSITION-005 共同构成论文 Discussion 章节"silent failure detection chain"的两个独立根因。两条都不写 → 失去整段叙事；只写 TRANSITION-005 不写本条 → 失去"two independent silent-failure modes that compound multiplicatively"的关键论点（v1_vs_v2_comparison.md 顶部明确指出："Each one alone would suppress ~50% of the attack mass; both undetected would suppress 100% on PM"）。
> - **Cross-link**: TRANSITION-005（pcap UTC vs CSV ADT）形成"两个独立 silent failure 根因"叙事对——两个根因分别独立存在时各抑制 ~50% 攻击质量，同时未检测时 100% 抑制 PM 攻击；M4-002（三守恒律）是本条修复正确性的代数级证明；M4-010a（下游 F1 验证）是本条修复的训练级最终验证；TRANSITION-007（ETL-level 端到端验证）是 silent failure chain 的方法学补充层；TRANSITION-003（cp1252 编码）同属 silent failure mode 模式族但量级更小（2,180 样本 vs 本条 4 类 PM 攻击全 0）。
> - **Priority**: HIGHEST


### Finding M4-002: 三守恒律证明 12h 修复正确性（无副作用、无 silent loss/gain）
- **Context**: M4 task 4.7 — 12h 修复实施完成后的 v1 vs v2 ETL 输出对照（docs/v1_vs_v2_comparison.md）
- **Discovery**: 修复前后 100ms ETL 输出三个守恒律全部满足：
  1. **AM 攻击 6 类 v1==v2 字节级一致**（FTP-Patator 583/583, DoS-slowloris 1598/1598, DoS-Slowhttptest 623/623, DoS-Hulk 588/588, DoS-GoldenEye 368/368, Bot 44/44 — 全部 delta=0）
  2. **BENIGN 减量等于 PM 攻击增量之和**：106,979 → 103,055（-3,924）= 933 (SSH-Patator) + 1,375 (DDoS) + 111 (PortScan) + 1,505 (Heartbleed) = 3,924。无 silent loss / gain。
  3. **总 windows 数守恒**：110,783 (v1) == 110,783 (v2)。windowing 决策不受影响。
  含义：修复仅作用于 PM 行（hour ∈ [1,7]），AM 行（[8,11]+12）零干扰，没有创造或销毁 windows。
- **Evidence**: docs/v1_vs_v2_comparison.md "v1 → v2 delta" 章节 + 100ms_v2 ETL 完整 manifest 聚合
- **Decision rationale**: 该三守恒律是修复正确性的"数学证明级"实证；reviewer 无法质疑；论文级证据归档
- **Status**: documented in `docs/v1_vs_v2_comparison.md`

> 🎯 **论文价值标注**
> - **Section**: Methods (Dataset Preprocessing — Fix Verification) + Discussion (Silent Failure Modes)
> - **Use**: **M4-001 修复正确性的代数级证明**。v1 vs v2 ETL 输出三守恒律全部满足：
>   1. **AM 攻击 6 类 v1==v2 字节级一致**（FTP-Patator 583/583, slowloris 1598/1598, Slowhttptest 623/623, Hulk 588/588, GoldenEye 368/368, Bot 44/44——全部 delta=0）
>   2. **BENIGN 减量 = PM 攻击增量之和**：106,979 → 103,055（-3,924）= 933 (SSH-P) + 1,375 (DDoS) + 111 (PortScan) + 1,505 (Heartbleed) = 3,924
>   3. **总 windows 数守恒**：110,783 (v1) == 110,783 (v2)
>   含义：修复仅作用于 PM 行（hour ∈ [1,7]），AM 行零干扰，没有创造或销毁 windows。这是论文级"reviewer 无法质疑"的修复正确性证据——不是诉诸 unit test 或 inspection，而是大样本数学守恒。
> - **Figure idea**: 表格 figure（与 M4-001 柱状图配套）——左列 11 类标签 + BENIGN，中列 v1 windows，右列 v2 windows，最右列 delta。AM 6 类 delta 全 0、BENIGN -3,924、PM 4 类 +933/+1375/+111/+1505、Total 110,783==110,783，三守恒律视觉直接呈现。
> - **Quote candidate**: "We verify the 12-hour fix via three conservation laws on ETL output: (1) all six AM-attack classes show byte-identical window counts pre/post fix (delta=0); (2) the BENIGN-class decrement (-3,924) exactly matches the sum of PM-attack increments (+3,924); (3) total window count is preserved (110,783 in both runs). These laws collectively certify that the fix is surgical—touching only PM rows (hour ∈ [1,7]) and neither creating nor destroying windows."
> - **Risk if missed**: 不写这条 → M4-001 修复成为"trust-me"声明而非可独立验证的事实；reviewer 可能质疑"修复后 v2 数字看起来对，但你怎么证明没在 AM 类引入次生副作用"。三守恒律的代数级证明把这种质疑彻底封闭。
> - **Cross-link**: M4-001（12h 修复发现）的修复正确性证据；TRANSITION-005（TZ 修复）+ TRANSITION-007（ETL-level 验证）+ M4-010a（下游 F1 验证）共同构成 silent failure detection chain 的核心五件套；与 m3_perf.md 实证导向方法学一脉相承（数字驱动的修复证据，不是定性 claim）。
> - **Priority**: HIGHEST


### Finding M4-003: dominant-attack rule 双向不对称（长连接放大、短爆发压制）
- **Context**: M4 task 4.7 — 100ms ETL 修复后标签分布 + 1000ms ETL 对比观察
- **Discovery**: 同一 dominant-rule 在不同攻击类型下产出截然相反的"窗口命中率"：
  - **长连接攻击（Heartbleed）**：CIC 仅 ~11 个 flow 但跨 20 min span，每个 1.6s 窗口里只要有 1 个 Heartbleed 包就被标 → **1505 windows ≈ 攻击时段 93%**
  - **短爆发攻击（PortScan）**：94 min 攻击时段、~7000 个潜在滑窗，背景 BENIGN 流量包数压过 PortScan 包 → **111 windows ≈ 1.6%**
  - 两者相同 dominant-rule，结果差 60×
  延伸到 1000ms（Δt=1s 步长）：低 RPS 攻击的相对命中率反而更高（larger window catches more attack packets）。Bot 1.2× / FTP-P 3.3× / PortScan 3.3× / SSH-P 4.5× / Hulk 4.4× （vs 高 RPS 9-10×），与多尺度训练设计预期吻合
- **Evidence**: M4.7 100ms vs 1000ms ETL 标签分布对照表（M4 整体完成报告会贴）+ TRANSITION-008 配套
- **Decision rationale**: M4 阶段保留 dominant-rule 不改（公平基线 + 与 baseline 论文可比）；M5 baseline 比对时同时报告 window-count 与 attack-detection-rate 两层指标
- **Status**: documented; cross-links TRANSITION-008 (initial observation in low-RPS attacks) — together they constitute "dominant-rule asymmetry" methodology discussion candidate

> 🎯 **论文价值标注**
> - **Section**: Methods (Labeling Strategy) + Discussion (Methodology Limitations)
> - **Use**: TRANSITION-008 dominant-rule 标注偏好的全数据集级量化证据——同一 dominant-rule 在不同攻击类型下产出截然相反的窗口命中率：**Heartbleed（长连接 / 11 个 flow / 20 min span）→ 1505 windows ≈ 攻击时段 93%**；**PortScan（短爆发 / 94 min 攻击时段 / ~7000 潜在滑窗）→ 111 windows ≈ 1.6%**。两者相同规则、结果差 60×。延伸到 Δt=1s 多尺度旁路：低 RPS 攻击的相对命中率反而更高（Bot 1.2× / FTP-P 3.3× / PortScan 3.3× / SSH-P 4.5× / Hulk 4.4× vs 高 RPS 9-10×），与 Idea.md §3.4 多尺度训练设计预期吻合——这是 scale token 多尺度采样设计的实证支持。
> - **Quote candidate**: "The dominant-attack labeling rule exhibits asymmetric sensitivity by attack duration: long-lived flows like Heartbleed (a handful of flows spanning 20 minutes) yield ~93% window hit rate within their span, while short-burst attacks like PortScan yield only ~1.6%—a 60× ratio under identical rules. This asymmetry justifies the multi-scale training scheme: at Δt=1s, low-RPS attack hit rates rise 3-5× while high-RPS attacks rise 9-10×, narrowing the relative gap."
> - **Risk if missed**: Reviewer 看到 PortScan / Bot 等稀有类 F1 偏低 → 质疑"是方法不行还是样本量少"。本条提供"标注规则的内禀偏好"答案，并把 multi-scale 设计从"理论上支持长短攻击"提升到"有实证数据支持各类攻击在多尺度下的相对命中率"。
> - **Cross-link**: TRANSITION-008（dominant-rule 标注偏好初始观察 / dry-run 9 个 FTP-Patator）的全数据集级延伸证据；M2-008（极稀疏类披露）同属"诚实数据处理"方法学簇；M4-010b（Bot/GoldenEye F1=0/AUROC>0.7 豁免规则）是本条标注偏好的下游训练表现；与 Idea.md §3.4 multi-scale scale token 设计直接对应。
> - **Priority**: MEDIUM


### Finding M4-004: 时间位置 split 的嵌套窗口失败模式（4.7 first redesign）
- **Context**: M4 task 4.7 — splits.parquet 第一次生成后的 split-by-attack 分布异常
- **Discovery**: 4.1 设计的"在所有攻击 attack windows 中按 tmin 升序 tiebreak、用 winning range 算时间位置"对**嵌套**攻击窗口失败：CIC Wednesday slowloris 09:01-14:25（5h24min）完全包含 Slowhttptest（22min）/ Hulk（24min）/ GoldenEye（9min）。tmin tiebreak 让所有内部窗口用 slowloris's range 算位置 → 全落在 slowloris 0-70% → **4 类 Wed-AM DoS 全 100/0/0 by split**
- **Evidence**: M4.7 第一次 splits.parquet 验证报告（split-by-attack 分布表显示 Slowhttptest/Hulk/GoldenEye 全 100/0/0）+ 当时讨论
- **Decision rationale**: 第一次 redesign 改为"label-aware"：每个攻击窗口用 (pcap_source, label_id) 查 own attack range，删 tiebreak。但仅修了"嵌套"问题，未修"分布不均"问题（见 M4-005）
- **Status**: superseded by M4-005 (label-aware time-position 仍假设 windows 在 own range 内时间均匀)

> 🎯 **论文价值标注**
> - **Section**: Methods (Train/Val/Test Split) — historical context only
> - **Use**: M4.7 split 设计的迭代失败记录——4.1 设计的"在所有攻击 attack windows 中按 tmin 升序 tiebreak、用 winning range 算时间位置"对**嵌套**攻击窗口失败：CIC Wednesday slowloris 09:01-14:25（5h24min）完全包含 Slowhttptest（22min）/ Hulk（24min）/ GoldenEye（9min）。tmin tiebreak 让所有内部窗口用 slowloris's range 算位置 → 全落在 0-70% → 4 类 Wed-AM DoS 全 100/0/0 by split。这条已被 M4-005 superseded，论文最终方案不包含本条，但作为方法学迭代记录在 ablation 或 supplementary 章节有价值。
> - **Risk if missed**: 不写则失去"我们如何发现 split 设计的失败模式并迭代修正"的方法学透明度叙事。但论文核心叙事不依赖这条。
> - **Cross-link**: M4-005（second redesign with index-based partition）supersedes 本条；二者形成"split 设计两次失败 → 终极修复"的迭代记录。
> - **Priority**: LOW


### Finding M4-005: 时间位置 split 的"窗口非均匀分布"失败模式（4.7 second redesign）
- **Context**: M4 task 4.7 — label-aware redesign 实施后第二次 splits.parquet 验证
- **Discovery**: 4.1 设计的"按时间位置 partition" 隐含假设是 **windows 在 attack [tmin, tmax] 内时间均匀分布**。CIC 数据违反此假设：
  - DoS slowloris 1598 windows 全在 flow range [09:01, 14:25] 的前 70%（< 12:47）→ 仍 100/0/0 split
  - PortScan 111 windows 集中在 flow range [13:05, 15:23] 的 70-85% 切片 → **9/82/9 split**（91 windows 在 val 切片）
  Root cause：flow-level [tmin, tmax]（CSV 5-tuple `min/max(start_ts)`）≠ window-level dominant-rule active period。修法：**index-based partition** —— 每 (pcap, label_id) 组按 start_time 排序后取前 `int(n*0.7)` train / 中 `int(n*0.15)` val / 余 test。数学保证 70/15/15 by count，"early 入 train、late 入 test" no-session-leakage 性质保留
- **Evidence**: src/nid_video/data/split.py `compute_split_assignments` index-based 实现 + tests/test_split.py 8 新测试（test_attack_window_clustered_split_uses_index_not_time / test_split_70_15_15_guaranteed_for_each_attack_class / test_portscan_split_distribution_within_5pct / test_split_index_based_is_deterministic / test_split_index_based_input_order_invariant / test_attack_windows_in_different_pcaps_split_independently / test_allocate_counts_for_all_n_values / test_small_attack_class_warns_when_under_10_windows）+ M4.7 second redesign 重跑后实测 11 类全部 70/15/15（4 类精确）
- **Decision rationale**: 彻底删 AttackWindow + attack_windows_for_pcaps 设计 debt（index-based 不需要 [tmin, tmax]）。scripts/run_split.py 从 80 行简化到 40 行，不再需要 CSV / LabelIndex
- **Status**: persisted in code + CI

> 🎯 **论文价值标注**
> - **Section**: Methods (Train/Val/Test Split) + Reproducibility
> - **Use**: split 设计的最终方案——index-based partition（每 (pcap, label_id) 组按 start_time 排序后取前 `int(n*0.7)` train / 中 `int(n*0.15)` val / 余 test）。修法的合理性论证：(1) 数学保证 70/15/15 by count；(2) "early 入 train、late 入 test" no-session-leakage 性质保留；(3) 不依赖 windows 在 attack [tmin, tmax] 内时间均匀分布的隐含假设——CIC 数据违反该假设的具体实例：DoS slowloris 1598 windows 全在 flow range [09:01, 14:25] 前 70%（仍 100/0/0 split），PortScan 111 windows 集中在 70-85% 切片（9/82/9 split）。**论文 Methods 章节直接采用本条最终方案**，M4-004 是方法学迭代记录。
> - **Quote candidate**: "We split each (pcap, attack-class) group by sorted index rather than by time position within the attack span: the first ⌊0.7n⌋ windows by start_time go to train, the next ⌊0.15n⌋ to val, the remainder to test. This guarantees a 70/15/15 split per class regardless of how non-uniformly windows distribute within the attack span, while preserving the no-session-leakage property that earlier windows train and later windows test."
> - **Risk if missed**: 不写则 split 设计成为"trust-me"。本条提供：(1) 设计选择的失败模式实证（M4-004 + slowloris 100/0/0 / PortScan 9/82/9 具体实例）、(2) 修法的数学保证、(3) 8 个 CI-pinned 测试守护——三层证据强度叠加。
> - **Cross-link**: M4-004（first redesign 失败模式）supersedes 关系——本条是 split 设计迭代的终态；TRANSITION-008（dominant-rule 标注偏好）→ M4-003（双向不对称）→ 本条（修法）形成"标注偏好引发分布不均 → split 设计修法"的逻辑链；M4-010a（split 修法的下游 F1 验证——slowloris 1118/239/241 v2 split → F1=0.7407）是本条修法的训练级最终验证。
> - **Priority**: HIGH


### Finding M4-006: MultiScaleNidDataset epoch 终止策略（slow_exhausted → round_robin）
- **Context**: M4 task 4.8 — 真训练首跑发现 4 个攻击类 F1=0
- **Discovery**: 4.2 实施时默认 `epoch_end_strategy="slow_exhausted"`（slow 流耗尽即 epoch 结束）。多尺度数据上 slow 是 fast 的 1/10 大小，配 50/50 mix 后 epoch ≈ 2 × slow_n；fast 80% 样本从未训练。M4.8 第一次跑：
  - 实际 grad_steps=479（vs total_steps=2426 = 19.7%）
  - lr=1.44e-04 仍在 warmup（warmup_steps=500）→ cosine decay 一次都没触发
  - val n=3,345（vs 完整 ~16,500 = 20%）
  - macro_f1=0.31，FTP/SSH/DDoS/Bot 全 F1=0（含与 12h fix 无关的 AM 类 FTP-P）
  修法：`round_robin` 改为默认。fast 流耗尽即 epoch 结束，slow 在耗尽时 cycle（重 iter）。1 epoch 实际 grad_steps=4853（10× 提升），val n=33,056（10× 提升），macro_f1=**0.45**（+44.6%），6/7 严格要求 F1>0
- **Evidence**: src/nid_video/data/dataset.py `MultiScaleNidDataset.__iter__` round_robin 实现 + tests/test_dataset.py 4 新测试（test_multi_scale_round_robin_strategy_cycles_slow_until_fast_done / test_multi_scale_round_robin_is_default_strategy / test_multi_scale_round_robin_slow_reshuffles_each_cycle，及保留的 test_multi_scale_slow_exhausted_strategy_stops_at_slow_end pin 旧行为）+ outputs/run_20260430_223105 训练日志
- **Decision rationale**: round_robin 让 fast 走完即 epoch 结束是干净的语义；slow_exhausted 保留为可选（小训练量调试）；max_len 仍 NotImplementedError 留给 M5/M6
- **Status**: persisted in code + CI

> 🎯 **论文价值标注**
> - **Section**: Implementation Notes (no paper exposure expected)
> - **Use**: 维护笔记。多尺度 dataset 的 epoch 终止策略修法：slow_exhausted（slow 流耗尽即结束）配 50/50 mix 让 fast 80% 样本从未训练；round_robin（fast 耗尽即结束、slow 在耗尽时 cycle）让 1 epoch grad_steps 从 479 → 4853（10×），val n 从 3,345 → 33,056（10×），macro_f1 0.31 → 0.45（+44.6%）。属于训练 infrastructure 决策，对论文叙事不直接相关。
> - **Cross-link**: M4-007（cosine decay 早终止）是 round_robin 修法的下游耦合 bug；M4-010a（M4.8 macro_f1=0.45 baseline）是本条修法的下游训练表现。
> - **Priority**: LOW


### Finding M4-007: total_steps × epoch_end_strategy 的耦合 bug（cosine decay 提前结束）
- **Context**: M4 task 4.8 — round_robin 修复后训练日志观察
- **Discovery**: 4.6 的 `_compute_total_steps` 从 splits.parquet 读 train sample 数计算 `total_steps = ceil(train_n / (B × accum)) × num_epochs`。这个公式假设了 slow_exhausted 策略下 epoch ≈ train_n / batch。改为 round_robin 后实际 epoch ≈ 2 × train_n / batch（slow 被 cycle 计入步数），但 total_steps 仍按 fast-only 算。M4.8 实测：
  - total_steps = 2426
  - 实际 grad_steps = 4853（≈ 2× total_steps）
  - cosine decay 在 step 2425 完成 → 后 ~50% 训练 lr=1.5e-06 plateau（min_lr_ratio × base_lr）
  - 含义：稀有类（Bot, GoldenEye）后半段几乎没机会再学习。Bot F1=0 / GoldenEye F1=0 但 AUROC=0.74/0.94 → 模型有 representation 但 argmax 阈值未充分校准
- **Evidence**: outputs/run_20260430_223105 训练日志 lr trace + scripts/train.py `_compute_total_steps` + 4.3 schedule
- **Decision rationale**: M4.8 一次性运行接受 plateau；M5+ 多 epoch 训练时需修。修法草案：`total_steps_for(strategy)`，round_robin 用 `(2 × train_n_fast / batch) × num_epochs`（fast 是 anchor stream），slow_exhausted 保留旧公式
- **Status**: pending followup (M5 multi-epoch task) — MUST FIX before M5 multi-epoch training; explanation candidate for M4.8 single-epoch macro-F1 ceiling in paper Discussion

> 🎯 **论文价值标注**
> - **Section**: Discussion (Limitations) + Future Work (M5 multi-epoch)
> - **Use**: explanation candidate for M4.8 single-epoch macro-F1 ceiling in paper Discussion. **Conditional**: if M5 multi-epoch training reaches macro-F1 > 0.7, this finding becomes Methods §3 implementation note rather than Discussion limitation (the ceiling explanation is no longer needed)
> - **Quote candidate**: "Single-epoch macro-F1 of 0.45 reflects a known coupling between the multi-scale epoch strategy and the cosine decay schedule: under round-robin, effective grad steps are roughly 2× the schedule's anchor (4,853 vs 2,426), causing the latter half of training to operate at floor learning rate. We expect this to be resolved in M5 multi-epoch training."
> - **Risk if missed**: Reviewer 看到 single-epoch macro-F1=0.45 → 质疑"为什么稀有类 F1=0 但 AUROC>0.7"。不写本条则 ceiling 数字成为"待解释的不一致"，写本条则成为"已知系统性原因 + 修法清晰"。
> - **Cross-link**: M4-006（round_robin 策略）的下游耦合 bug；M4-010a（M4.8 macro_f1=0.45 baseline）+ M4-010b（Bot/GoldenEye F1=0/AUROC>0.7 豁免规则）的解释根因；M5 multi-epoch followup 的关键 prerequisite。
> - **Priority**: MEDIUM


### Finding M4-008: HF VideoMAEModel.layernorm 可能为 None（config-conditional）
- **Context**: M4 task 4.2 — scale token 实施后第一次跑测试
- **Discovery**: M3 行为靠 `self.backbone(x)` 一把过，HF VideoMAEModel.forward 内部对 `self.layernorm` 做 `if self.layernorm is not None` 守卫。M4.2 加 scale token 时改为手动 `patch_embed → 加 scale_token → 加 pos_emb → encoder → layernorm`，初版直接 `self.backbone.layernorm(x)` 在 layernorm=None 配置下 `TypeError: 'NoneType' object is not callable`。修法：`if getattr(self.backbone, "layernorm", None) is not None` 守卫
- **Evidence**: src/nid_video/models/videomae_nid.py `forward` + 5 个测试在初版下 fail，加守卫后全过
- **Decision rationale**: 与 HF 内部 None 检查对齐（不要假设永远存在）；副产品收益是未来加载 layernorm-disabled 配置（如 VideoMAE pre-train 模式）时不会 crash
- **Status**: persisted in code (no paper exposure expected; documented for future maintainers)

> 🎯 **论文价值标注**
> - **Section**: Implementation Notes (no paper exposure expected)
> - **Use**: 维护笔记。M4.2 加 scale token 时手动展开 forward（patch_embed → 加 scale_token → 加 pos_emb → encoder → layernorm），需对 `self.backbone.layernorm` 加 None 守卫与 HF 内部行为对齐。属于实现细节，对论文叙事不相关。
> - **Cross-link**: M3-003 (q/v_bias loss) / M3-004 (HF state_dict naming) / M3-006 (VideoMAE-Small attention heads) — 共同构成 "HF transformers 5.x VideoMAE adaptation footgun cluster" (q/v bias / state_dict naming / attention heads / layernorm None 四类适配陷阱)
> - **Priority**: LOW


### Finding M4-009: 真实流量 tensor 极端可压缩（0.3% gzip）vs 合成数据（92.6%）
- **Context**: M4 task 4.7 — 100ms ETL 跑到 32 GB 时的磁盘紧张诊断
- **Discovery**: webdataset 用未压缩 `.tar` 存 raw float32 tensor，每 sample 786,432 bytes (= 16×6×32×64×4)，tar 加 header 实际 768 KB/sample。诊断对比 5 个真实 sample 的 gzip 压缩率：
  - 真实数据 npy：平均压到 **2,590 bytes (0.3%)**（极端稀疏）
  - 合成 randn：压到 **728,547 bytes (92.6%)**（密集高斯不可压）
  原因：真实流量稀疏（多数源 IP × 端口组合无流量、空帧多），tensor 大量 0 值，gzip 友好；合成 randn 每个 cell 都是密集噪声。100ms 全集 v2 ≈ 82 GB 未压缩，理论 gzip 后 ≈ 0.5-1 GB。**当前选择不压缩**（833 GB free，IO 速度优先）
- **Evidence**: 4.7 阶段诊断脚本输出（5-sample probe）+ data/processed/cicids2017_dt100ms_v2 实际 disk usage 82 GB / 113 shards
- **Decision rationale**: M4 默认 `.tar` 不压缩（训练 IO 速度 > 磁盘节省）；M5+ 跨数据集如磁盘吃紧再加 `.tar.zst`（zstd 解压快、磁盘减 250×）；m3_perf.md / etl_performance.md 估算修订（之前的 "150 KB/sample" 来自一个不存在的"tar gzip"路径，幽灵数字）
- **Status**: documented; format change deferred to M5+

> 🎯 **论文价值标注**
> - **Section**: Implementation Notes / Computational Considerations
> - **Use**: 维护笔记 + 未来扩展依据。真实数据 npy 平均压到 2,590 bytes (0.3%)，合成 randn 压到 728,547 bytes (92.6%)——250× 差异源自真实流量稀疏性（多数源 IP × 端口组合无流量、空帧多）。当前选择不压缩（833 GB free，IO 速度优先），M5+ 跨数据集如磁盘吃紧再加 .tar.zst（zstd 解压快、磁盘减 250×）。也是 m3_perf.md / etl_performance.md 估算修订（之前 "150 KB/sample" 来自不存在的 tar gzip 路径——幽灵数字）。
> - **Cross-link**: M2-009（ETL 性能档案）+ etl_performance.md 磁盘 sizing 表的 root-cause 解释；M5+ 跨数据集 disk budget 决策的依据。
> - **Priority**: LOW


### Finding M4-010a: 三个 silent failure 修复链路下游 F1 验证（M4.8 真训练首跑）
- **Context**: M4 task 4.8 — round_robin 修复后真训练 1 epoch（77,615 train + 16,463 val + 16,705 test）
- **Discovery**: 三个 silent failure 修复（TRANSITION-005 TZ / M4-001 12h-format / M4-005 split index-based）的下游 F1 全部从 v1 阶段的 0 恢复到非零，关键攻击类有强信号：
  - **DDoS** (12h fix 下游): v1=0 windows → v2 训练 F1=**0.7824** (Prec=0.84, Rec=0.73, AUROC=1.00)
  - **SSH-Patator** (12h fix 下游): v1=0 → v2 F1=**0.1025** (Prec=0.38, Rec=0.06, AUROC=0.94 显示信号充足)
  - **DoS-Slowloris** (split fix 下游): split v1 100/0/0 → v2 1118/239/241 → 训练 F1=**0.7407** (Prec=0.96, Rec=0.60, AUROC=0.98)
  - **Heartbleed** (双 fix 下游): v1=0 → v2 F1=**0.9549** (AUROC=1.00) — 长连接攻击 + 充足训练样本（1505 windows）的最佳 case
  整体 single-epoch baseline 数字：**accuracy=0.9470 / macro_f1=0.4474 / auroc_macro=0.7826**（slow_exhausted 修复前 0.31 → +44.6%）
- **Evidence**: outputs/run_20260430_223105/ckpt/best.pt + run log + Evaluator pretty_print 13×13 confusion matrix + docs/v1_vs_v2_comparison.md
- **Decision rationale**: 三个 silent failure 各自有独立的 fix mechanism (TZ via zoneinfo / 12h via hour-based inference / split via index-based partition)，这条 finding 把它们的下游训练验证打包成一份"修复链路完整"的硬证据
- **Status**: documented in M4 完成报告 + outputs/run_20260430_223105 ckpt; **HIGHEST PRIORITY** candidate (3-fix downstream verification → paper Methodology + Discussion main flagship evidence)

> 🎯 **论文价值标注**
> - **Section**: Discussion (Silent Failure Detection Chain — Flagship Downstream Verification) + Methods (Empirical Validation)
> - **Use**: **silent failure detection chain 的训练级最终验证、论文 Discussion 章节旗舰证据**。三个独立 silent failure 修复（TRANSITION-005 TZ / M4-001 12h-format / M4-005 split index-based）的下游 F1 全部从 v1 阶段的 0 恢复到非零，关键攻击类有强信号：
>   - **DDoS**（12h fix 下游）: v1=0 windows → v2 训练 F1=**0.7824**（Prec=0.84, Rec=0.73, AUROC=1.00）
>   - **SSH-Patator**（12h fix 下游）: v1=0 → v2 F1=0.1025（Prec=0.38, Rec=0.06, AUROC=**0.94** 显示 representation 信号充足）
>   - **DoS-Slowloris**（split fix 下游）: split v1 100/0/0 → v2 1118/239/241 → 训练 F1=**0.7407**（Prec=0.96, Rec=0.60, AUROC=0.98）
>   - **Heartbleed**（双 fix 下游）: v1=0 → v2 F1=**0.9549**（AUROC=1.00）——长连接攻击 + 充足训练样本（1505 windows）的最佳 case
>   整体 single-epoch baseline 数字：**accuracy=0.9470 / macro_f1=0.4474 / auroc_macro=0.7826**。**这是从 silent failure 发现 → 修法 → 守恒律证明 → 端到端 ETL 验证 → 真训练 F1 表现的完整闭环**——论文 Discussion 章节"silent failure detection chain"的训练级证据。
> - **Figure idea**: 三层证据塔的视觉化——底层 packet-level 6.1M 包对照表（TRANSITION-005）+ 中层 ETL-level v1 vs v2 守恒律表（M4-002）+ 顶层 training-level F1 表（本条），每层独立验证、量级递进、向下可追溯。或者更紧凑：单 figure 把 4 个关键攻击类的 v1 windows / v2 windows / v2 F1 三列放在一起，显示"每个 silent failure 都从 0 恢复到可观察的训练表现"。
> - **Quote candidate**: "We verify the three silent-failure fixes end-to-end at the training level: classes that registered zero windows in v1 (SSH-Patator, DDoS, PortScan, Heartbleed) and the class collapsed by buggy time-position splits (DoS-Slowloris) all recover non-zero F1 in the M4.8 single-epoch baseline. Heartbleed reaches F1=0.9549 (AUROC=1.00); DDoS reaches F1=0.7824; DoS-Slowloris reaches F1=0.7407. Overall: accuracy=0.9470, macro-F1=0.4474, AUROC-macro=0.7826—setting the baseline for M5 multi-epoch comparisons."
> - **Risk if missed**: 这条是 silent failure detection chain 的最终闭环。不写则前面所有修复（TRANSITION-005 / M4-001 / M4-002 / M4-005）都停在"修法看起来对、守恒律满足、ETL 输出对"——但 reviewer 可以质疑"修复后下游训练真的 work 吗"。本条把整个修复链路从"data correctness"延伸到"empirical training performance"，封闭最后的可质疑面。
> - **Cross-link**: TRANSITION-005（packet-level ground truth, 6.1M 包对照表）+ TRANSITION-007（ETL-level 端到端验证）+ M4-001（12h-without-AM/PM 发现）+ M4-002（三守恒律证明）+ M4-005（split index-based 修法）+ 本条 = silent failure detection chain 核心六件套；M3-007（显存假设修正——485 MB peak GPU 实测验证）+ M3-009（throughput 真训练验证）+ M1-001（消费级硬件可复现基线）共同构成"消费级硬件 + 工程严谨性"双线叙事；M4-010b（single-epoch 已知局限）是本条 ceiling 的下游说明。
> - **Priority**: HIGHEST


### Finding M4-010b: M4.8 single-epoch baseline 已知局限（Bot/GoldenEye F1=0 with AUROC>0.7）
- **Context**: M4 task 4.8 — 验收线核对发现 2 个攻击类 F1=0 但 AUROC 显示信号充足
- **Discovery**: 两个攻击类在单 epoch baseline 下出现 F1=0 / AUROC>0.7 的不一致：
  - **Bot** (n_test=8 in this eval split): F1=0.0000, AUROC=0.7411
  - **DoS-GoldenEye** (n=111): F1=0.0000, AUROC=0.9366
  解释：模型已学到 representation（AUROC>0.7 表示对该类的 ranking 能力 > 随机），但 argmax 阈值在 13-class softmax 下输给多数类（BENIGN/Hulk）。两个攻击类是数据集中最稀疏的（Bot 全集仅 44 windows、GoldenEye 368 windows），加 M4-007 的 cosine decay 早终止使后半段 lr=1.5e-06 plateau，稀有类几乎没机会再校准 logits 阈值。
  修复路径（M5+）：
  1. focal loss / class-balanced cross-entropy 让稀有类 logit 校准
  2. M4-007 fix → cosine decay 走完整 4853 grad steps，后半段仍有有效 lr 训练阈值
  3. 多 epoch 训练（M4 是 single-epoch baseline，M5 默认 ≥ 5 epochs）
  实证驱动豁免规则：当 F1=0 且 AUROC > 0.7 时，归类为"representation 学到 + single-epoch 阈值未收敛"，与 PortScan 极稀疏豁免同等待遇——这条规则是 M4.8 验收线层面的修订（不是放低标准）
- **Evidence**: outputs/run_20260430_223105 confusion matrix（Bot/GoldenEye 行）+ M4.8 训练日志 lr trace
- **Decision rationale**: M4 旗舰验收 ≠ "全 F1>0"；接受 AUROC>0.7 + single-epoch 的实证驱动豁免；M5 多 epoch + class reweight 后预期这两类 F1>0.1
- **Status**: documented for M5 followup; **MEDIUM PRIORITY** candidate (single-epoch limitation + future-fix path → paper Discussion / Limitations)

> 🎯 **论文价值标注**
> - **Section**: Discussion (Limitations) + Future Work (M5 multi-epoch + class reweight)
> - **Use**: M4-010a baseline 数字的"已知局限"诚实披露——Bot (n_test=8) F1=0 / AUROC=0.7411，DoS-GoldenEye (n=111) F1=0 / AUROC=0.9366。AUROC>0.7 表明模型已学到 representation，argmax 阈值在 13-class softmax 下输给多数类（BENIGN/Hulk）。两类是数据集中最稀疏的（Bot 全集 44 windows / GoldenEye 368 windows）+ M4-007 cosine decay 早终止使后半段 lr=1.5e-06 plateau，稀有类几乎没机会校准 logits 阈值。修复路径清晰（focal loss / class-balanced CE / M4-007 fix / 多 epoch 训练），M5 多 epoch + class reweight 后预期这两类 F1>0.1。**实证驱动豁免规则**：F1=0 + AUROC>0.7 = "representation 学到 + single-epoch 阈值未收敛"，与 PortScan 极稀疏豁免同等待遇——这是 M4.8 验收线层面的修订（不是放低标准）。
> - **Quote candidate**: "Two classes—Bot (44 windows total, 8 in test) and DoS-GoldenEye (368 windows)—register F1=0 in the single-epoch baseline despite AUROC of 0.74 and 0.94 respectively. We interpret this as 'representation learned, argmax threshold not yet calibrated': the model ranks these classes informatively but the 13-class softmax threshold is dominated by majority classes. The pattern is consistent with class sparsity compounded by the cosine-decay coupling described above; we expect M5 multi-epoch training with class-balanced loss to recover non-zero F1."
> - **Risk if missed**: Reviewer 看到 Bot/GoldenEye F1=0 → 质疑"single-epoch baseline 不可信"。本条提供"已知系统性原因 + AUROC 信号充足 + 修复路径清晰"答案，把"baseline 缺陷"转化为"已识别的 single-epoch limitation"。
> - **Cross-link**: M4-010a（M4.8 macro_f1=0.45 baseline）的下游说明；M4-007（cosine decay 早终止）的根因解释；M4-003（dominant-rule 双向不对称）+ M2-008（极稀疏类披露）+ TRANSITION-008（dominant-rule 标注偏好）共同支持"稀有类样本量内禀偏好 + single-epoch 训练限制"完整解释；M5 followup 的 prerequisite。
> - **Priority**: MEDIUM

---

# M5 — Baseline 复现阶段

### Finding M5-001: total_steps × epoch_end_strategy 耦合 bug（M4-007 的策略级泛化）

- **Context**: M5.1 任务 — 在准备 M5.2 production training 时发现 M4-007 的 cosine-decay 早终止 bug 不仅适用于 slow_exhausted，也对 round_robin 成立但形态不同；总公式需按 epoch_end_strategy 分支。
- **Discovery**: M4-007 描述的耦合在 round_robin 下表现为 `total_steps = ceil(train_n_fast / B) × num_epochs` 即 4853 grad steps for 1 epoch，但实际 round_robin 终止条件让 fast 流单独耗尽即 epoch 结束 — 与 fast-only 公式重合（无 silent step under-count）。但若 `mix_ratio ≠ 0.5` 就 silent under-count；若改回 slow_exhausted 则用 fast-only 公式会让 cosine 在 grad step 2425 (50%) 完成、后半段 lr=1.5e-6 plateau。修法是 strategy-conditional：
  - **round_robin**: anchor_n = ceil(train_n_fast / mix_ratio) → 公式覆盖任意 mix_ratio
  - **slow_exhausted**: anchor_n = train_n_slow / (1 − mix_ratio)
  - **single-scale 路径**（mix_ratio=None）: 不变，与 M3 一致
- **Evidence**: `scripts/train.py::_compute_total_steps` + `tests/test_train_cli.py` 5 tests 钉死 round_robin/slow_exhausted/single-scale/multi-epoch/未知 strategy 边界；commit `ea651d4`（feat(trainer): M5.1 fix total_steps × epoch_end_strategy coupling）
- **Decision rationale**: prompts/04_M4_real_training.md M4.8 后续讨论中由 Claude conversation 评审拍板分支化公式；M5.1 实施。
- **Status**: persisted in code + CI（5 测试覆盖 4 条分支 + 1 条 fail-fast）

> 🎯 **论文价值标注**
> - **Section**: Implementation Notes / Reproducibility (no major paper exposure expected)
> - **Use**: 维护笔记。`total_steps × epoch_end_strategy` 耦合公式按 strategy 分支 (round_robin / slow_exhausted / single-scale)。M4-007 的 cosine decay 早终止 bug 在 M5.1 阶段被 strategy-conditional 修法泛化解决 (commit ea651d4)。论文一般不会写到这种 framework-specific 细节,但代码层已被 5 测试守护。
> - **Risk if missed**: 复现者运行多尺度训练不修此公式,实际 grad steps 可能超出 cosine 设计预期,后半段训练 lr=1.5e-6 plateau,与论文 reported 数字不匹配。
> - **Cross-link**: M4-007 (cosine decay 早终止 bug 原始发现) 的策略级泛化修法;M4-006 (round_robin epoch 终止策略) 的 schedule-side 配套修复;M5-003 (eval-side 解耦) 共同构成 "data loader epoch strategy 三件套"。
> - **Priority**: MEDIUM

### Finding M5-002: vanilla CE noise-free ceiling 0.4677 + Bot AUROC step-collapse 轨迹（epoch 1-3 之间）

- **Context**: M5.3.5 noise-free retrofit — 三个里程碑的训练 ckpt 在统一 no_cycle eval 下重新跑出 noise-free 数字，构成可比较的训练-预算 trajectory。
- **Discovery**: 三个 vanilla CE 训练在统一 no_cycle eval 下：
  - **M4.8 1ep**: combined 0.3324 / Bot AUROC 0.7247（under-fit, Bot 信号偶然保留）
  - **M5.1 3ep**: combined 0.4230 / Bot AUROC 0.4237（已 step-collapse）
  - **M5.2 10ep**: combined 0.4677 / Bot AUROC 0.4077（vanilla CE 真实 ceiling）
  Bot AUROC 在 epoch 1 → 3 之间发生 step-collapse（−0.30），不是 gradual decay；epoch 3 → 10 仅 −0.016。Implication: focal/class-reweight 干预若放在 epoch 3 后启动只能 "correct" 已塌缩的 logit calibration（无法恢复 representation），必须从 epoch 0 即介入"prevent"塌缩。
- **Discovery additional**(N+2 closeout 修订):Bot AUROC collapse 形态 run-to-run 间不同 — M5.1 是 epoch 1→3 间 sharp drop(0.7247 → 0.4237);M5.2 是 6-epoch gradual decline(epoch 0 = 0.6728 → epoch 9 = 0.4077);dim 1 random Phase 1 trajectory **不 collapse**(全程 sustained > 0.66 across all 10 epochs)= reverse evidence。Collapse 形态非 constant,具体形态依 representation × loss × training-stream 联合;原 "M5.1 单步 sharp drop" 描述是 1 个 instance 而非 generic claim。
- **Evidence additional**:`outputs/run_20260501_162117/metrics/per_epoch.json`(M5.2 retrofit 全程 trajectory)+ `outputs/run_20260507_205921/metrics/per_epoch.json`(dim 1 random Phase 1 sustained > 0.66 reverse evidence)。
- **Evidence**: `outputs/run_20260430_223105/m4_8_rerun/eval_metrics.json` + `outputs/run_20260501_143946/m5_1_rerun/eval_metrics.json` + `outputs/run_20260501_162117/m5_3_rerun/eval_metrics.json` + `docs/m5_baseline_trajectory.md` §"Bot AUROC trajectory"; commits `266d56c`（generalize baseline_rerun.py + retrofit M4.8/M5.1）+ `0b79ca0`（focal loss + M5.4 P1）
- **Decision rationale**: 三 ckpt 数字本身一直存在；M5.3.5 把它们 normalised 到 no_cycle eval 才能比较。Step-collapse 物理含义直接驱动 M5.4 from-scratch focal-loss 路径决策。
- **Status**: documented in `docs/m5_baseline_trajectory.md`; M5.4 路径选择的 root cause anchor

> 🎯 **论文价值标注**
> - **Section**: Discussion (Training Dynamics) + Methods (Loss Function Motivation)
> - **Use**: vanilla CE 在严重不平衡 NIDS 下的 representation collapse 实证。三 ckpt noise-free macro_f1 trajectory (M4.8 0.3324 / M5.1 0.4230 / M5.2 0.4677) 证明 vanilla CE ceiling 在 epoch 3 已基本撞死,长训仅多涨 +0.045。Bot AUROC 阶跃 collapse (epoch 1→3 跌 0.301,epoch 3→10 仅再跌 0.016) 证明 representation 退化集中在 critical window epoch 1-3,"focal loss 必须 from epoch 0 介入 prevent 而非 correct" 的物理直觉得以确立。这条直接驱动 M5.4 from-scratch 路径选择 (不 fine-tune from M5.2 ckpt)。
> - **Risk if missed**: 不写则 M5.4 from-scratch 设计选择像 ad-hoc 决定。论文 Methods 章节对 "为什么 focal 从 scratch 而不是 fine-tune from collapsed ckpt" 缺乏实证理由。
> - **Cross-link**: M5-003 (noise-free eval 是这条 finding 的 measurement prerequisite);M5-004 (focal loss 设计的下游响应);M4-010b (single-epoch limitation 的多 epoch 升级版)。
> - **Updated (N+2 closeout)**: collapse 形态 not constant,具体形态依 representation × loss × training-stream 联合;论文 narrative 限定 "M5.1 trajectory shows step-collapse" 而非 "Bot AUROC collapse is step-shape generic"。
> - **Priority**: HIGHEST

### Finding M5-003: no_cycle eval strategy + cycling-induced metric inflation maturity-dependent

- **Context**: M5.3 任务 — 把 epoch_end_strategy 与 eval_strategy 解耦，引入 `no_cycle` 选项；M5.3.5 retrofit 验证三 ckpt 在两策略下的差异。
- **Discovery**: round_robin eval 让 slow 流（每次 ~1693 样本）在每个 eval pass 内被循环 ~10× 直到 fast 流（~16463）耗尽，每 cycle reseed shuffle = 不同 batch context 看每个 slow 样本 → softmax-averaging at metric computation time = unintended TTA。Cycling delta（reported macro_f1 − no_cycle macro_f1）与 grad_steps 反相关：
  - M4.8 1ep（4853 steps, ln=8.49）: Δ = −0.115
  - M5.1 3ep（14559 steps, ln=9.59）: Δ = −0.088
  - M5.2 10ep（48530 steps, ln=10.79）: Δ = −0.047
  Log-linear slope ≈ +0.030 per ln-step（fits 三点 within 0.007）。Mechanism: per-sample logit variance 高于 mature checkpoint，TTA-by-cycling benefit 与 logit variance 成正比 → 早期 ckpt 拿大头、晚期 ckpt 拿小头。**Δ 不是 constant offset** — round_robin reported 数字与训练成熟度耦合。
- **Evidence**: `docs/m5_baseline_trajectory.md` §"Cycling-delta is training-maturity dependent" + 三 retrofit bundles `m{4,5}_X_rerun/eval_metrics.json`; `--eval-strategy` CLI flag in `scripts/train.py` + `--epoch-end-strategy` decoupled at trainer init; commits `345fc5d`（add no_cycle epoch end strategy）+ `266d56c`（generalize baseline_rerun.py）
- **Decision rationale**: 论文要 publishable baseline 数字必须 noise-free；round_robin 是 training-end terminator 但 eval policy 应该 deterministic。`baseline_rerun.py` 是 retrofit 工具 — 任意已 ckpt 都可以 noise-free 重新评估。
- **Status**: persisted in code（CLI flag + scripts/baseline_rerun.py）+ documented in trajectory.md

> 🎯 **论文价值标注**
> - **Section**: Discussion (Evaluation Methodology Footgun — flagship) + Reproducibility
> - **Use**: **NIDS 文献里几乎无人讨论的 evaluation methodology silent failure**。round_robin eval 让 slow 流被 cycle ~10× per pass + 每 cycle reseed shuffle = unintended test-time augmentation;不仅引入 metric inflation (M5.2 round_robin 0.5143 vs no_cycle 0.4677 = +0.047),且 inflation 量级与 log(grad_steps) 反相关 (slope ≈ +0.030 per ln-step,trajectory M4.8 -0.115 / M5.1 -0.088 / M5.2 -0.047)。这意味着 **任何 NIDS 论文用 cycling-style eval 比较不同 epoch budget 的模型都是 invalid comparison** —— 不同成熟度模型的 inflation 不同,trajectory 形状被人为扭曲。这是与 silent failure detection chain (M4-001 + TRANSITION-005) 同性质的方法学 footgun。
> - **Risk if missed**: 不写则 NIDS 复现者可能在 round_robin eval 下报告虚高数字,论文社区水位被人为拉高。这条是 community service 价值。
> - **Cross-link**: M4-006 (train-side round_robin 修法) + M5-001 (schedule-side total_steps 修法) + 本条 (eval-side no_cycle 修法) 共同构成 "data loader epoch strategy 三件套";与 M4-001 / TRANSITION-005 / M4-002 / M4-010a silent failure detection chain 同性质 evaluation footgun,论文 Discussion 章节并列叙事;M5-002 (noise-free trajectory 数字) 的 measurement prerequisite。
> - **Priority**: HIGHEST

### Finding M5-004: focal loss prevent vs correct + M5.4 P2 reweight saturation（4 sentinels still fail）

- **Context**: M5.4 Phase 1（focal γ=2 from-scratch）+ Phase 2（+ inverse_sqrt α + head_lr ×5）+ 决策接受 P2 作 baselines fairness contract anchor。
- **Discovery**: Phase 1 焦点损失单独使用 → noise-free combined 0.4584（**低于** vanilla CE ceiling 0.4677 by −0.009）；4/4 sentinels（Bot/GoldenEye/PortScan/SSH-Patator）FAIL；Bot per-class AUROC 0.5060。Phase 2 加上 inverse_sqrt class-frequency reweight + head LR ×5 → noise-free combined **0.4756**（+0.0079 over CE，+0.0172 over Phase 1）；4/4 sentinels 仍 FAIL；Bot AUROC 0.4968。中等稀疏 class（n_train > 200）reweighting 单调改进（GoldenEye F1 0.2278 → 0.4130；PortScan F1 0.5957）；极稀疏（n_train < 50）loss-level 不可救（Bot F1 仍 0.0；Web Attack/Infiltration n_val=0 by construction）。
- **Discovery additional**(N+2 closeout 修订):round 1 期间 dim 1+2+4 共 8 forward cells(dim 1 random + dim 1 SSv2 + dim 2 C=4 + dim 4 Cell A/B/C/D + 主方法 P2 reuse)GoldenEye F1 trajectory 全部 oscillate(magnitude [0.06, 0.44] between epochs);α reweight 抬高 attractor mean 但 oscillation universally present。8/8 retrofits + forward 全 confirm noisy-attractor pattern(α 非消除 lever)。
- **Evidence additional**:8 cells per_epoch.json GoldenEye F1 trajectory(详 `docs/m5_10_pretrained_ablation.md` + `docs/m5_10_motion_channel_ablation.md` + `docs/m5_10_scale_token_ablation.md` Phase 1 notable trajectory features 段)。
- **Evidence**: `outputs/run_20260502_134735/m5_4_eval/eval_metrics.json`（P1）+ `outputs/run_20260502_184512/m5_4_phase2_eval/eval_metrics.json`（P2）+ `docs/m5_baseline_trajectory.md` §"M5.4 Phase 1" / §"M5.4 Phase 2"; commits `0b79ca0`（focal loss + P1）+ `8ebd3c8`（inverse_sqrt + head_lr + P2）+ `1d1a61e`（accept P2 + 推 M5.10 数据级 intervention）
- **Decision rationale**: M5.4 任务 spec — 评估 loss-level 单独是否能突破 vanilla CE ceiling + 救稀疏 class；结论 "yes by 0.008 combined, no for 4/4 sentinels"；data-level intervention（rare-class oversampling）推 M5.10 ablation；M5.4 P2 接受作为 M5.5 baselines 的 fairness contract anchor。
- **Status**: documented in `docs/m5_baseline_trajectory.md`; **M5.5 baselines contract anchor**

> 🎯 **论文价值标注**
> - **Section**: Methods (Loss Design) + Discussion (Loss-level Optimization Limits)
> - **Use**: focal loss 在 NIDS 不平衡场景的实证 + 极稀疏类的物理边界。Phase 1 focal γ=2 from-scratch combined 0.4584 比 vanilla CE 0.4677 退步 0.009 —— focal alone 不够。Phase 2 + inverse_sqrt α + head_lr×5 拿到 0.4756 (仅 +0.0079 over CE)。物理边界:**当前架构 + 训练规模下 class reweighting 临界 sample size 在 n_train ~100-200 之间** —— GoldenEye n=257 单调修复 (0.2278→0.4130),Bot n=30 不可救 (F1=0 across all configs)。论文 Limitations 章节诚实声明 "loss-level fix 对极端稀疏 (n<50) 类不够,需要 data-level 干预";M5.10 ablation 章节将做 data-level 对照。
> - **Risk if missed**: Reviewer 看到主方法 macro_f1 < 0.50 → 质疑 "loss 设计是不是没做好"。本条把 Phase 1+2 的实证 + 物理边界 + 后续 ablation 路径锁定,将 "M5.4 P2 不达 0.55" 从 limitation 框定为 "loss-level fix 的物理上限"。
> - **Cross-link**: M5-002 (vanilla CE ceiling 是 P2 比较的对照);M5-005 (head_lr×5 在 K400 上有效是 Phase 2 lever 的 K400-side 实证);M2-008 / M4-003 / M4-010b (极稀疏类的 "诚实数据处理" 叙事簇)。
> - **Updated (N+2 closeout)**: 8 forward cells(round 1)+ 11 retrofits = 19 cells GoldenEye F1 oscillation universal;α reweight 不消除 oscillation,只抬高 attractor mean。
> - **Priority**: HIGH

### Finding M5-005: head_lr × pretrained-status 耦合 — Path B fairness contract（R1 vs R1.5 ablation 实证）

- **Context**: M5.5 R1（TimeSformer-Small 0.4836，head_lr ×1 effective due to M5-006 matcher bug）vs R1.5（matcher 修后 head_lr ×5 intentional, 0.4616）ablation。
- **Discovery**: head_lr ×5 在 K400 pretrained backbone 上有效（M5.4 P2 +0.0079 over Phase 1，I3D / R(2+1)D-18 同 contract 也 over main 0.039 / 0.044）；但在 from-scratch random init backbone 上 **HURTS**：R1.5 vs R1 = combined −0.022（0.4836 → 0.4616）/ Bot per-class F1 collapse from 0.0909 → 0.0 / Bot AUROC 0.7151 → 0.5940。Mechanism：K400 backbone 已学到 stable representations，head 需 fast learn new task；random init 都 fresh，head_lr ×5 让 head 在 backbone 学到 useful features 之前 overshoot 到 majority-class boundary。Path B contract 决策：`head_lr_multiplier` 按 pretrained status 分组 — K400 group ×5；random init group ×1。
- **Evidence**: `outputs/run_20260502_232207/m5_5_timesformer_small_eval/eval_metrics.json`（R1，cited per Path B）+ `outputs/run_20260503_121046/m5_5_timesformer_small_eval/eval_metrics.json`（R1.5，ablation supplementary）+ `docs/m5_5_baselines.md` §"R1.5 ablation supplementary" + `docs/m5_baseline_trajectory.md` §"M5.5 R1 → R1.5 forensic finding"; commits `bac6c67`（R1）+ `3187a67`（R1.5）+ `56b029f`（trajectory.md Path B revisions）
- **Decision rationale**: 用户决策 Path B over Path A（"修 matcher + 重训 R1 + 全部 baselines 用 ×5"）— 理由是 R1.5 实证显示 ×5 不 universal good，5 baselines 中 3 个 random init + 2 个 K400 pretrained 应该按 pretrained 状态分组才是 honest fairness contract。
- **Status**: persisted in code（`--head-lr-multiplier` CLI per-baseline）+ documented in `docs/m5_5_baselines.md` + `docs/m5_baseline_trajectory.md`; R1.5 artefact 标 "ABLATION SUPPLEMENTARY — DO NOT cite as baseline"

> 🎯 **论文价值标注**
> - **Section**: Methods (Hyperparameter Design Principle) + Discussion (Pretrained-state Coupling Finding)
> - **Use**: **head LR × pretrained 状态耦合的实证 + Path B fairness contract 的设计依据**。R1 (random init + head_lr×1 effective) macro_f1 = 0.4836;R1.5 (random init + head_lr×5 intentional) = 0.4616 = 退步 0.022。物理:K400 pretrained backbone 已 stable,head 需 fast 学新 task → head_lr > backbone_lr 合理;random init backbone head 与 backbone 都 fresh,head_lr×5 让 head 在 backbone 学到 useful features 之前 overshoot 到 majority-class boundary,反向压制 backbone 的 representation 学习。**这条是 head_lr 设计原则的实证而非工程 anecdote** —— 作为 baselines fairness contract 的数据依据,让 "K400 ×5 / random ×1" 分组从 ad-hoc 决定升级为有实证支撑的 design principle。
> - **Risk if missed**: Reviewer 看 baselines 表 → 质疑 "为什么主方法 + I3D + R(2+1)D 用 head_lr×5 而 TimeSformer + C3D + ConvLSTM 用 ×1,这不是 unfair contract 吗"。本条提供 R1 vs R1.5 的实证答案:head_lr×5 不是 universal good,按 pretrained 状态分组才反映物理。
> - **Cross-link**: M3-001 (patch_embed 5.28× norm ratio preservation 的 pretrained 状态保留机制) + M3-002 (ignore_mismatched_sizes silent failure) 共同构成 "pretrained 状态决定下游设计" 叙事簇;M5-006 (matcher bug 是本条 ablation 数据的产生机制);M5-007 (baselines fairness contract 的 head_lr 分组依据)。
> - **Priority**: HIGHEST

### Finding M5-006: trainer head matcher prefix bug + segment-match fix（multi-baseline silent failure）

- **Context**: M5.5 R1.5 forensic — 在准备 R2 baselines 时发现 R1 训练 log 报告 `configured_head_lr=7.50e-04` 但 head_param_count 实际为 0。
- **Discovery**: `trainer._build_param_groups` 用 `startswith("classifier.", "scale_embedding.")` 匹配 head 参数。设计针对 VideoMAE 的扁平 `model.classifier`；对 HF wrapper（TimeSformer 的 `backbone.classifier.*` — 5005 params）/ pytorchvideo I3D（`blocks.6.proj.*`）/ torchvision R(2+1)D-18（`backbone.fc.*`）全部静默丢 head — 5005 / 26637 / 6669 head params 全部进 backbone group。Training log 上仍报告 `configured_head_lr=7.50e-04`（因为 group[1] 的 lr 字段未变），但 group[1] 是 empty list，head_lr 实际从未应用到任何参数。修法是 segment-based matcher：分隔符切分 param name，检查 ancestor module name 是否 `∈ {classifier, scale_embedding, fc, proj}`（避免 substring false positive 如 `pre_classifier.weight`）。
- **Evidence**: `src/nid_video/trainer/trainer.py::_build_param_groups`（startswith → segment match）+ `tests/test_trainer.py` +2 tests（`test_build_param_groups_matches_head_in_nested_backbone` HF nested + `test_build_param_groups_matches_torchvision_fc_and_pytorchvideo_proj` mock 命名）；commit `3187a67`（fix head matcher and retrain timesformer-small for m5.5 round 1.5）
- **Decision rationale**: Sanity check 加入 baseline 实施流程 — pre-train preflight 必须 log `head_param_count` + `optimizer.param_groups[1]['lr']` 实际值，验证非空后再启动。R2 4 baselines 实施时全部走过此 sanity（C3D head_n=6669 / ConvLSTM 13325 / I3D 5005 / R(2+1)D 6669）。
- **Status**: persisted in code + CI（segment matcher + 2 nested-discovery tests）+ R2 4 baselines preflight 日志 line; **silent failure detection chain 续集**（M3-002 + M5-006 是 transformer / param-group 的两处 silent failure 模式）

> 🎯 **论文价值标注**
> - **Section**: Discussion (Implementation Footgun cluster) + Reproducibility
> - **Use**: silent failure detection chain 的 PyTorch param group / optimizer 层面续集。`_build_param_groups` 用 `startswith("classifier.", "scale_embedding.")` 匹配,对 HF wrapper (`backbone.classifier.*`) / pytorchvideo (`blocks.6.proj.*`) / torchvision (`backbone.fc.*`) 静默丢 head;group[1] 报告 lr=7.5e-4 但 empty list,optimizer 实际跑 head_lr=1× backbone_lr。这条与 M3-002 (ignore_mismatched_sizes 静默丢预训练权重) 同性质 —— **training infrastructure 中 silent failure mode 不会触发 exception,仅通过 sanity verification (head_param_count 验证) 可发现**。修法是 segment-based ancestor-set matcher + sanity preflight。M5.5 R2 4 baselines 全部走过此 sanity,head matcher 工作正确。
> - **Risk if missed**: 复现者 / 后续工作直接复用 trainer 代码 + 不同 backbone wrapper (任何用 `backbone.X.classifier` 嵌套结构的 HF / torchvision / 自建模型),head_lr 静默 bypass。这是 community service 价值的 footgun 警告。
> - **Cross-link**: M3-002 (ignore_mismatched_sizes 静默丢预训练权重) 形成 silent failure mode "transformer / param-group 两处" 模式族,与 silent failure detection chain (M4-001 + TRANSITION-005) 的 "data preprocessing 层面" silent failure 互补 —— 共同构成项目实施透明度叙事;M5-005 (本 bug 副产物是 R1 vs R1.5 ablation 数据)。
> - **Priority**: HIGH

### Finding M5-007: M5.5 cross-baseline 6-row 总览 — K400 vs random group-level gap + R(2+1)D-18 / I3D 双 K400 超主方法 + R(2+1)D-18 唯一 Bot F1 > 0

- **Context**: M5.5 R2 全 4 baselines 完成 + 综合 6-row 比较（commits 135d3e6 / 2d054cb / 23bd16b / e0f29ff / 23d76a0 + 56b029f）。
- **Discovery**: 6 行（M5.4 P2 main + 5 baselines）combined macro_f1：
  - **K400 group**（main 22M / I3D 27M / R(2+1)D-18 31M）平均 0.5034
  - **random init group**（TimeSformer-S 31M / C3D-Small 19M / ConvLSTM 13M）平均 0.4682
  - **Group-level gap +0.035**，但 within-group spread > group gap：K400 group 内 main 0.4756 vs R(2+1)D-18 0.5197 = **+0.044**；random group 内 C3D 0.4464 vs TimeSformer-S 0.4836 = **+0.037**
  关键 sub-findings:
  - **R(2+1)D-18（0.5197）和 I3D（0.5149）双双超越主方法（0.4756）**：同 K400 source、相似参数量（27-31M vs 22M），架构差异是 active mechanism；ResNet-18/50 3D-conv 在 (T=16, 32×64) NID 输入上利用 K400 pretrain 比 22M VideoMAE-S 更有效
  - **Bot F1 仅 R(2+1)D-18 > 0**（0.1429，1/12 correctly classified）；其他 5 行全 0；Bot AUROC trajectory 揭示 architecture 比 pretrained 更决定 rare-class signal preservation：TimeSformer-S R1（random + head_lr ×1）= **0.7151**（唯一 > 0.7）vs main 0.4968 / C3D 0.3755 / ConvLSTM 0.3772 / I3D 0.5341 / R(2+1)D 0.4994
  - **ConvLSTM slow stream 反向**：fast 0.4530（≈ main）但 slow 0.5542（−0.053 vs main 0.6069），与 TimeSformer-S 模式相反（slow +0.019, fast +0.002）— 递归 inductive bias 可能 saturate 在 1s slow 窗口前；OR 2×2 spatial pool between cells（memory budget concession）丢失 low-frequency spatial signal
  - **C3D-Small（0.4464）最弱**：DDoS F1 0.3357 vs ConvLSTM 0.7886 / I3D 0.8018 / R(2+1)D 0.8243 — 8-layer feed-forward conv stack 在 from-scratch + (T=16, 32×64) 上失去 DDoS 特定 signal；ConvLSTM（同 random init，更小 13M）远超 → architecture > pretrained 单独贡献 for DDoS
- **Evidence**: `docs/m5_5_baselines.md`（6-row table + 13×6 per-class F1 grand table + 13×6 per-class AUROC grand table + 4-section findings + Methods draft + Source artefacts table）+ `outputs/run_<6 timestamps>/m5_5_*_eval/eval_metrics.json` 6 套；commit `23d76a0`（add m5.5 baselines doc）+ `56b029f`（trajectory Path B revisions）
- **Decision rationale**: 论文 main method narrative 决策（A: 宣称 cross-architecture 主导 / B: 宣称 K400 + 表征对齐合作 / C: 弱化主方法、强调 multi-scale conditioning 是核心贡献 / D: 重新定位主方法 as "K400 + lightweight + scale-aware" 工程权衡）由用户在 M5.10 ablation 完成后或论文写作前决定；本条 finding 是 6-row 数据 + 4 个 sub-finding 的事实层 anchor，不预设 narrative。
- **Status**: documented in `docs/m5_5_baselines.md` + `docs/m5_baseline_trajectory.md`; **paper Table 1 candidate**; pending paper-writing narrative decision

> 🎯 **论文价值标注**
> - **Section**: Results (Cross-baseline Table 1) + Discussion (Architecture × Pretraining decomposition)
> - **Use**: **论文 Table 1 候选**。6 行 cross-baseline 表 (主方法 P2 + 5 baselines) 含 combined / fast / slow macro_f1 + Bot F1+AUROC + params + wall time + pretrained 状态。Group-level finding:K400 pretrained group 平均 0.5034 vs random init group 0.4682 = +0.035 gap;within-group spread > group gap (K400: 主方法 0.4756 → R(2+1)D 0.5197 = +0.044;random: C3D 0.4464 → TimeSformer 0.4836 = +0.037) → **architecture choice 的影响量级与 pretraining 相当**。Sub-findings: (a) R(2+1)D-18 唯一 baseline 在所有 10 epoch 上 Bot F1 > 0,run-average 0.1404,epoch 0 peak 0.2000;TimeSformer R1 final-epoch Bot F1 = 0.0909 非零但非 sustained;(b) TimeSformer-S R1 唯一 Bot AUROC > 0.7 (0.7151) —— attention-based + random init + head_lr×1 三因素交互的 unique result;(c) ConvLSTM slow stream 反向 (0.5542 < fast 0.4530) —— LSTM 对低帧密度 (1s 间隔) 处理 degraded,对 Idea.md "2D+LSTM 把时间当外挂索引" 的批评提供实证;(d) C3D-Small 在 DDoS F1=0.3357 vs ConvLSTM 0.7886 (同 random init,C3D 19M > ConvLSTM 13M params) —— architecture > pretrained 单独贡献 for DDoS 类型。
> - **Risk if missed**: 不写则 M5.5 6-baseline 实测数据是 outputs/ 散落的 6 个 eval_metrics.json,论文写作时无统一 anchor。docs/m5_5_baselines.md (commit 23d76a0) 已建好统一表,本 finding 把它的论文价值显式标注。
> - **Cross-link**: M5-002 (vanilla CE noise-free ceiling 0.4677 是所有 baselines 的对比基准);M5-005 (head_lr 分组的 fairness contract 依据);Idea.md §1.1 (主方法立意 video > 1D/2D,6 baselines 平均 0.487 > vanilla CE 0.4677 提供 video backbone 范式整体优势的实证);M2-008 / M4-003 / M4-010b (极稀疏类整体叙事簇);M3-007 / M3-008 / M3-009 / M4-010a (消费级硬件可复现叙事簇 —— baselines wall time 1.95-9.10h 全部 8GB 卡内可跑)。
> - **Updated (N+2 closeout)**: sub-finding (a) 数字精化 — R(2+1)D-18 是 sustained Bot F1 > 0 in all 10 epochs(run-average 0.1404);TimeSformer R1 final-epoch non-zero(0.0909)但非 sustained。
> - **Priority**: HIGHEST

### Finding M5-005 v3: Bot AUROC collapse 需要 (head_lr ×5) ∧ (multi-scale training) 双条件 jointly hold（dim 4 4-cell factorial isolation,supersedes M5-005 v2）[HIGHEST]

- **Context**: M5-005 v2 推测 Bot AUROC collapse 与 head_lr 相关。dim 4 4-cell factorial isolation 后清晰化。
- **Discovery**: Bot AUROC collapse 需要 (head_lr ×5) ∧ (multi-scale training) 双条件 jointly hold;任一 broken 则 Bot AUROC ≥ 0.67 preserved。Scale token 在双条件下是 partial mitigator(Bot AUROC 0.4968 中间态)。
- **Evidence**: 5 cells — dim 1 random 0.6743 / dim 4 C(fast-only ×5)0.6715 / dim 4 D(slow-only ×5)0.6931 / dim 4 A(token+ms ×5)0.4968 / dim 4 B(no-token+ms ×5)0.3224;commits `572711d` + `3d830af` + `1d8a2bc` + `5bc6b32`。
- **Status**: open finding, 5-cell evidence anchored, supersedes M5-005 v2 (M5-005 v2 留作 historical record);paper-value annotation deferred to design layer

> 🎯 **论文价值标注**
> - **Section**: Methods §3.5 Fairness contract + Discussion §6.4 Three-way coupling
> - **Use**: Paper §6 旗舰 finding — Bot AUROC collapse 联合条件 (head_lr ×5) ∧ (multi-scale training) jointly held isolated via dim 4 4-cell factorial。任一条件 broken → Bot AUROC ≥ 0.67 preserved。Scale token 是 joint regime partial mitigator(Cell A 0.4968 中间态);supersedes M5-005 v2 "head_lr × pretrained-status" scope。
> - **Quote candidate**: "Bot AUROC collapse to 0.3224 (Cell B, head_lr ×5 + multi-scale + no scale token) is contingent on the joint presence of head_lr ×5 and multi-scale training; ablating either condition restores Bot AUROC to at least 0.67 (Cells C/D), with the scale token providing partial mitigation at 0.4968 under the joint condition."
> - **Figure idea**: 5-cell factor isolation bar chart (5 cells × Bot AUROC + factor decomposition table annotation)
> - **Risk if missed**: Reviewer 问 "Bot AUROC 0.3224 vs 0.6931 跨 cell 1.7× 差异,what mechanism";本 finding 提供 5-cell factorial isolation evidence,without this hp design choice 看起来 ad-hoc。
> - **Cross-link**: M5-005 v2 (superseded historical anchor) + M5-007 (cross-arch context) + M5-016 (scale token specific role) + M6-012 (1D byte partial overturn cross-paradigm)
> - **Priority**: HIGHEST

### Finding M5-008: K400 pretrain transfer = 5-epoch loss-level inductive head start [HIGHEST]

- **Context**: 是否 K400 pretrain 提供超出 final-F1 的 advantage 未 isolate。
- **Discovery**: K400-pretrained cells(M5.5 I3D / R(2+1)D-18 / main P2)epoch 0 per_epoch.json combined + slow macro_f1 ≈ random-init cells(dim 1 random)epoch 5+ 数字。K400 transfer = 5-epoch loss-level inductive head start。
- **Evidence**: `outputs/run_20260504_015958/metrics/per_epoch.json` ep0 vs `outputs/run_20260507_205921/metrics/per_epoch.json` ep5;`outputs/run_20260502_184512/metrics/per_epoch.json` ep0 vs 同 random ep5。
- **Status**: open finding, 2-pair epoch-aligned evidence anchored;paper-value annotation deferred to design layer

> 🎯 **论文价值标注**
> - **Section**: Methods §3.4 Pretraining choice + Discussion §6.2 K400 prior mechanism
> - **Use**: K400 prior 不只下游 final-F1 加分,而是 5-epoch loss-level inductive head start。Paper §3.4 narrative 从 "K400 helps" precision 化为 "K400 = loss-level inductive prior worth 5 epochs of from-scratch training"。
> - **Quote candidate**: "Kinetics-400 pretrained backbones reach at epoch 0 the combined and slow-stream macro_f1 levels random-init backbones achieve only after 5 epochs of training; K400 transfer can be quantified as a 5-epoch loss-level inductive head start rather than a uniform asymptotic lift."
> - **Figure idea**: Figure #10 — K400 head start anchor(8-run grouped bar K400 vs random epoch 0 + arrow annotation "K400 epoch 0 ≈ random epoch 5+")
> - **Risk if missed**: Paper §3.4 K400 choice 看起来是 ad-hoc convention rather than quantified design choice;reviewer 问 "K400 pretrain 与 random init Δ 量化"。
> - **Cross-link**: M5-013 (SSv2 反例 K400 specific not video-generic) + M5-015 (K400 slow-stream 主受益者) + M6-008 (12-cell cross-paradigm hierarchy anchor)
> - **Priority**: HIGHEST

### Finding M5-013: SSv2 pretrain 对 NID corpus 零 transfer(combined Δ +0.003 within noise + Bot AUROC −0.263)— Idea.md v1 §2.2.3 corpus-agnostic claim 反例 [HIGHEST]

- **Context**: Idea.md v1 §2.2.3 假设 video pretrain(K400/SSv2)学到 motion 语义 transferable,corpus-agnostic claim。
- **Discovery**: dim 1 SSv2 combined 0.4413 vs random 0.4386 = Δ +0.003(within noise);Bot AUROC 0.4115 vs random 0.6743 = −0.263。K400 transfer(+0.037 combined)corpus-specific 非 video-pretrain-generic。
- **Evidence**: dim 1 cells 主表 + commit `92d2055` + `outputs/run_20260508_213702/m5_10_ssv2_videomae_eval/eval_metrics.json`。
- **Status**: open finding, evidence anchored, conflicts with Idea.md v1 §2.2.3 claim;Idea.md 文本 revision deferred to round 1 closeout cross-doc task;paper-value annotation deferred to design layer

> 🎯 **论文价值标注**
> - **Section**: Methods §3.4 + Discussion §6.2 + Limitations §7.4
> - **Use**: Idea.md v1 §2.2.3 corpus-agnostic claim 反例。SSv2 同 video pretrain corpus,但 NID transfer ≈ 0(combined Δ +0.003 within noise);Bot AUROC 比 random 还低 −0.263。**K400 transfer specific 而非 video-pretrain generic — corpus 选择是 active mechanism**。Idea.md §2.2.3 v2 narrative anchor + paper §3.4 + §6.2 corpus-specific transfer evidence + §7 honest scope acknowledgment。
> - **Quote candidate**: "Something-Something-V2 pretrained backbones contribute +0.003 combined macro_f1 over random initialisation on the NID task — within noise — while Kinetics-400 pretraining contributes +0.037 (an order of magnitude larger). The asymmetry locates K400 transfer in corpus-specific semantic content rather than a generic 'video pretrain helps' inductive prior."
> - **Figure idea**: 3-cell pretrained-source per-class F1 spider chart(random / K400 / SSv2)anchored at paper §6.2
> - **Risk if missed**: Idea.md v1 over-claim "video pretrain transfer 通用" 未被 SSv2 反例 retract → reviewer 用 SSv2 数据 attack。
> - **Cross-link**: M5-008 (K400 head start mechanism) + M5-015 (K400 slow-stream specific lift) + M6-005 (M6.3 IN vs RN ResNet-18 within-noise 跨 paradigm pretrain transfer cluster) + Idea.md §2.2.3 v2 narrative
> - **Priority**: HIGHEST

### Finding M5-014: motion channels(ch5+ch6) + scale token combined macro_f1 贡献 ≈ 0 [HIGH]

- **Context**: motion channels(ch5+ch6) + scale token 设计时假设为 explicit inductive bias 贡献项。
- **Discovery**: dim 2 C=4 vs C=6 combined Δ −0.007;dim 4 Cell B vs A combined Δ +0.0004。两个组件 combined macro_f1 within noise 贡献 ≈ 0。Actual load-bearing role 在 secondary metrics(slow stream + Bot AUROC,详 M5-015 / M5-016)。
- **Evidence**: dim 2 commit `0aefac4` + dim 4 commit `5bc6b32`;`outputs/run_20260510_091547/m5_10_c4_videomae_eval/eval_metrics.json` + `outputs/run_20260510_154227/m5_10_b_videomae_eval/eval_metrics.json`。
- **Status**: open finding, 2-cell pair Δ anchored;paper-value annotation deferred to design layer

> 🎯 **论文价值标注**
> - **Section**: Methods §3 framing reframe + Discussion §6.5 Pareto navigators
> - **Use**: Motion channels(C=4 vs C=6)combined macro_f1 contribution +0.0049 within noise;scale token combined contribution +0.0004 within noise。Paper §3 framing 从 "novel contributions" 改为 "Pareto navigators / secondary-metric load-bearing":motion channels 维持 slow-stream macro_f1 + per-class signature;scale token 是 joint regime Bot AUROC stabilizer(M5-016 detail)。诚实 negative finding 是 paper §3 framing 修订的 evidence anchor。
> - **Quote candidate**: "Motion channels and the scale token contribute near-zero combined macro_f1 (within ±0.005 noise band); their load-bearing role lies at the secondary-metric layer — slow-stream macro_f1 preservation and Bot AUROC stabilization respectively. We reframe these design choices as Pareto navigators rather than primary-metric contributions."
> - **Figure idea**: 5-cell factor decomposition bar chart(combined / fast / slow / Bot AUROC)with motion-channel + scale-token isolation
> - **Risk if missed**: Paper §3 over-claim motion / scale-token as primary contributions → reviewer ablation 不动 challenges。
> - **Cross-link**: M5-005 v3 (三维耦合) + M5-016 (scale token specific role) + M5-017 (single-stream OOD asymmetry — Pareto navigator framework supplementary)
> - **Priority**: HIGH

### Finding M5-016: scale token 在 K400+×5+multi-scale 联合 regime 下 stabilize Bot AUROC(Bot AUROC Δ −0.175 with combined ≈ 0)[HIGH]

- **Context**: M5-014 后 scale token combined role ≈ 0;但 dim 4 4-cell factorial 显示 token vs no-token 在 multi-scale + ×5 regime 下 Bot AUROC 有显著差异。
- **Discovery**: dim 4 Cell A(token+ms+×5)Bot AUROC 0.4968 vs Cell B(no-token+ms+×5)0.3224 = Δ −0.175 full collapse。Scale token 在 K400+×5+multi-scale 联合 regime 下 stabilize Bot AUROC,combined ≈ 0 contribution。
- **Evidence**: dim 4 commit `5bc6b32` + Cell A/B eval_metrics.json + dim 4 doc `docs/m5_10_scale_token_ablation.md` Bot AUROC 段落。
- **Status**: open finding, single-cell-pair Bot AUROC Δ anchored;paper-value annotation deferred to design layer

> 🎯 **论文价值标注**
> - **Section**: Methods §3.4 scale token reframe + Discussion §6.5
> - **Use**: Scale token combined macro_f1 contribution +0.0004,but Bot AUROC stabilization +0.175(Cell A 0.4968 vs Cell B 0.3224 collapse)。在 K400 + ×5 + multi-scale 联合 regime 下 specific load-bearing role 是 Bot rare-class signal stabilizer 不是 multi-scale lever。Paper §3.4 framing 转向 "Pareto navigator for Bot rare-class secondary-metric"。
> - **Quote candidate**: "The scale token contributes +0.0004 combined macro_f1 (within noise) but +0.175 Bot AUROC under the joint head_lr ×5 + multi-scale training regime; its load-bearing role is rare-class signal stabilization in a specific hp × training-stream regime, not a primary-metric lever."
> - **Risk if missed**: Paper §3.4 scale token 仍以 "multi-scale conditioning innovation" framing,reviewer ablation 不动 challenges(dim 4 Cell B factorial 已 isolate)。
> - **Cross-link**: M5-005 v3 (三维耦合) + M5-014 (Pareto navigator framing source) + M2-008 (Bot n=12 statistical edge — single-correct flip 局限)
> - **Priority**: HIGH

### Finding M5-010: macro_f1 trajectory dip 数 + magnitude 与 architecture × head_lr 4×2 matrix 强相关(head_lr ×5 cells dip 3-5× ×1 cells)[HIGH]

- **Context**: M5.5 retrofits + round 1 forward cells combined macro_f1 trajectory 普遍 oscillation;dip pattern 与 architecture / hp 关系待 isolate。
- **Discovery**: 12 cells dip 数 + magnitude 与 architecture × head_lr 4×2 matrix 强相关;head_lr ×5 cells 平均 dip 数是 ×1 cells 3-5×。
- **Evidence**: 12 cells per_epoch.json 系列(M5.5 R2 retrofits + dim 1+2+4 forward);具体 dip count 表 deferred to design-layer aggregation。
- **Status**: open finding, qualitative pattern across 12 cells;quantitative dip-count table deferred to design layer;paper-value annotation deferred to design layer

> 🎯 **论文价值标注**
> - **Section**: Discussion §6.3 Training dynamics + per-cell trajectory analysis
> - **Use**: Combined macro_f1 trajectory dip 数 + magnitude 与 architecture × head_lr 4×2 matrix 强相关 — head_lr ×5 cells 平均 dip 是 head_lr ×1 cells 3-5×。Paper §6.3 narrative anchor:training dynamics 不平稳,具体 dip shape 是 hp × architecture 交互的 measurable signature 而非 noise floor。
> - **Risk if missed**: Paper §6 仅 report end-epoch number 错过 trajectory shape information;reviewer 看 dip count 多 cell 但不知其原因。
> - **Cross-link**: M5-005 v3 (head_lr × init × stream 三维耦合,dip pattern 是其 derived signature) + M5-007 (6-baseline cross-arch dip count comparison) + M5-009 (final-epoch jump pattern sibling)
> - **Priority**: HIGH

### Finding M5-012: TimeSformer R1 唯一 sustained Bot AUROC > 0.7 + DDoS smooth gradient — property tightly bound to random + head_lr ×1 regime [HIGH]

- **Context**: M5.5 R1 vs R1.5 forensic + M5.5 R2 baselines + dim 1 SSv2 + dim 4 Cell B 后 cross-cell 对照完整。
- **Discovery**: TimeSformer R1(random init + head_lr ×1 effective)唯一 sustained Bot AUROC > 0.7(0.7151)+ 唯一 DDoS smooth gradient(无 final-epoch jump)。Property tightly bound to random + ×1 regime;R1.5 intentional ×5 落到 0.4616 / Bot 0.5940。
- **Evidence**: M5.5 R1 `outputs/run_20260502_232207/` + R1.5 `outputs/run_20260503_121046/` + 其余 baseline + dim 1 SSv2(commit `92d2055`)+ dim 4 Cell B(commit `5bc6b32`)对照。
- **Status**: open finding, cross-cell evidence anchored;refines M5-005 v3 splitting-variable mechanism;paper-value annotation deferred to design layer

> 🎯 **论文价值标注**
> - **Section**: Discussion §6.4 TimeSformer R1 双重 unique architectural anchor
> - **Use**: TimeSformer R1(random init + head_lr ×1 effective)唯一 sustained Bot AUROC > 0.7(0.7151)+ 唯一 DDoS smooth gradient(无 final-epoch jump,M5-009 例外)。Paper §6.4 narrative anchor:某些 cross-architecture × hp regime combinations 产生 unique trajectory properties,但 regime-tied attribution(R1.5 intentional ×5 ablation supplementary 0.4616 + Bot 0.5940 验证 attribute)。
> - **Risk if missed**: Reviewer 用 R1 0.4836 / 0.7151 数据 attack "为什么 R1 跨 5 baselines unique";本 finding R1 vs R1.5 forensic 解释 + attribute mechanism。
> - **Cross-link**: M5-005 v3 (三维耦合) + M5-006 (matcher bug 产生 R1 vs R1.5 mechanism) + M6-012 (M6.1 1D byte Bot AUROC 0.7402 进一步 cross-paradigm 推广)
> - **Priority**: HIGH

### Finding M5-009: 18 cells DDoS F1 final-epoch sharp jump universal(5/6 M5.5 baselines + 7 dim 1+2+4 forward cells);M6.1 1D byte cell macro_f1 final-epoch jump observed but DDoS-specific not confirmed [MEDIUM]

- **Context**: M5.5 baselines + round 1 forward cells DDoS F1 trajectory 末端行为待 isolate。
- **Discovery**: 5/6 M5.5 baselines + dim 1+2+4 cells DDoS F1 在 final epoch 出现 sharp jump;唯独 TimeSformer R1(M5-012)smooth gradient。NID + dominant-rule labeling + cosine decay 末端 universal 现象。M6.1 1D byte cell(N+2 closeout 测)observed macro_f1 final-epoch jump +0.097(ep8 0.1471 → ep9 0.2444),但 DDoS F1 specific final-epoch Δ = +0.0744(ep8 = 0.0000 → ep9 = 0.0744)未达 sharp-jump 阈值 +0.10。M5-009 仍 18 cells 严格 anchor;M6.1 macro_f1 final-jump 是 macro-aggregated artifact 而非 DDoS-specific pattern。Paradigm-generality of M5-009 在 M6.1 数据上 inconclusive。
- **Evidence**: 18 cells per_epoch.json DDoS trajectory(M5.5 R2 6 baselines retrofits + dim 1+2+4 7 forward + 5 supplementary)+ `outputs/run_20260516_090240/metrics/per_epoch.json`(M6.1 DDoS F1 epoch 0-9 trajectory: 0 / 0 / 0 / 0 / 0 / 0 / 0 / 0 / 0 / 0.0744)。
- **Status**: open finding, universal pattern across 18 cells (1 exception: TimeSformer R1);paper-value annotation deferred to design layer(本 P1 update)

> 🎯 **论文价值标注**
> - **Section**: Discussion §6.3 Training dynamics universal patterns
> - **Use**: NID + dominant-rule labeling + cosine decay 末端 universal final-epoch DDoS F1 jump pattern,18 cells confirm(M5.5 R2 + dim 1+2+4 + retrofits)。M6.1 1D byte 是 macro-aggregated jump but DDoS-specific 不 strict confirm(Δ +0.0744 < +0.10),M5-009 仍 18-cell anchor。Paper §6.3 用作 "training dynamics 跨 architecture × pretraining × paradigm 表现差异 vs 一致性" nuanced 数据点 — universal at video paradigm,不 generalize strict to 1D byte paradigm。
> - **Risk if missed**: Reviewer 看 paper §6 final-epoch numbers,问 "DDoS final-epoch jump 是 universal 还是 cell-specific";本 finding 提供 18-cell 实证 anchor + 跨 paradigm scope honest disclosure。
> - **Cross-link**: M5-011 (GoldenEye oscillation universal sibling 18-19 cells) + M6-009 (M6.1 macro-jump observed but DDoS-specific not confirm — paradigm scope refinement)
> - **Priority**: MEDIUM

### Finding M5-011: 18 cells GoldenEye F1 epoch-to-epoch oscillation universal(magnitude 0.1-0.4,α reweight 抬高 mean 但不消除)[MEDIUM]

- **Context**: M5.5 + round 1 cells GoldenEye F1 oscillation 是否 universal 待确认。
- **Discovery**: 18 cells(11 retrofit + 7 forward)GoldenEye F1 epoch-to-epoch oscillation 全部 present,magnitude 0.1-0.4。Universal across architectures + losses;α reweight 抬高 mean 但不消除 oscillation。
- **Evidence**: 18 cells per_epoch.json GoldenEye trajectory。
- **Status**: open finding, universal pattern across all 18 cells with no exception;paper-value annotation deferred to design layer

> 🎯 **论文价值标注**
> - **Section**: Discussion §6.3 Noisy-attractor universal pattern
> - **Use**: GoldenEye F1 epoch-to-epoch oscillation 跨 18-19 cells universal,magnitude [0.06, 0.44],α reweight 不消除(P2 update M5-004 19-cell sibling)。Paper §6.3 用作 "small-support softmax + focal+α 联合 produce noisy-attractor trajectory 是 representation × loss 联合的 stable signature"实证;non-removable architectural / hp lever。
> - **Risk if missed**: Reviewer 看 GoldenEye F1 oscillation 问 "cell-specific noise 还是 universal";本 finding 19-cell 实证 close 这个 attack vector。
> - **Cross-link**: M5-004 (focal loss prevent vs correct + α saturation) + M5-009 (DDoS final-jump sibling universal) + M6-011 (M6.1 GoldenEye AUROC decay directional variant — single-cell anomaly within universal oscillation pattern)
> - **Priority**: MEDIUM

### Finding M5-015: K400 prior 在 slow stream 上 lift 是 fast stream 3.6×(Δ slow +0.098 vs Δ fast +0.027)[MEDIUM]

- **Context**: dim 1 K400 transfer 在 combined / fast / slow 上 lift 量级是否一致 待确认。
- **Discovery**: K400 vs random Δ slow macro_f1 +0.098 vs Δ fast +0.027;K400 prior 在 slow stream(Δt=1s)上 lift 是 fast stream 3.6×。
- **Evidence**: dim 1 三 cell 主表 fast / slow / combined(`outputs/run_20260502_184512/m5_4_phase2_eval/eval_metrics.json` + `outputs/run_20260507_205921/m5_10_random_videomae_eval/eval_metrics.json`)。
- **Status**: open finding, stream-decomposed Δ anchored from dim 1 主表;paper-value annotation deferred to design layer

> 🎯 **论文价值标注**
> - **Section**: Discussion §6.2 K400 stream-decomposed transfer
> - **Use**: dim 1 cross-cell K400 prior Δ slow macro_f1 +0.098 远超 Δ fast +0.027,K400 prior 在 slow stream(1s 窗口)上 lift 是 fast stream 的 3.6×。Stream-decomposed evidence supports "K400 prior 对 longer-time-scale temporal patterns 信号 stronger" 推测。
> - **Quote candidate**: "K400 pretraining lifts slow-stream macro_f1 by 0.098 over random initialisation while lifting fast-stream by only 0.027 — a 3.6× asymmetry. K400's 1-second temporal sensitivity transfers preferentially to the slow stream's 1-second window, with fast stream's 100ms granularity less aligned with K400's pretraining regime."
> - **Risk if missed**: Paper §6.2 K400 prior framing 缺 stream-decomposed evidence,reviewer 问 "K400 prior helps slow / fast / uniform?"。
> - **Cross-link**: M5-008 (K400 5-epoch head start mechanism) + M5-013 (SSv2 反例 corpus-specific) + M5-007 sub(c) (ConvLSTM slow-stream inverse — paradigm-architecture × stream interaction)
> - **Priority**: MEDIUM

### Finding M5-017: dim 4 single-stream OOD asymmetry — Framing A(within-cell)slow→fast heavier vs Framing B(cross-cell vs A)fast→slow heavier;两 framing 方向相反 [MEDIUM]

- **Context**: dim 4 Cell C/D 单 stream 训练 + 双 stream eval 跨 stream OOD 现象。
- **Discovery**: Framing A(within-cell)slow→fast(Cell D)Δ −0.395 heavier than fast→slow(Cell C)Δ −0.171;Framing B(cross-cell vs A)fast→slow(C slow vs A slow)Δ −0.317 heavier than slow→fast(D fast vs A fast)Δ −0.284。两 framing 测不同物理量,方向相反,都 record。
- **Caveat**: Cell C/D undertraining confound(Cell D 仅 2,420 grad_steps vs A/B 48,530)— `docs/m5_10_scale_token_ablation.md` §"Training budget caveat" 段已 explicit。
- **Evidence**: dim 4 commits `5bc6b32` + `3d830af` + `1d8a2bc` + `424d066`(framing 修订);`docs/m5_10_scale_token_ablation.md` §"OOD asymmetry: two framings" + 主表 stream-decomposed 数字。
- **Status**: open finding, dual-framing macro_f1 Δ anchored;paper-value annotation deferred to design layer

> 🎯 **论文价值标注**
> - **Section**: Discussion §6.3 single-stream OOD asymmetry + Limitations §7
> - **Use**: dim 4 Cell C/D 单 stream 训练 + 双 stream eval cross-stream OOD evaluation。Framing A(within-cell)slow→fast(D)Δ −0.395 heavier than fast→slow(C)Δ −0.171;Framing B(cross-cell)fast→slow(C vs A)Δ −0.317 heavier than slow→fast(D vs A)Δ −0.284。两 framing measure 不同物理量(within-cell stream consistency vs cross-cell stream-loss decomposition),都 record。Cell C/D undertraining confound(Cell D 2,420 grad_steps vs A/B 48,530)论文 §7 显式 acknowledge。
> - **Risk if missed**: Reviewer 问 single-stream training implication on cross-stream eval;本 finding 提供 dual-framing 实证 + undertraining caveat。
> - **Cross-link**: M5-005 v3 (stream-count factor) + M5-014 (Pareto navigator) + M5-015 (K400 stream-decomposed transfer)
> - **Priority**: MEDIUM

### Finding M5-018: 跨 session 8 instances CC stop-and-report 主动 catch 设计层错误 — collaborative error correction methodology infrastructure [LOW]

- **Context**: Round 1 + M6 + N+1 期间设计层多次主动 catch CC 错误;CC stop-and-report 是 project methodology infrastructure 而非个例。
- **Discovery**: 8 instances 跨 session 累积(7 strict stop-and-report + 1 procedural transparent post-hoc surfacing):
  1. Round 1 dim 2 adapter regime misclass(strict)
  2. Round 1 dim 2 GPU threshold 错推(strict)
  3. Round 1 dim 4 adapter norm doc citation 误读(strict)
  4. Round 1 dim 4 OOD framing direction 错算(strict)
  5. N+1 M6.1 Phase 0 报告缺失(strict)— 设计层 over-trust 了 handoff §7 "spec 已在 Phase 0 报告内" 字面表达,未 verify disk state;CC stop-and-report 触发 (a)/(b)/(c) trade-off review,选 (a) real Phase 0
  6. N+1 M6.1 Phase 0 peak GPU > 4 GB(strict)— pre-norm Transformer 关掉 PyTorch nested tensor fast-path → slow-path materialize 4.3 GB fp16 attention matrix → CC propose SDPA custom encoder layer(Option 1),设计层 approve
  7. N+1 M6.1 Phase 1 epoch-0 sanity trigger(strict)— loss < 0.1 within 11 steps + val macro_f1 0.0971 < 0.10 → CC stop-and-report,设计层 review 后 continue full 10 epochs(Option 2)
  8. N+1 M6.1 Phase 0 docs/m6_1_byte_transformer.md force-add(procedural transparent post-hoc)— CC 用 `git add -f` 强制 add gitignored docs/ file mirror m5_10/m5_5 precedent,but 严格说违反铁律 §3 "不 force-add 除非 prompt 明确",CC 在 commit report ambiguities 段 transparent 报告 + ask confirmation;设计层 accept outcome but note 未来类似不可逆 git op 必须 stop-and-ask before doing。
- **Evidence**: Round 1 + N+1 conversation history(no commit anchor for process-level findings);N+1 期间 stop-and-report instances 在 `592e9f0`(Phase 0)+ `aae6df3`(Phase 1)commit 周边 report 通信内可 trace。
- **Status**: open finding, process-level methodology record;8 instances anchor;paper-value annotation deferred to design layer(本 P1 update)

> 🎯 **论文价值标注**
> - **Section**: Methodology / Acknowledgments (可选 mention)
> - **Use**: CC stop-and-report collaborative error correction methodology infrastructure 跨 round 1 + N+1 sessions 累积 8 instances(7 strict + 1 procedural transparent post-hoc)。Paper §6.1 Silent failure detection chain narrative process-level companion — 不仅 silent failure 在 code / data 层 detect,设计层 prompt error 也 by execution layer detect。可选 acknowledgments section mention 或 §6 methodology infrastructure subsection。
> - **Risk if missed**: Paper §6 methodology rigor narrative 缺 process-level evidence。
> - **Cross-link**: TRANSITION-005 / M4-001 / M4-002 / M4-010a / M5-003 / M5-006 (silent failure detection chain code-data anchor) + 本 finding (process-layer companion)
> - **Priority**: LOW

---

# M6 — 实验矩阵阶段

### Finding M6-001: M6.2 max-confidence aggregation BENIGN-biased pathology — combined macro_f1 0.12-0.18, 多数 attack F1 = 0, paradigm × aggregation-rule artifact 非 model bug [HIGHEST]

- **Context**: M6.2 RF + XGB 在 per-window aggregation 下 combined macro_f1 远低于 video 最低 cell,机制待 isolate。
- **Discovery**: M6.2 RF combined 0.1793 / XGB 0.1165;BENIGN F1 0.96,多数 attack F1 = 0;DDoS / DoS Hulk 例外。机制:per-flow RF/XGB 在 BENIGN 上 max(predict_proba)高于 minority class;混合 window 内 BENIGN flow argmax 赢 max-conf aggregation。Paradigm + aggregation rule artifact,非 model bug。文献 per-flow metric 0.90+ 是 eval space artifact(详 M6-003)。
- **Evidence**: M6.2 commit `15df74e` + `outputs/m6_rf/eval_metrics.json` + `outputs/m6_xgb/eval_metrics.json` + `outputs/m6_rf/per_class_table.csv`(9 attack class F1 = 0)。
- **Status**: open finding, mechanism-isolated;attack-priority aggregation rule alternative 留作 round-2 / closeout consideration;paper-value annotation deferred to design layer

> 🎯 **论文价值标注**
> - **Section**: Discussion §6.5 paradigm × aggregation-rule pathology
> - **Use**: M6.2 RF/XGB max-confidence aggregation BENIGN-bias pathology — combined macro_f1 0.1165-0.1793,多数 attack F1=0,paradigm × aggregation-rule artifact 非 model bug。Paper §5 cross-paradigm 12-row 表 flow-feature row 数字解释 anchor + paper §6 paradigm decision-rule pathology 双 case study(与 M6-009 1D byte argmax-collapse 形成 paradigm-decision-rule double anchor)。文献 per-flow metrics 0.90+ 是 eval space artifact(M6-003 detail)。
> - **Quote candidate**: "M6.2 flow-feature baselines (RF / XGB) achieve combined macro_f1 0.1165–0.1793 under per-window max-confidence aggregation, with BENIGN F1 = 0.96 dominating and most attack classes F1 = 0 — a paradigm × aggregation-rule artifact rather than a model implementation bug. Published per-flow flow-feature accuracy figures (≥ 0.90) reflect a different evaluation space; cross-paradigm comparison under window-level aggregation produces this BENIGN-biased pathology."
> - **Risk if missed**: Paper §5 flow row 数字看起来 incompetent;reviewer 问 "为什么 RF/XGB 这么低,文献 0.90+"。本 finding 提供 mechanism + eval-space difference explanation。
> - **Cross-link**: M6-002 (Heartbleed flow-paradigm specific vulnerability) + M6-003 (per-flow vs per-window eval space difference) + M6-009 (1D byte argmax-collapse parallel paradigm pathology)
> - **Priority**: HIGHEST

### Finding M6-002: M6.2 Heartbleed F1 = 0 — per-flow paradigm 对 long-lived rare-class 本征 vulnerability(11 flow rows vs 248 val window samples)[HIGH]

- **Context**: M6.2 Heartbleed F1 = 0 现象 — 长时序攻击在 per-flow paradigm 下的表现。
- **Discovery**: Heartbleed CSV 11 flow rows vs 248 val window samples;M6.2 RF Heartbleed F1 = 0,XGB AUROC 0.38 反向。Per-flow paradigm 对 long-lived rare-class 本征 vulnerability(per-flow 视角下 1 个 long-lived flow 在 per-window aggregation 后 spread 到 ~22 windows,但 model 只见 1 条 train row → 严重 undertraining)。
- **Evidence**: M6.2 `outputs/m6_rf/per_class_table.csv` + `outputs/m6_xgb/per_class_table.csv` Heartbleed row;CSV row count from `data/raw/cicids2017/TrafficLabelling/Wednesday-workingHours.pcap_ISCX.csv`。
- **Status**: open finding, structural per-flow paradigm limitation;paper-value annotation deferred to design layer

> 🎯 **论文价值标注**
> - **Section**: Discussion §6.5 per-flow paradigm long-lived rare-class vulnerability
> - **Use**: Heartbleed CSV 11 flow rows vs 248 val window samples — per-flow paradigm 对 long-lived rare-class 本征 vulnerability(11/248 = 4.4% per-flow representation,远低于 per-window 1.4%)。M6.2 RF Heartbleed F1 = 0;XGB AUROC 0.38 anti-correlated。Paper §6.5 paradigm per-class capacity scope:per-flow representation 把 long-duration attacks 压缩进 single feature vector,丢失 within-flow temporal evolution 信息。
> - **Risk if missed**: Paper §6 paradigm per-class signature 仅 isolate 整体 ordering 不 surface 具体 mechanism。
> - **Cross-link**: M6-001 (aggregation rule artifact sibling) + M6-007 (Heartbleed paradigm-specific signal-capture cross-paradigm)
> - **Priority**: HIGH

### Finding M6-003: per-flow IDS 0.90+ macro_f1 vs 本工作 per-window 0.18 数字差异 = eval-space artifact 非 model 性能突变 — SOC deployment workflow framing [HIGH]

- **Context**: 已发表 per-flow IDS 数字 0.90+ macro_f1 vs 本工作 per-window 0.18 数字差异需要 framing。
- **Discovery**: 数字差异不是 RF/XGB 性能突变,是 eval space 改变。Per-window = NID 实际 deployment workflow(分析师按 1.6s 时段消费 alert);per-flow = data-view artifact 不映射 SOC 任务。
- **Evidence**: M6.2 commit `15df74e` 主表(combined 0.18 / RF + 0.12 / XGB)+ 文献 cite(N+5 paper drafting 时填)。
- **Status**: open finding, framing argument anchored;literature cite list 留 paper-drafting 阶段填;paper-value annotation deferred to design layer

> 🎯 **论文价值标注**
> - **Section**: Methods §4 Evaluation policy + Discussion §6.5
> - **Use**: per-flow eval(literature standard for flow-feature paradigm)vs per-window aggregation(本工作)产生不同 paradigm rankings。M6.2 per-flow macro_f1 文献风格 0.85+ vs per-window 0.12-0.18。Paper §4 eval policy choice + §6.5 acknowledge eval-space discontinuity。Reproducibility 角度:per-window aggregation 与 video / 2D / 1D paradigm 一致,fair contract 跨 paradigm。Per-flow 仅适合 flow-feature paradigm self-eval。
> - **Risk if missed**: Paper §4 eval policy 选择不解释;reviewer 用文献 0.90+ "你测错了" attack。
> - **Cross-link**: M6-001 (BENIGN-bias artifact under per-window) + M6-002 (Heartbleed flow vulnerability) + M5-003 (no_cycle eval policy fairness contract sibling)
> - **Priority**: MEDIUM

### Finding M6-004: 11-cell paradigm 排名 strict — video > 2D snapshot > flow(combined macro_f1)— paper §1 立论关键 empirical anchor [HIGHEST]

- **Context**: 跨 paradigm combined macro_f1 ordering 是 paper §1 立论关键 empirical anchor。
- **Discovery**: 11-cell paradigm 排名 strict — video [0.2471, 0.4756] > 2D snapshot [0.2061, 0.2097] > flow [0.1165, 0.1793]。2D / flow within-paradigm spread tight;video spread wider 反映 architecture × pretrain。Idea.md §1.2 立论 "different paradigms produce systematically different per-class signatures" 直接 empirical support。
- **Evidence**: M6.3 commit `daa675b` + `outputs/m6.3_cross_paradigm_summary.json` 11-cell(7 video + 2 flow + 2 2D snapshot)。
- **Status**: open finding, 11-cell ordering anchored;paper Table 1 candidate;paper-value annotation deferred to design layer

> 🎯 **论文价值标注**
> - **Section**: Results §5.1 12-row cross-paradigm table(refined by M6-008)
> - **Use**: 跨 paradigm hierarchy video > 1D byte > 2D snapshot > flow,12-cell evidence(原 M6-004 11-cell + M6.1 加 1 row → M6-008 hierarchy update)。Paper §5 12-row table flagship finding。Note hierarchy 是 single-stream apples-to-apples(24,260 grad_steps fast-only)for paradigm-bound assessment + multi-stream full-budget(48,530 grad_steps)for paradigm-favorable assessment。
> - **Quote candidate**: "Under matched single-stream training budget (24,260 grad steps, head_lr ×1 contract), the cross-paradigm hierarchy is video [0.4341, 0.5197] > 1D byte 0.2444 > 2D snapshot [0.2061, 0.2097] > flow [0.1165, 0.1793]; the 0.19 gap between 1D byte and the video fast-only counterpart is substantial and paradigm-bound."
> - **Figure idea**: paper §5 12-row table + 2D scatter plot(paradigm vs combined macro_f1 + per-class colors)
> - **Risk if missed**: Paper §5 表的核心 hierarchy claim 缺 12-cell apples-to-apples 实证;reviewer 问 "fair comparison contract"。
> - **Cross-link**: M6-008 (12-cell update + apples-to-apples comparison) + M6-006 (Bot AUROC cross-paradigm refinement) + M6-007 (Heartbleed cross-paradigm refinement) + M5-007 (within-video cross-architecture)
> - **Priority**: HIGHEST

### Finding M6-005: ImageNet pretrain 对 NID 6-channel snapshot 零 measurable benefit(M6.3.RN 0.2097 vs IN 0.2061 Δ +0.0036 within noise)— pretrain transfer 是 corpus × modality × downstream 联合特征 [HIGH]

- **Context**: 2D snapshot paradigm 是否 ImageNet pretrain 与 video paradigm K400 pretrain 一样 transfer。
- **Discovery**: M6.3.RN combined 0.2097 vs M6.3.IN 0.2061 = Δ +0.0036(within noise)。ImageNet pretrain 对 NID 6-channel snapshot 零 measurable benefit。Conv1 norm 12.58(IN)vs 2.45(RN)反映 feature scale 但 downstream 训练 adapt 到 comparable per-class signature。与 M5-008(K400 head start)+ M5-013(SSv2 反例)形成 3 段对照:pretrain transfer 是 corpus × modality × downstream 联合特征,非 universal benefit。
- **Evidence**: M6.3 commit `daa675b` 两 sub-cell eval_metrics.json(`outputs/run_20260514_084832/m6_3_in_eval/eval_metrics.json` + `outputs/run_20260514_103615/m6_3_rn_eval/eval_metrics.json`)。
- **Status**: open finding, IN vs RN Δ anchored at within-noise magnitude;adds 3rd data point to pretrain-transfer cluster (M5-008 + M5-013 + this);paper-value annotation deferred to design layer

> 🎯 **论文价值标注**
> - **Section**: Discussion §6.2 pretrain transfer paradigm-by-paradigm consistency
> - **Use**: M6.3 IN ResNet-18(ImageNet pretrain)vs RN ResNet-18(random)combined Δ −0.0036 within noise(IN 0.2061 vs RN 0.2097)。**Pretrain transfer 在 2D snapshot paradigm 上 effectively zero**,与 video paradigm(K400 transfer +0.037 substantial / SSv2 transfer +0.003 within noise)形成对照 — pretrain transfer 是 corpus × modality × downstream-task 联合特征,non-universal。M5-008(K400)+ M5-013(SSv2)+ M6-005(IN ResNet)= 3 段 pretrain transfer 跨 paradigm 实证 anchor。
> - **Risk if missed**: Paper §6.2 pretrain transfer narrative 仅 video paradigm 内 evidence;reviewer 问 "pretrain transfer 在 2D / 1D paradigm 上 generalize?"。
> - **Cross-link**: M5-008 (K400 head start) + M5-013 (SSv2 反例 corpus-specific) + M6-008 (12-cell hierarchy data anchor)
> - **Priority**: MEDIUM

### Finding M6-006: M5-005 v3 三维耦合 Bot signal preservation 仅在 video paradigm 内有效 — Bot rare-class 检测需要 temporal axis [HIGH]

- **Context**: M5-005 v3 三维耦合 finding 跨 paradigm 适用性待验证。
- **Discovery**: Bot AUROC 跨 paradigm — video dim1-random 0.6743 / TimeSformer-R1 0.7151 vs 2D snapshot M6.3.IN 0.348 / M6.3.RN 0.283 vs flow M6.2 RF 0.50 / M6.2 XGB 0.47。M5-005 v3 三维耦合(head_lr × init × multi-scale)Bot signal preservation 仅在 video paradigm 内有效;Bot rare-class 检测需要 temporal axis 非 head_lr 调度可 compensate。论文 narrative 必须 explicit 限定 finding scope。
- **Evidence**: 4 cells M6.3 + M6.2 + 6 video cells Bot AUROC table 汇总;`outputs/m6.3_cross_paradigm_summary.json` per_class_combined_auroc.Bot 字段。
- **Status**: open finding, 10-cell Bot AUROC cross-paradigm comparison anchored;refines M5-005 v3 scope from "across regimes" to "within video paradigm";paper-value annotation deferred to design layer

> 🎯 **论文价值标注**
> - **Section**: Discussion §6.4 Bot rare-class paradigm scope(refined by M6-012)
> - **Use**: Original 11-cell finding "Bot AUROC video-paradigm-exclusive preservation"。N+2 P1 M6-012 partial overturn(M6.1 1D byte AUROC 0.7402 > video top 0.7151)→ refined scope "Bot AUROC preservation 在 video + 1D byte paradigm 都 viable,2D snap + flow 失败;argmax F1 paradigm-uniform 0 due to n=12 statistical edge"。**Temporal axis 不是 strict prerequisite for ranking signal preservation**。Paper §6.4 narrative anchor with M6-012 refinement note。
> - **Quote candidate**: "Bot AUROC preservation (≥ 0.67) is achieved by random-init video backbones, TimeSformer-Small under head_lr ×1 contract, and the 1D byte-level Transformer (0.7402); 2D snapshot and flow paradigms collapse Bot ranking; per-class F1 = 0 across all paradigms reflects the n = 12 statistical edge, not paradigm capacity."
> - **Risk if missed**: Paper §6 overstates Bot 需要 temporal axis;reviewer 用 M6.1 数据 attack。本 finding 主动 refinement scope。
> - **Cross-link**: M5-005 v3 (三维耦合 video-internal) + M6-012 (1D byte partial cross-paradigm overturn) + M2-008 (Bot n=12 statistical edge)
> - **Priority**: HIGH

### Finding M6-007: Heartbleed F1 paradigm-specific signal-capture case study(video ≥ 0.97 vs 2D snapshot 0.13-0.26 vs flow 0.00)[MEDIUM]

- **Context**: Heartbleed 跨 paradigm 表现差异是否反映 signal-capture mechanism。
- **Discovery**: Heartbleed F1 — video ≥ 0.97+ / 2D snapshot 0.26(IN)/ 0.13(RN)/ flow 0.00(both)。Heartbleed signal = static spatial pattern + temporal continuity 联合;2D 抓 static 丢 temporal;flow 抓 per-flow detail 丢 per-window aggregation。Paradigm signal-capture mechanism case study。
- **Evidence**: 11-cell per_class table(`outputs/m6.3_cross_paradigm_summary.json` per_class_combined_f1.Heartbleed across cells)。
- **Status**: open finding, single-class paradigm-decomposition case study;paper-value annotation deferred to design layer

> 🎯 **论文价值标注**
> - **Section**: Discussion §6.4 paradigm-specific signal-capture case study
> - **Use**: Heartbleed F1 paradigm-specific:video ≥ 0.97 / 2D snap [0.13, 0.26](IN/RN)/ flow 0.00(RF/XGB)/ 1D byte 0.2367(M6.1 Phase 1)。Heartbleed signal = static spatial pattern + temporal continuity 联合;video tube capture 两者;2D snap 抓 static 丢 temporal;flow per-flow detail 丢 per-window aggregation;1D byte 抓 byte content 但失 cross-packet temporal binding。Paper §6.4 case study + M6-010(slowloris 1D-favored single-class)形成 paradigm-specific signal-capture 双 case anchor。
> - **Risk if missed**: Paper §5 12-row 表 average ranking 错过 per-class paradigm signature heterogeneity。
> - **Cross-link**: M6-002 (Heartbleed flow vulnerability mechanism) + M6-008 (12-cell hierarchy) + M6-010 (slowloris 1D-favored sibling case)
> - **Priority**: MEDIUM

### Finding M6-008: Cross-paradigm 12-cell hierarchy update — 1D byte joins between video and 2D snapshot [HIGH]

- **Context**: M6.1 1D byte Transformer Phase 1 完成,跨范式 hierarchy 11 → 12 cells。
- **Discovery**: M6.1 fast-only combined macro_f1 = 0.2444 (val_n_fast = 16,463, 24,260 grad steps);apples-to-apples vs dim 4 Cell C(video fast-only same 24,260 grad steps + head_lr ×1):Δ = −0.1897。12-cell hierarchy(排除 dim 4 Cell D under-training outlier):video [0.4341, 0.5197] > **1D byte 0.2444** > 2D snap [0.2061, 0.2097] > flow [0.1165, 0.1793]。1D byte 落在 video 与 2D snapshot 之间;substantially below video at matched single-stream budget。
- **Evidence**: M6.1 commit `aae6df3` + `outputs/run_20260516_090240/m6_1_byte_transformer_eval/eval_metrics.json`(canonical fast-slice macro_f1 = 0.2444 / val_n_fast = 16,463;combined / fast / slow 三栏数值一致 due to baseline_rerun.py fast+slow 强制 → fast shards 双 pass artefact,详 `docs/m6_1_byte_transformer.md §Reproduction`);`outputs/m6.3_cross_paradigm_summary.json`(本 P1 prompt §D 更新含 M6.1 row)。
- **Status**: open finding, 12-cell hierarchy + apples-to-apples 同 budget 比较 anchored;paper §5 12-row table anchor

> 🎯 **论文价值标注**
> - **Section**: Results §5.1 Cross-paradigm 12-row table + Discussion §6 paradigm comparison
> - **Use**: 4-paradigm 矩阵 final data point。Idea.md §2.1 矩阵 close + §6.5 contribution #5 ("4 paradigm × ~12 row 跨范式 per-class signature")数据 anchor。1D byte 落在 video 与 2D 之间是 paradigm hierarchy 实证而非 hand-waving。
> - **Quote candidate**: "The 1D byte-level Transformer achieves combined macro_f1 = 0.2444 on the fast slice (val n = 16,463) at 24,260 grad steps; under matched single-stream budget and head-lr-multiplier ×1 contract, it falls 0.19 absolute below the video fast-only counterpart (dim 4 Cell C, 0.4341) and 0.03-0.05 absolute above the 2D snapshot paradigm."
> - **Risk if missed**: Paper §5 数据 lacuna;4-paradigm narrative 与 results 表不一致。
> - **Cross-link**: M6-001 / M6-004 / M6-006(M6-006 partial overturn 见 M6-012)。本 finding 是全 12 cell hierarchy 数据基础。
> - **Priority**: HIGH

### Finding M6-009: M6.1 epoch-0 argmax-2-class collapse → epoch-9 7-class recovery — paradigm × training dynamics (parallels M6-001 but training-resolvable) [HIGH]

- **Context**: M6.1 epoch 0 sanity trigger fired(loss < 0.1 within 11 grad steps + val macro_f1 0.0971 < 0.10 threshold);设计层 review 后 continue full 10 epochs。
- **Discovery**: Epoch 0 argmax 只 distinguish 2 类(BENIGN 14,646 = 89% + Heartbleed 1,817 = 11%),11/13 类 F1 = 0;但 AUROC 显示模型已学 ranking 信号(DDoS 0.7062 / Hulk 0.8103 / GoldenEye 0.8167 / slowloris 0.8823)。Macro_f1 trajectory:plateau ep0-2 = 0.0971 → step-up ep3 = 0.1368 → plateau ep3-6 → step-up ep7 = 0.1471 → plateau ep7-8 → final-epoch jump ep9 = 0.2444(+0.097)。Epoch-9 argmax 7-class distinguishable(BENIGN 15,054 + Heartbleed 527 + GoldenEye 302 + Slowhttptest 194 + Hulk 190 + slowloris 187 + DDoS 9);4 attacks 仍 F1=0(PortScan / FTP-Patator / SSH-Patator / Bot,3/4 AUROC < 0.75 → ranking signal 弱 not just argmax collapse)。形态与 M6-001(M6.2 BENIGN-bias collapse)相似但机制不同:M6.2 是 per-flow max-confidence aggregation rule artifact;M6.1 是 training dynamics 早期 decision-threshold collapse,但 sufficient training(10 epochs / 24,260 grad steps)partial escape。Sub-bullet:eval bundle val_n_total = 32,926 是 baseline_rerun.py dual-stream-contract artefact(fast shards 双 pass);canonical M6.1 cross-paradigm metric = fast-slice macro_f1 0.2444 at val_n_fast = 16,463;详 `docs/m6_1_byte_transformer.md §Reproduction`。
- **Evidence**: M6.1 commits `592e9f0`(Phase 0)+ `aae6df3`(Phase 1)+ `outputs/run_20260515_232145/ckpt/best.pt`(forensic epoch-0 ckpt,保留 evidence)+ `outputs/run_20260516_090240/metrics/per_epoch.json` + `outputs/run_20260516_090240/metrics/confusion_per_epoch.npz`(epoch_0 + epoch_9 keys per-class breakdown)。
- **Status**: open finding, two-paradigm decision-rule pathology with M6-001;paper §6 paradigm dynamics narrative anchor

> 🎯 **论文价值标注**
> - **Section**: Discussion §6 paradigm × training dynamics + paradigm decision-rule pathology subsection
> - **Use**: 与 M6-001(M6.2 BENIGN-bias aggregation artifact)形成 paradigm-decision-rule pathology 双 case study。M6-001 = paradigm × aggregation-rule artifact(non-training-resolvable);M6-009 = paradigm × training-dynamics artifact(partial training-resolvable)。Reviewer 看 M6.1 epoch-9 macro_f1 0.24 时,trajectory + 7-class final argmax + AUROC structure preservation 是 "1D byte paradigm 在 sufficient training 下 partial recover,但仍 4 attacks F1=0" 的实证。Paper §6 框架:paradigm × decision rule × training budget 三轴 interaction。
> - **Risk if missed**: Reviewer 可能认为 "M6.1 macro_f1 0.24 是 implementation 没调好",忽略 trajectory pattern + AUROC structure。本 finding 把 0.24 framing 为 "paradigm partial recovery from cold-start decision-rule pathology" 而非 implementation failure。
> - **Cross-link**: M6-001(M6.2 BENIGN-bias aggregation artifact)+ M6-008(12-cell hierarchy 数据)+ M5-009(universal final-epoch jump pending DDoS verification — see updated M5-009)。
> - **Priority**: HIGH

### Finding M6-010: DoS slowloris F1 0.82 / AUROC 0.94 in 1D byte paradigm — single attack class effectively captured by byte-level Transformer [MEDIUM]

- **Context**: M6.1 per-class 表 slowloris 表现 isolate paradigm strength。
- **Discovery**: slowloris epoch-9 F1 = 0.82 / AUROC = 0.9393(epoch-0 F1 = 0.00 / AUROC = 0.8823 已 indicate paradigm signal preserved at random init)。是 M6.1 唯一 F1 > 0.5 attack class(其他 attacks F1 范围 [0, 0.31])。slowloris 攻击 signature = 慢 HTTP request 长 keep-alive partial header,byte-level content 高度 distinctive(repeated incomplete HTTP request lines + low TCP throughput pattern 可见于 first 128 bytes of L2 frame)。Cross-paradigm slowloris 比较(从 outputs/m6.3_cross_paradigm_summary.json 现有 video / 2D / flow 数据 + 本 P1 §D 加 M6.1 row 后):**M6.1 1D = 0.82 是跨 12 cells slowloris F1 高值之一**(具体 ranking 待 paper §5 grand table render)。Single attack class as paradigm strength case study;non-trivial paradigm-specific signal-capture demonstration。
- **Evidence**: `outputs/run_20260516_090240/m6_1_byte_transformer_eval/eval_metrics.json`(per_class.DoS slowloris F1 + AUROC)+ `outputs/run_20260516_090240/m6_1_byte_transformer_eval/per_class_table.csv`。
- **Status**: open finding, single-class paradigm-strength case study;paper §6 paradigm signal-capture mechanism narrative anchor

> 🎯 **论文价值标注**
> - **Section**: Discussion §6 paradigm signal-capture mechanism
> - **Use**: 与 M6-007(Heartbleed video ≥ 0.97 / 2D 0.13-0.26 / flow 0.00)形成 paradigm-specific signal-capture 双 case study。M6-007 = video-favored class;M6-010 = 1D-favored class。Paper §6 narrative:paradigm 不是 monolithic ranking,而是 per-class signal-capture profile 异质。
> - **Quote candidate**: "DoS slowloris achieves F1 = 0.82 / AUROC = 0.94 in the 1D byte-level Transformer cell (val n = 239); the attack's signature — repeated incomplete HTTP request headers with long keep-alive — fits the byte-level paradigm's first-128-byte L2 frame snapshot exceptionally well, while video and flow paradigms capture the same class through their respective representational strengths."
> - **Risk if missed**: Paper §5 12-row 表只 highlight macro_f1 ordering;reviewer 错觉 "1D 整体 inferior to video",忽略 per-class paradigm signal-capture profile heterogeneity。
> - **Cross-link**: M6-007(Heartbleed paradigm-specific)+ M6-008(12-cell hierarchy)+ M5-007(cross-arch within-video per-class trajectory)。
> - **Priority**: MEDIUM

### Finding M6-011: DoS GoldenEye AUROC decay 0.82 → 0.65 during M6.1 10-epoch training — single-class ranking-quality loss anomaly [MEDIUM]

- **Context**: M6.1 10-epoch AUROC trajectory 中是否任意 class 出现 ranking-quality decay。
- **Discovery**: DoS GoldenEye 是 M6.1 4 个 epoch-0-positive-signal attacks 中**唯一** AUROC decay case:epoch 0 = 0.8167 → epoch 9 = 0.6529(Δ −0.164)。其他 3 个 grew:Hulk 0.8103 → 0.8447(+0.034)/ slowloris 0.8823 → 0.9393(+0.057)/ DDoS 0.7062 → 0.7885(+0.082)。F1 同期 0.00 → 0.05 微 argmax 改进,但 ranking quality 实际 lost。Mechanism speculation(未 verify):early training 1D byte 学到 GoldenEye-like burst content signal(可能 HTTP request burst 与 Hulk 早期 representation 混淆),late training over-fit BENIGN-Heartbleed 分类 + slowloris representation 学习挤压 GoldenEye 表示空间。本条与 M5-011(GoldenEye F1 oscillation universal [0.09, 0.44] across 18+ cells)pattern 不同:M5-011 是 oscillation(无方向);M6-011 是单调 decay(有方向)。
- **Evidence**: `outputs/run_20260516_090240/metrics/per_epoch.json`(GoldenEye AUROC per epoch trajectory)+ `outputs/run_20260516_090240/metrics/confusion_per_epoch.npz`(epoch_0 vs epoch_9 GoldenEye row breakdown)。
- **Status**: open finding, single-class AUROC decay anomaly within M6.1 10-epoch training;mechanism speculative;paper Discussion supplementary detail

> 🎯 **论文价值标注**
> - **Section**: Discussion §6 paradigm × training dynamics + per-class trajectory supplementary
> - **Use**: paradigm × class × training-budget interaction 案例。论文 §6 narrative:training dynamics 在 1D byte paradigm 不是均匀 improving,per-class trajectory diverge — most grow but some decay。Reviewer 通常不见这一层;本 finding 强化 "10-epoch budget 在 1D paradigm 上未达 saturation,但 budget extension 是否 recover GoldenEye 不可 a priori 判断"。
> - **Risk if missed**: Paper §6 narrative 缺 per-class training dynamics 复杂性 layer。
> - **Cross-link**: M5-011(GoldenEye universal oscillation 18+ cells,本条是 directional decay variant)+ M6-009(epoch-0 collapse → 7-class recovery 主 trajectory)+ M6-008(12-cell hierarchy)。
> - **Priority**: MEDIUM

### Finding M6-012: M6.1 Bot AUROC 0.7402 partial overturns M6-006 — 1D byte paradigm also preserves Bot ranking signal but argmax F1 still 0 [HIGH]

- **Context**: M6-006 原 finding 锁 "Bot AUROC 跨 paradigm:video preserves(TimeSformer R1 0.7151,dim1-random 0.6743),2D snapshot collapses(0.348/0.283),flow weak(0.50/0.47);Bot rare-class 需要 temporal axis 非 head_lr 调度可 compensate"。
- **Discovery**: M6.1 1D byte Bot AUROC = 0.7402,**实际 > video top TimeSformer R1 0.7151**(Δ +0.0251)+ > video dim1-random 0.6743(Δ +0.0659)。但 Bot F1 仍 0.0000(Bot val n = 12 full split, fast slice n = 6, small-sample statistical edge per Idea.md §7.3)。1D byte paradigm **also preserves Bot ranking signal**,M6-006 原 finding "Bot rare-class 需要 temporal axis" 部分 overturned:不仅 video paradigm,1D byte paradigm 也 preserves;但 argmax F1 = 0 是跨 paradigm uniform pathology(无 paradigm 解决 n = 12 small-sample statistical edge)。**M6-006 scope refinement**:从 "Bot AUROC video-paradigm-exclusive preservation" → "Bot AUROC preservation 在 video paradigm + 1D byte paradigm 都 viable;2D snapshot + flow paradigm 失败;argmax F1 跨 paradigm uniform 0";temporal axis 不是 strict prerequisite for ranking signal preservation。M6-006 finding text 不重写(留作 historical anchor),scope refinement 通过本 M6-012 entry + Cross-link backreference 实现(N+3 review session 时设计层可决是否 explicit revise M6-006)。
- **Evidence**: M6.1 commit `aae6df3` + `outputs/run_20260516_090240/m6_1_byte_transformer_eval/eval_metrics.json`(per_class.Bot AUROC = 0.7402, F1 = 0.0000);M6-006 anchor evidence(原 `outputs/m6.3_cross_paradigm_summary.json` Bot 字段 + M5.5 baselines + M5.10 dim 1 trajectory)+ 本 P1 §D 更新加 M6.1 row。
- **Status**: open finding, M6-006 partial overturn;refinement of 三维耦合 paradigm scope;paper §6 Bot rare-class discussion 关键 anchor

> 🎯 **论文价值标注**
> - **Section**: Discussion §6 rare-class signal preservation + §7.3 Bot statistical edge
> - **Use**: M5-005 v3 三维耦合 + M6-006 paradigm scope finding 升级。Paper §6 Bot rare-class signal preservation narrative:Bot AUROC preservation 不是 video-paradigm-exclusive,1D byte 也 viable;但 argmax F1 = 0 是跨 paradigm uniform pathology(n=12 statistical edge)。这条对 paper "三维耦合 scope" 的 honest assessment 至关重要 — overstating findings 是 reviewer 常见 attack vector。
> - **Quote candidate**: "Bot AUROC preservation (≥ 0.67) is achieved by random-init video backbones, TimeSformer-Small under head_lr ×1 contract, and the 1D byte-level Transformer; per-class F1 = 0 for Bot across all paradigms reflects the n = 12 validation sample edge rather than a paradigm-specific limitation."
> - **Risk if missed**: Paper §6 narrative overstates "Bot 需要 temporal axis";reviewer 用 M6.1 数据 attack 这个 over-claim。本 finding 主动 refinement 把 scope 缩到 "argmax F1 paradigm-uniform 0"。
> - **Cross-link**: M5-005 v3(三维耦合)+ M6-006(原 video-only scope finding;本条 partial overturn)+ M2-008(Bot n=12 statistical edge)。
> - **Priority**: HIGH

---

## 论文价值层维护备忘

**已完成的批量标注**：
- M4 收官批量标注完成（39 条 finding 全部带 🎯 段，含 Section / Use / Figure idea / Quote candidate / Risk if missed / Cross-link / Priority 字段）
- Priority 全局分布：HIGHEST × 7 / HIGH × 7 / MEDIUM × 11 / LOW × 14
- Cross-link 双向闭合校验通过（39 条互引网络无单向悬挂）

**核心叙事簇索引（论文写作时按簇查询）**：

1. **Silent failure detection chain**（5 条 HIGHEST）：
   `TRANSITION-005` ↔ `TRANSITION-007` ↔ `M4-001` ↔ `M4-002` ↔ `M4-010a`
   论文 Discussion 章节旗舰叙事 — 两个独立 silent failure 根因 + packet/ETL/training 三层验证 + 三守恒律代数级证明 + 下游 F1 闭环。

2. **预训练运动语义迁移**（2 条 HIGHEST）：
   `M3-001` ↔ `M3-002`
   项目立意核心实证 — "陷阱 + 防御"叙事对，论文 Methods §3.4 + Discussion。

3. **CIC dataset preprocessing pitfalls**（5 条 MEDIUM/LOW）：
   `TRANSITION-001` / `TRANSITION-003` / `TRANSITION-004` / `M2-007` / `TRANSITION-006`
   论文 Reproducibility 章节 community service 清单。

4. **HF VideoMAE 适配陷阱**（5 条 LOW）：
   `M3-003` / `M3-004` / `M3-005` / `M3-006` / `M4-008`
   维护笔记，不预期论文出口。

5. **消费级硬件可复现 + memory abundant reframing**（6 条）：
   `M1-001` / `M3-007` / `M3-008` / `M3-009` / `M3-010` / `M4-010a`
   论文 Reproducibility / Computational Considerations，对应 Idea.md §4 修订。

6. **诚实数据处理方法学**（4 条）：
   `M2-001` / `M2-006` / `M2-008` / `M4-010b`
   论文 Methods §3.2 + Limitations + Experimental Setup。

7. **空间布局设计**（5 条）：
   `M2-001` / `M2-002` / `M2-003` / `M2-004` / `M2-005`
   论文 Methods §3.2 H/W 轴构造 + Channel Encoder。

8. **标注策略不对称**（4 条）：
   `TRANSITION-008` / `M4-003` / `M4-005` / `M4-010b`
   论文 Methods (Labeling Strategy) + Limitations + multi-scale 设计实证支持。

9. **ETL 性能档案**（4 条）：
   `M2-009` / `TRANSITION-002` / `M4-009` + etl_performance.md 锚点
   论文 Implementation / Computational Efficiency efficiency table 依据。

**下次 Claude conversation 标注的检查点**：
- M5 收官后做一次批量标注（baseline 复现实证）
- M6 收官前做一次完整 review，规划论文 figure / table 清单
- 投稿前做一次 sanity check：所有 HIGHEST + HIGH 发现是否已写进论文
