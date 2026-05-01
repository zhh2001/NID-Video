# 流量视频化的网络入侵检测：最终方案

> **一句话定位**：入侵检测本质是对"通信行为演化"的识别，而非"通信状态快照"的分类；视频是天然编码行为演化的模态，也是唯一同时具备空间局部性、时间因果性、运动显性与大规模预训练先验的模态。
>
> **设计原则**：所有设计决策兼顾立意完整性与硬件可行性。"轻量可部署"本身是论文卖点。

---

## 0. 目标与定位

- **目标期刊层级**：CCF-A / 顶刊（IEEE TIFS, IEEE TDSC, IEEE TON, USENIX Security 扩展版等）
- **主要数据集**：CIC-IDS 2017（主）、CIC-IDS 2018、UNSW-NB15、TON_IoT（跨数据集泛化验证）
- **核心卖点**：表征范式的本质升级（representation paradigm shift），而非模型替换
- **论文类型**："表征—归纳偏置—模型家族"三位一体的方法论文

---

## 1. 核心立意

### 1.1 问题诊断

现有网络入侵检测方法——统计特征流（CICFlowMeter + 树模型）、1D序列（包字节 + CNN/Transformer）、2D图像（FlowPic等）、2D堆叠 + LSTM/ConvLSTM——**都在不同程度上压扁了"行为演化"这个核心维度**。要么完全忽略时间，要么把时间当作外挂的序列索引，而不是与空间结构紧密耦合的共变维度。

### 1.2 核心主张

> **入侵检测 = 识别"异常的时空行为演化模式"，而不是"识别异常的快照"。**

- DDoS 的本质：包速率突变 + 源IP分布扩散速度——**运动**
- 端口扫描的本质：探测在端口空间上的漂移轨迹——**运动**
- C&C心跳的本质：周期性的时空节律——**时空纹理**
- 数据渗漏的本质：上下行对称性随时间的缓慢破坏——**运动不对称性**

**这些都是"运动态"特征，是现有表征根本无法显式编码的。**

### 1.3 Abstract 首句候选

> Network intrusion is fundamentally a **temporal behavioral** phenomenon, yet prevailing representations—flat statistics, 1D sequences, or 2D snapshots—all flatten the very dimension that defines it: the co-evolution of communication patterns in space and time. We propose to represent traffic as **video**, casting intrusion detection as a spatiotemporal motion recognition problem, and show that this reframing unlocks capabilities that no prior representation can access, all on a commodity single GPU.

---

## 2. 差异化论证

### 2.1 相对 1D / 2D / 2D+LSTM 的区别

| 维度 | 1D+Transformer | 2D+CNN | 2D+LSTM | **视频（本工作）** |
|---|---|---|---|---|
| 时间 | 显式（1D） | 无 | 外挂 | **内嵌** |
| 空间（包间结构） | 弱 | 强（单帧） | 强但帧独立 | **强且跨帧关联** |
| 运动信息 | 隐式 | 无 | 隐式（LSTM学） | **显式（时空注意力直接捕获）** |
| 时空共变 | 无 | 无 | 几乎无 | **有（核心优势）** |
| 预训练可迁移 | 少 | ImageNet | 少 | **Kinetics / SSv2 / Ego4D** |

**关键论点**：2D+LSTM 中 LSTM 处理的是"独立快照的序列"——帧间运动（变化率、方向、空间局部性）**从未被表征本身显式呈现**。视频模型把运动当作一等公民。

### 2.2 相对 3D 张量 / 3D-CNN 的本质区别（立意的生死线）

**视频 ≠ 3D 张量。** 必须讲透，否则审稿人会说"你就是 3D-CNN 换了输入"。

#### 2.2.1 数据本质：各向异性 vs 各向同性

| 维度 | 3D张量 | 视频 |
|---|---|---|
| 三轴地位 | **各向同性**（可旋转） | **各向异性**（时间 ≠ 空间） |
| 邻域 | 3D欧氏 | 2D空间 + 1D因果链 |
| 因果性 | 无 | **有** |
| 采样率 | 三轴一致 | 空间密集 / 时间稀疏 |
| 物理意义 | 静态结构的体素化 | **动态过程的离散化** |

流量数据本质是视频：时间有强因果（SYN 必先于 ACK）、时空采样率异质、语义异构。3D-CNN 的对称卷积核是**归纳偏置的错误匹配**。

#### 2.2.2 建模假设：体素聚合 vs 运动分解

- **3D建模假设**：信号是静态结构的3D分布（C3D, V-Net, 3D U-Net）
- **视频建模假设**：信号 = 外观 + 运动，两者**异构建模**
  - Two-Stream：外观 + 光流显式分离
  - (2+1)D：2D空间 + 1D时间分解
  - SlowFast：不同时间分辨率分支
  - TimeSformer / VideoMAE：divided space-time attention
  - V-JEPA：时间维度预测性自监督

**视频范式"时间与空间异构建模"的哲学与"状态—行为分解"的攻击本质天然对齐。**

#### 2.2.3 预训练生态

- 3D-CNN 预训练（医学影像、体素物体）对流量**零迁移**
- 视频模型预训练（Kinetics, SSv2）学到的是**运动语义**——爆发、扫过、周期、加速，**正是攻击的行为特征**：
  - DDoS 爆发 ≈ "爆炸/人群聚集"
  - 端口扫描 ≈ "横向移动"
  - C&C 心跳 ≈ "周期性摆动"

---

## 3. 完整 Pipeline

```
Stage 1: Ingestion
  CIC-IDS pcap → Packet parser → Label aligner

Stage 2: Framing (time-window driven)
  Time bucketing (Δt=100ms, T=16)
  → Spatial layout (H=32 semantically-ordered IP buckets × W=64 port buckets)
  → Channel encoder (C=6, including 2 explicit motion channels)

Stage 3: Tensor & augmentation
  4D tensor (T,C,H,W) = (16, 6, 32, 64)
  → Log normalisation
  → Temporal jitter + masking (no spatial flip)

Stage 4: Video backbone (single-branch + multi-scale)
  VideoMAE-Small (ViT-S/16, Kinetics/SSv2 pretrained)
  Tube patch: (2, 8, 8) → 256 tokens / sample
  Learnable scale token: 50/50 mixed Δt=100ms / Δt=1s training
  Bidirectional attention (causal optional at inference)

Stage 5: Heads & outputs
  Classification head (benign / attack family)
  Localisation head (frame + cell saliency)
  Open-set head (energy score, zero-day)
```

### 3.1 Stage 1 · 数据摄入

- 输入：CIC-IDS 2017 原始 pcap，不用 CICFlowMeter CSV
- 每包抽取：5-tuple, TCP flags, 包大小, 时间戳, 方向, 载荷长度/熵
- 标签对齐：按 CIC-IDS 官方攻击时间窗逐包打标

### 3.2 Stage 2 · 帧构造（核心设计）

**原则：时间窗口驱动，不是流生命周期驱动。**

#### 时间维度
- **Δt = 100ms per frame**（DDoS 包速率 >1000 pps，100ms 内约 100 包）
- **T = 16 frames per sample**（对齐 VideoMAE 标准输入）
- **总窗口 1.6 秒**
- 滑窗步长：50% 重叠（800ms）平衡样本量与独立性

#### 空间维度

**H = 32：源IP语义聚类排序**（不是纯哈希）
- 对每个训练窗口，按 IP 的聚合特征（总包数、主端口、方向比例）做 k-means (k=32)
- 同类 IP 分到相邻行 → 空间局部性假设在统计意义上成立
- 解决 "permutation invariance" 问题，同时自然成为一个消融卖点

**W = 64：目的端口分桶**
- 16 列常见服务端口（80, 443, 22, 53, 21, 3389, 25, 110, 143, 3306, 5432, 8080, 8443, 445, 139, 23）
- 剩余 48 列按 log₂(port) 分桶
- DDoS → 某列持续变亮；端口扫描 → 从左到右的亮线轨迹

#### 通道维度（C = 6，含 2 个显式运动通道）

| # | 通道 | 类型 |
|---|---|---|
| 1 | 包数（log scale） | 静态 |
| 2 | 总字节数（log scale） | 静态 |
| 3 | 平均包大小 | 静态 |
| 4 | TCP flags 掩码（bit-packed） | 静态 |
| 5 | **方向比例的帧间变化率 (Δ)** | **运动** |
| 6 | **帧间包数差分 (Δpacket_count)** | **运动** |

**显式运动通道的意义**：替代 two-stream 的第二分支（双 backbone 对消费级硬件开销过大），在输入层把运动特征喂给模型。论文叙事："we unify appearance and motion within a single video Transformer by injecting motion features as input channels, avoiding the computational redundancy of dual-stream designs."

#### 空帧 / UDP 处理
- **不补帧、不插值**，空帧是信息（C&C 静默、慢扫节奏）
- 视频 Transformer 基于 patch，对稀疏输入天然鲁棒
- UDP/短流：时间窗口驱动后协议无关，窗口内有包即有像素

### 3.3 Stage 3 · 张量与增强

- **张量形状**：(T, C, H, W) = **(16, 6, 32, 64)**，每样本 196K 值（比原 64×128 缩 4×）
- **归一化**：log-scale + 按通道 z-score（包数/字节长尾）
- **增强**：
  - Temporal jitter（随机起点偏移 ±50ms）
  - Temporal masking（类 MAE 风格，掩蔽 25% tokens）
  - Channel dropout（按 0.1 概率随机置零通道）
  - **禁止空间翻转**（IP/端口维度无平移不变性）

### 3.4 Stage 4 · 视频骨干（单分支多尺度）

#### 骨干：VideoMAE-Small（关键决策）
- ViT-S/16 基础，~22M 参数
- 官方发布 Kinetics-400 / SSv2 预训练权重
- **预训练先验与 Base 版一致**，只是容量小

#### Tube patch 适配
- **(2, 8, 8)** 替代原 (2, 16, 16)
- 每样本 token 数：(16/2) × (32/8) × (64/8) = **256 tokens**
- 对比 VideoMAE-B 原配 (224×224, T=16) 的 1568 tokens，缩 6×
- Position embedding 重新插值初始化（尺寸变化）

#### 通道适配（6通道 vs 预训练3通道）
- Patch embedding 第一层：前 3 通道加载预训练权重，后 3 通道 Kaiming 初始化
- 全层微调（不冻结），让运动通道在训练中与外观通道同步适配

#### 多尺度：Learnable Scale Token（替代 SlowFast）
- 训练时以 50/50 概率用 **Δt=100ms**（快速爆发攻击）或 **Δt=1s**（慢速潜伏攻击）采样
- 输入 token 序列前加一个 learnable scale token（类似 CLS）
- 推理时两种尺度各前向一次，logit 平均融合（显存不叠加，计算量×2）

**理由**：双分支 SlowFast 显存翻倍，消费级单卡扛不住；单分支 + scale token 是更优雅的统一建模，且可作为"轻量部署"卖点。

#### 注意力方向
- **训练：双向 attention**（与预训练权重一致，最大化迁移红利）
- **推理：可切换 causal mask**（在线部署场景）
- 论文里双向作为主方法，causal 作为"在线部署变体"报告，**两个场景都有故事**

### 3.5 Stage 5 · 输出头

- **分类头**：benign + 攻击家族多分类（CIC-IDS 2017 共 14 类）
- **定位头**：帧级 + cell 级的 attention rollout / saliency map（可视化）
- **开放集头**：energy score 或 MSP，评估零日攻击检测

---

## 4. 训练配置与硬件可行性

> **修订说明（M4.8 实测后）**：本节 §4.1–4.4 早期基于 M3 合成数据估算
> ("8 GB 显存吃紧 → batch=2 + grad accumulation=16" 的内存优先叙事)。
> M4.8 真实 CIC-IDS-2017 100ms+1s 多尺度训练完成后，实测峰值显存仅
> 485 MB，远低于 8 GB 上限——内存不再是约束，吞吐量才是。M4 默认配置
> 因此切换至 `configs/training_perf.yaml`（B=32 / accum=1 / workers=2），
> 单 epoch 1237 秒 / 4853 grad steps / macro-F1=0.45。详细测量见
> `docs/m3_perf.md` + `docs/v1_vs_v2_comparison.md`，本节保留原内存估算
> 表作为消融实验的基线参考。

### 4.1 显存节省组合拳

下表是 M3/M4 早期为 8 GB 单卡考虑的内存优化栈。M4.8 实测显示 batch=32
直跑（无 accumulation）已能稳定运行，前两项 (AMP + GC) 仍是默认开启
但更多是"白送的优化"而非"必要约束"。剩下两项可作 ablation 路径。

| 技术 | 显存节省 | 代价 | M4.8 现状 |
|---|---|---|---|
| FP16 混合精度 (AMP) | ~40% | ≈无 | ✓ default-on |
| Gradient checkpointing | ~30%（叠加） | 慢 20% | ✓ default-on |
| 8-bit AdamW（bitsandbytes） | ~30% 优化器状态 | ≈无 | ✓ default-on |
| Batch=2 + grad accumulation=16 | 等效 batch=32 | 慢 7-12× | ✗ 改用 B=32 直跑 |
| 冻结 patch embed 前 2 层 | ~15% | 可能掉点 | 未启用，作 ablation |

### 4.2 显存预算估算（旧）vs 实测（新）

| 项 | M3 估算 | M4.8 实测 |
|---|---|---|
| 模型参数（VideoMAE-S, FP16） | 44 MB | 同（44 MB） |
| 优化器状态（8-bit AdamW） | 44 MB | 同（44 MB） |
| 激活 + 梯度（GC + AMP） | batch=2: ~3–4 GB | **batch=32: ~300 MB** |
| dataloader + cudnn workspace | ~1 GB | ~100 MB |
| **峰值显存合计** | **~5 GB（估算偏高）** | **485 MB（实测）** |

实测远低于估算的原因：tube patch (2,8,8) 把 (T=16, H=32, W=64) 输入压成
4×4×8=128 token / 256 patch（含 scale token 共 257），远小于原 VideoMAE
预训练 8×14×14=1568 token 的体量，激活体量随之线性下降。M3 估算用了
"原 VideoMAE 体量 × 6/3 通道 × batch=2"的保守乘法，未考虑 token 数差异。

8 GB 卡有 ~7 GB 余量；12 GB 及以上卡可考虑 batch=64 或更大。

### 4.3 训练规模（M4.8 实测）

| 项目 | M3 估算（修订前） | M4.8 实测 |
|---|---|---|
| CIC-IDS-2017 Tue+Wed+Fri 100ms 窗口 | ~50万–100万（projection） | **110,783 windows** |
| 多尺度 1s 窗口（Δt=1s 旁路） | n/a | **11,074 windows** |
| Train / Val / Test 分割（70/15/15 by 类） | n/a | **77,615 / 16,463 / 16,705** |
| Batch 配置 | batch=2 × accum=16 | **batch=32 × accum=1（等效 32）** |
| 微调 epochs（强预训练） | 20 | 20（M4.8 单 epoch 验证 macro-F1=0.45） |
| 单 epoch wall time（1 GPU） | ~1.5 小时 projection | **1237 秒（≈ 20.6 分钟）** |
| 单 epoch grad steps | 1000 (heuristic) | **4853** |

实测窗口数比 projection 偏低（原估算未考虑 dominant-rule 标注 + sliding
window 重叠的实际密度）。M4.8 单 epoch 已显著学习；20 epochs 总训练
时间约 6.9 小时（vs M3 估算的 30+ 小时）。

### 4.4 Baseline 训练可行性

| Baseline | 参数量 | 单卡可行性 |
|---|---|---|
| RF / XGBoost on CICFlowMeter | — | ✓ CPU 即可 |
| 1D Transformer on packet bytes | ~10M | ✓ |
| 2D CNN (FlowPic) | ~5M | ✓ |
| ConvLSTM | ~15M | ✓ 略吃紧 |
| C3D | ~80M → 用 C3D-Small 变体 ~20M | ✓（需缩减） |
| I3D | ~12M (RGB only) | ✓ |
| R(2+1)D | ~33M → 用 R(2+1)D-18 | ✓ |
| TimeSformer-S | ~22M | ✓ |
| **VideoMAE-S（本工作）** | **22M（peak 485 MB）** | **✓ 8 GB 卡有大量余量** |

**关键**：C3D 原版 80M 显存会超，需要用 C3D-Small 变体（等比缩减通道数，~20M）。论文里需要诚实声明"for fair comparison under the same compute budget"——这本身也强化了"轻量可部署"叙事。

M4.8 实测进一步印证轻量化叙事：485 MB peak GPU 意味着本方法在
laptop-class GPU（RTX 4060 Mobile 8 GB）上即可完成全量训练 +
real-time 推理，对 NID 场景的边缘部署友好。

---

## 5. Open Issues 的最终决策

### Issue 1：IP 哈希桶的 permutation invariance

**决策：语义聚类排序（不是纯哈希）**

- 做法：每个训练窗口先对活跃 IP 按行为特征做 k-means (k=32)，同类分到相邻行
- 优势：空间局部性假设成立；零成本；成为一个独立的小贡献
- 消融：随机哈希 vs 语义排序 vs 完全打乱三种布局对比，预期 3-5 pt F1 差距
- 反驳"视频≠3D"质疑的额外武器（3D对称性用不了这种布局）

### Issue 2：时间因果性

**决策：训练双向 + 推理可选 causal**

- 训练：双向 attention，完全利用预训练权重
- 推理：报告双向（离线检测）+ causal（在线检测）两种模式
- 论文处理：讨论章节明确区分"离线 vs 在线"部署场景，主实验用双向

### Issue 3：流量"光流" / two-stream

**决策：不独立分支，在 C=6 中保留 2 个显式运动通道**

- 通道 5：方向比例帧间变化率
- 通道 6：帧间包数差分
- 叙事："unify appearance and motion within a single Transformer"
- 消融：C=4 无运动通道 vs C=6 有运动通道，预期 1-3 pt 提升

### Issue 4：多尺度双分支

**决策：单分支 + learnable scale token + 多尺度训练**

- 训练 50/50 混合 Δt=100ms / 1s
- 推理两种尺度融合
- 论文叙事：主方法轻量（消费级单卡可训），避免双 backbone 冗余

---

## 6. 贡献点清单

1. **表征范式创新（主贡献）**：首次把流量建模为异构时空视频，明确区分于 2D+LSTM 与 3D 张量范式
2. **语义感知的空间布局**：通过行为聚类实现 IP 维度的空间局部性，解决哈希布局的 permutation 问题
3. **统一的外观-运动建模**：通过显式运动通道在单 Transformer 内完成，避免 two-stream 冗余
4. **多尺度尺度 token**：单分支实现快慢攻击的统一检测，替代 SlowFast 双分支
5. **视频预训练的跨模态迁移**：首次论证 Kinetics/SSv2 的运动语义可迁移到流量行为分析
6. **消费级硬件可训练**：全流程在消费级单卡（8GB 显存量级）即可完成，推动 NID 研究的可复现性
7. **可解释性**：时空 attention 让"模型看到的攻击"可被人类判读
8. **零日 / 开放集鲁棒性**：运动先验对未见攻击的泛化
9. **部署变体**：离线双向 / 在线 causal 两种推理模式

---

## 7. 实验矩阵

### 7.1 数据集

| 数据集 | 用途 |
|---|---|
| CIC-IDS 2017 | 主实验 |
| CIC-IDS 2018 | 跨年份泛化 |
| UNSW-NB15 | 跨数据集泛化 |
| TON_IoT | IoT 场景泛化 |

### 7.2 Baselines

| 类别 | 方法 | 证明什么 |
|---|---|---|
| 传统 ML | RF / XGBoost on CICFlowMeter | 打底线 |
| 1D 序列 | Transformer on packet byte sequence | 时间但无空间 |
| 2D 图像 | FlowPic-style CNN | 空间但无时间 |
| 2D + 时序 | ConvLSTM | 外挂时间 |
| **3D 体素** | **C3D-Small, I3D** | **关键：视频 ≠ 3D** |
| (2+1)D | R(2+1)D-18 | 异构分解的必要性 |
| **视频 Transformer** | **TimeSformer-S, VideoMAE-S（本工作）** | **主方法** |

### 7.3 评估维度

1. 标准指标：Accuracy, macro-F1, per-class P/R, AUC
2. 类别不均衡：macro-F1、稀有攻击 Recall
3. 零日检测：留一攻击类做 OOD
4. 概念漂移：2017 训练 → 2018 测试
5. 跨数据集：CIC-IDS → UNSW-NB15 / TON_IoT
6. 效率：推理延迟、吞吐、**峰值显存（强调消费级单卡可跑）**
7. 可解释性：saliency 与真实攻击对齐度

### 7.4 必做消融

| 消融 | 变量 |
|---|---|
| 帧率 | Δt ∈ {10, 50, 100, 500, 1000} ms |
| 帧数 | T ∈ {4, 8, 16, 32} |
| **空间布局** | **随机哈希 / 语义聚类 / 完全打乱** |
| 通道组合 | 逐通道移除 |
| **运动通道** | **C=4（无运动）vs C=6（有运动）** |
| Scale token | 单尺度 vs 多尺度 + token |
| 预训练 | 随机 / ImageNet-video / Kinetics / SSv2 |
| **核心消融** | **VideoMAE-S vs C3D-Small vs I3D (on same tensor)** |
| Patch size | (2,4,4) / (2,8,8) / (2,16,16) |
| 注意力方向 | 双向 / causal |

---

## 8. 论文故事线

1. **Hook**：入侵检测的本质是识别行为演化，不是快照
2. **问题**：现有表征全部压扁或外挂时间
3. **3D 的诱惑与陷阱**：自然想到 3D 卷积？但流量是异构的，3D 对称假设错位
4. **核心洞察**：流量天然是视频——异构时空、运动语义、行为演化
5. **方法概览**：时间窗口驱动的帧构造 + 视频 Transformer + 语义空间布局 + 显式运动通道
6. **意外红利**：Kinetics 预训练的运动先验竟能迁移到流量——这不是巧合，因为攻击本就是运动模式
7. **可部署性**：整套在消费级单卡上可训，推动领域可复现性
8. **贡献**（9 点）
9. **结果预告**：刷过所有 baseline，尤其在零日 / 低速 / 跨数据集优势显著，附可解释的时空定位

---

## 9. 预防审稿人质疑

| 质疑 | 回应 |
|---|---|
| "就是 3D-CNN 换输入" | 数据/建模/预训练三方面论证，加上 C3D/I3D 消融实证差距 |
| "为什么不用纯 Transformer" | 时空局部性归纳偏置在数据有限时更优；做数据效率曲线 |
| "计算开销大" | **消费级单卡即可训练，峰值约 5GB 显存**——这是反过来的卖点 |
| "CIC-IDS 刷烂了" | 不追 accuracy，重点打零日、漂移、跨数据集 |
| "帧布局任意" | 语义聚类排序的消融直接回应 |
| "慢速攻击怎么办" | Scale token 多尺度训练 |
| "UDP / 短流怎么办" | 时间窗驱动，协议无关 |
| "IP 哈希局部性不合理" | 语义聚类是解决方案 + 消融证据 |
| "双向注意力违反因果" | 离线双向 + 在线 causal 两种模式 |
| "22M 参数不够大" | 22M + 强预训练打过更大的随机初始化模型，是价值而非缺陷 |

---

## 10. 阶段规划

| 阶段 | 任务 | 产出 |
|---|---|---|
| **阶段 1 · 数据工程** | ETL pipeline：pcap → 4D tensor（含语义聚类、运动通道） | 可复用的数据集构造代码、预处理后的张量数据集 |
| **阶段 2 · 主方法构建** | VideoMAE-S 单分支训练，显存调优，打通训练流程 | 主方法的首个可训版本、训练日志 |
| **阶段 3 · 多尺度扩展** | 加入 learnable scale token，实现多尺度训练与推理 | 多尺度版主方法 |
| **阶段 4 · Baseline 复现** | C3D-Small, I3D, R(2+1)D, TimeSformer-S, ConvLSTM, 1D Transformer 等 | 统一评测框架、所有 baseline 的可对比结果 |
| **阶段 5 · 核心消融** | 空间布局、运动通道、预训练、视频 vs 3D-CNN | 消融表格、支撑立意的实证证据 |
| **阶段 6 · 泛化验证** | 跨数据集（2018, UNSW-NB15, TON_IoT）、零日检测、概念漂移 | 泛化性结果表 |
| **阶段 7 · 可解释性与鲁棒性** | 可视化、causal 推理、对抗鲁棒性 | 可视化图集、鲁棒性分析 |
| **阶段 8 · 效率与部署** | 推理延迟、吞吐、峰值显存、部署变体对比 | 效率分析表 |
| **阶段 9 · 论文撰写** | 图表打磨、Introduction 与 Method 撰写、投稿准备 | 投稿版论文 |
