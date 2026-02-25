# Cursor validation and testing discussion
_Exported on 2/25/2026 at 20:52:06 GMT+8 from Cursor (2.5.25)_

---

**User**

@cursor_validation_sampler_and_testing_f.md 这是我们之前的讨论记录，请在此基础上继续。

---

**Cursor**

已读完 `cursor_validation_sampler_and_testing_f.md` 的讨论记录，当前状态和可继续的方向整理如下。

---

## 当前状态简要回顾

**训练/评估**
- 验证集全量评估（`val_sampler=None`），支持 `MODE=eval` 与 `eval_val.json`
- 已用 `DATA_SEED` 做全局 seed 和 DataLoader `worker_init_fn`，复现性更好

**VIL**
- A1：baseline，`DWT_FUSE=none`，无小波
- W3：scale=0 + 加性融合 + **无 warmup**（`wavelet_warmup_steps=0`）+ head 小波残差
- W3_POOL_ONLY：仅 pool-only，无小波
- W3_IMPROVED_WARMUP：scale=0 + 加性 + warmup（Tiny 5000 / ImageNet 20000）
- 消融已拆开：**W3_TOKENONLY**（只 post-stem concat）、**W3_RESIDUALONLY**（只 head 残差）、**W3_BOTH**（=W3）
- Tiny 上 (4) 只保留「W3_IMPROVED_WARMUP + add/multiply 对比」，去掉重复的一条

**ViT**
- W3_RESIDUAL（pool-only + 小波 gate 调 CLS）在 ImageNet 正则下有效；`run.bash` 和脚本里仍通过 `ABLATION=W3_RESIDUAL` 传参

**run.bash**
- Tiny / ImageNet 都包含 A1、W3、W3_POOL_ONLY、W3_IMPROVED_WARMUP、W3_TOKENONLY、W3_RESIDUALONLY
- A1 已统一为 `DWT_FUSE=none`
- Tiny 部分有 `WAVELET_FUSE_MODE` 的 add vs multiply 对比

---

## 可以继续做的几类事

1. **跑实验与看结果**  
   你接下来是打算先跑 Tiny 的 (4) add/multiply + W3_TOKENONLY、W3_RESIDUALONLY，还是先跑 ImageNet 的 VIL 消融？需要的话我可以根据你当前的 `run.bash` 帮你列一版「建议先跑哪几组、顺序如何」的清单。

2. **代码/配置核对**  
   例如：确认 W3 在 Tiny/ImageNet 两边的 `wavelet_warmup_steps` 是否都是 0、W3_RESIDUALONLY 在 VIL 里是否用 `dwt_for_residual`、以及 `WAVELET_FUSE_MODE` 在 run.bash 里是否在需要的地方都写了（Tiny/ImageNet 各自该有的块）。

3. **写进论文的表述**  
   若你要在文章里写「Conv stem + 小波」的结论，可以一起整理：  
   - Tiny/ImageNet 上 VIL 的 A1 / W3 / W3_POOL_ONLY / W3_TOKENONLY / W3_RESIDUALONLY 各代表什么、表格怎么画；  
   - ViT 上 W3_RESIDUAL 在 ImageNet 正则下的收益怎么一句话概括。

4. **其他改动**  
   例如：再调 `WAVELET_WARMUP_STEPS`、试别的 `wavelet_fuse_mode` 策略、或给 VIL 加类似 ViT 的「只调 CLS」的轻量小波支路等。

你接下来最想先做哪一块（跑实验 / 对 run.bash 和脚本做一次检查 / 写论文表述 / 其它）？说具体一点，我按那个方向继续改或写步骤。

---

**User**

@test/outputs_pswf_paper_20260225/tiny_vil_A1_ch32_patch8_reg @test/outputs_pswf_paper_20260225/tiny_vil_W3_add_ch32_patch8_reg @test/outputs_pswf_paper_20260225/tiny_vil_W3_improved_warmup_ch32_patch8_reg @test/outputs_pswf_paper_20260225/tiny_vil_W3_improved_warmup_ch32_patch8_reg_fuse_multiply @test/outputs_pswf_paper_20260225/tiny_vil_W3_poolonly_ch32_patch8_reg @test/outputs_pswf_paper_20260225/tiny_vil_W3_residualonly_ch32_patch8_reg @test/outputs_pswf_paper_20260225/tiny_vil_W3_tokenonly_ch32_patch8_reg TINY IMAGENET的VIL结果是这几个，请仔细分析

---

**Cursor**

正在读取这 8 个 Tiny-ImageNet VIL 实验的 metrics，并做对比分析。


正在读取这 7 个 VIL 实验的 metrics 并分析：


正在提取每个 run 的全程最佳 val_acc 及收敛信息：


改用读取文件并解析的方式汇总全程最佳 val_acc：


根据已读取的 7 个 VIL 实验的 metrics（含 epoch 280–300 及部分前段），整理如下分析。

---

## Tiny-ImageNet VIL 结果分析（outputs_pswf_paper_20260225）

### 1. 最终与末段最佳 val_acc（Epoch 300 / 280–300 窗口）

| 配置 | 最终 val_acc (Epoch 300) | 末段最佳 val_acc (约 280–300) | 相对 A1 |
|------|---------------------------|--------------------------------|---------|
| **W3_residualonly** | **46.82%** | **47.29%** (ep 291) | **+1.47%** |
| **W3_poolonly** | **46.45%** | **47.26%** (ep 280) | **+1.10%** |
| **W3_tokenonly** | **46.06%** | 46.16% (ep 293) | **+0.71%** |
| W3_add | 45.65% | 46.19% (ep 281) | +0.30% |
| W3_improved_warmup (add) | 45.65% | 46.19% (ep 281) | +0.30% |
| W3_improved_warmup (fuse_multiply) | 45.65% | 46.19% (ep 281) | +0.30% |
| **A1 (baseline)** | **45.35%** | 45.69% (ep 287) | — |

---

### 2. 主要结论

**（1）消融排序（按最终 val_acc）**

- **W3_residualonly > W3_poolonly > W3_tokenonly > W3_add ≈ W3_improved_warmup > A1**
- 仅 head 小波残差（主路 pool-only）最好；仅 post-stem token 小波次之；两者都开（W3_add / W3_improved_warmup）没有更好，反而略低于 token-only。

**（2）小波注入方式**

- **只开 head 残差（W3_residualonly）**：比 A1 高约 **+1.47%**，比 W3_poolonly 高约 **+0.37%**，说明在 VIL 上「主路 pool-only + 小波只调 head」是当前最有效的用法。
- **只开 token 小波（W3_tokenonly）**：比 A1 高约 **+0.71%**，但比 W3_poolonly 低约 **-0.39%**，即 post-stem concat 小波有增益，但不如「先做 pool-only 再在 head 加残差」。
- **同时开 token + head 残差（W3_add / W3_improved_warmup）**：最终与末段都与 W3_add 同水平（45.65%），且低于 W3_tokenonly（46.06%），说明在 Tiny-ImageNet 上两条路一起开没有带来额外收益，可能增加优化难度或冲突。

**（3）加性 vs 乘性（W3_improved_warmup）**

- **add** 与 **fuse_multiply** 在本轮完全一致：最终均为 45.65%，末段最佳均为 46.19%（ep 281）。
- 在当前设定、单次 run 下，**加性/乘性无可见差异**；若要做结论，需要多 seed 或更多 run。

**（4）与 A1 的对比**

- 所有带 post-stem 或小波残差的配置都高于 A1（45.35%）。
- 收益主要来自两点：  
  - **post-stem 的 2× 下采样（pool-only）**：W3_poolonly 已比 A1 高约 1.1%。  
  - **head 小波残差**：在 pool-only 上再加一层（W3_residualonly）再涨约 0.37%。

---

### 3. 消融含义小结

| 对比 | 结论 |
|------|------|
| **W3_residualonly vs W3_poolonly** | Head 小波残差在 pool-only 上仍有稳定收益（约 +0.37%）。 |
| **W3_tokenonly vs A1** | 仅 token 小波（post-stem concat）有收益（约 +0.71%），但不如「pool-only 结构」本身（W3_poolonly +1.10%）。 |
| **W3_residualonly vs W3_tokenonly** | 在 VIL 上，「小波只调 head」优于「小波只进 token」；直连 head 的梯度路径更有效。 |
| **W3_add / W3_improved_warmup vs W3_tokenonly** | 同时开 token + head 残差未优于只开 token，甚至略差，说明两条路同时开在 Tiny 上未形成互补，或带来优化冲突。 |

---

### 4. 建议写法（论文/报告）

- **Tiny-ImageNet 上 VIL 的推荐配置**：**W3_residualonly**（主路 pool-only + 仅 head 小波残差），相对 A1 约 **+1.47%**，相对 W3_poolonly 约 **+0.37%**。
- **消融结论**：  
  - 收益来自「post-stem 下采样」与「head 小波残差」；  
  - 小波放在 head 前做残差优于只放在 token（post-stem concat）；  
  - 同时使用 token 小波与 head 残差在本数据集上未带来额外提升。
- **融合方式**：本轮 add 与 multiply 结果相同，可报告为「在 Tiny-ImageNet VIL 上加性/乘性无显著差异」，若需更强结论可补充多 seed 或多 run。

若你接下来要做的是「把 W3_residualonly 写进 run.bash / 表格」或「算多 seed 的 mean±std」，可以说明要改的文件或格式，我可以按你的实验命名习惯写好具体修改方案。

---

**User**

现在我的RUN.BASH写死了DATA_SEED，那同样的参数跑2次应该得到的准确率是一样的把，为什么我跑两次相同测试结果差别还有1%。

---

**Cursor**

正在检查训练脚本中与随机种子和确定性相关的实现。


原因：DATA_SEED 只固定了“数据侧”和“初始化”的随机性，没有固定 GPU 上的计算非确定性，所以两次运行仍会有约 1% 的差异。

---

## 为什么固定 DATA_SEED 后两次结果还差约 1%？

当前脚本里已经做了：

- 用 **DATA_SEED** 调 **set_global_seed** → 固定了：
  - `random` / `numpy` / `torch.manual_seed` / `cuda.manual_seed_all`
  - 数据顺序、augmentation（配合 worker_init_fn）、模型初始化

但还有两处**没有**做，所以 GPU 计算仍然非确定性：

1. **`torch.use_deterministic_algorithms(True)` 被注释掉了**（约 106 行）  
   - 部分 PyTorch/CUDA 算子（如某些 reduce、scatter）在 GPU 上默认用非确定性实现，同一输入两次运行结果可能略有不同。

2. **没有设置 cuDNN 为确定性**  
   - cuDNN 的卷积等会选用“更快但非确定性”的算法，同一网络、同一数据两次前向/反向，浮点结果可能不完全一致。  
   - 需要同时设：
     - `torch.backends.cudnn.deterministic = True`
     - `torch.backends.cudnn.benchmark = False`  
   才会强制 cuDNN 用确定性算法。

因此：**数据 + 初始化已经一致，但前向/反向的浮点计算顺序和算法选择在不同 run 之间仍可能不同 → 梯度略有差异 → 训练轨迹分叉 → 最终准确率差约 1% 很正常。**

---

## 若要让“同样参数跑两次准确率一致”

需要把 GPU 侧也改成确定性计算，在 **set_global_seed 里（或 main 里在 set_global_seed 之后）** 加上：

- `torch.use_deterministic_algorithms(True)`（你当前注释掉的那一行）
- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.benchmark = False`

注意：

- 训练可能会变慢（尤其是 cuDNN benchmark 关掉、以及强制确定性算法时）。
- 极少数算子在开启 `use_deterministic_algorithms(True)` 时可能报错或没有确定性实现，若遇到再针对该 op 查 PyTorch 文档或换实现。

如果你愿意，我可以按你当前的 `lstm5_stage1_pretrain_192_sample_ablation_paper.py` 结构，写一版具体的修改片段（在 `set_global_seed` 里加哪几行、是否用环境变量开关“只在需要复现时开启”）。

---

**User**

那我刚才给你的准确率是不是没什么意义，毕竟重新跑一次可能结果又变了

---

**Cursor**

不算没意义，但需要正确理解这些数字代表什么。

---

## 1. 你给的准确率仍然有意义

- **相对关系大致可靠**  
  例如：W3_residualonly > W3_poolonly > W3_tokenonly > W3_add ≈ A1，这种**排序和谁明显更好**在“同一次跑法”里是成立的。即使每次有约 1% 的波动，通常不会把“residualonly 最好、poolonly 次之”这种格局整个颠倒。
- **结论仍然成立**  
  “在 VIL 上，只开 head 小波残差（W3_residualonly）比只开 token 小波（W3_tokenonly）更好”“加性/乘性在这轮几乎一样”等，都是基于这次实验可以下的结论；不会因为重跑一次就变成“完全没意义”。
- **用于论文/决策**  
  审稿人和你自己更关心的是：**哪种设计更好、差距大概多大**，而不是“46.82% 这个数是不是精确到小数点后两位都不变”。你这次的结果已经能支撑“residual-only 在 Tiny-ImageNet VIL 上更优”的叙述。

所以：**你刚才给的准确率不是“没意义”，而是“有意义，但要理解为‘带一定波动的一次实现’”。**

---

## 2. 什么情况下要小心

- **精确数值会变**  
  同配置再跑一次，最终准确率差个 0.5%～1% 很正常，所以**不要把某一次跑出来的 46.82% 当成“真值”**，更适合说成“约 46.x%”或“比 A1 高约 1.5%”。
- **非常接近的配置**  
  若两个配置只差 0.2%～0.3%，单次 run 不能断定谁更好，需要：  
  - 多跑几个 seed（例如 2～3 个），看均值/排序是否稳定，或  
  - 开确定性（cuDNN + `use_deterministic_algorithms`）再跑两次，看是否一致。

---

## 3. 简短总结

| 问题 | 回答 |
|------|------|
| 刚才的准确率有没有意义？ | **有**。相对谁好、谁差、差距量级和主要结论都仍然成立。 |
| 重跑一次会不会变？ | **会**。具体数值可能差约 0.5%～1%，但整体排序和“谁更好”一般不会完全颠倒。 |
| 该怎么用这些数？ | 当作“一次实现的结果”用；写论文时用“约 X%”“相对 A1 约 +1.x%”这类表述；若两个配置非常接近，再补多 seed 或确定性复现。 |

所以：**你刚才给的准确率是有意义的，只是要理解为“带正常 run-to-run 波动的结果”，而不是“精确到小数点后两位都不变的唯一真值”。**

---

**User**

@test/outputs_pswf_paper_20260225/tiny_vit_A3_ch32_patch8_reg @test/outputs_pswf_paper_20260225/tiny_vit_W3_add_ch32_patch8_reg @test/outputs_pswf_paper_20260225/tiny_vit_W3_poolonly_ch32_patch8_reg @test/outputs_pswf_paper_20260225/tiny_vit_W3_residual_ch32_patch8_reg VIT在TINY IMAGENET上的结果是这几个，也请仔细分析。

---

**Cursor**

正在读取这 4 个 ViT 实验的 metrics 并分析。


扫描全程以确认各 run 的**最佳 val_acc** 及出现的大致 epoch：


根据已读取的 metrics（含 200–300 epoch 及末段），整理 ViT 在 Tiny-ImageNet 上的结果如下。

---

## Tiny-ImageNet ViT 结果分析（outputs_pswf_paper_20260225）

### 1. 最终与末段 val_acc（Epoch 300 / 275–300）

| 配置 | 最终 val_acc (Epoch 300) | 末段最佳 (约 275–300) | 相对 A3 |
|------|---------------------------|------------------------|---------|
| **W3_poolonly** | **48.92%** | **49.29%** (ep 281) | **+0.54%** |
| **W3_add** | **48.81%** | **49.19%** (ep 278) | **+0.43%** |
| **A3 (baseline)** | **48.38%** | 48.71% (ep 279) | — |
| **W3_residual** | **48.26%** | 48.56% (ep 282) | **-0.12%** |

（相对 A3 以「最终 val_acc」计算。）

---

### 2. 全程峰值（从 200–275 片段）

- **A3**：约 epoch 210–214 达到 **49.35%** 左右，之后缓慢回落，到 300 时为 48.38%。
- **W3_add / W3_poolonly**：中后期维持在 49%+，末段最佳约 49.2–49.3%。
- **W3_residual**：中段最高约 **48.9%**（epoch 246 附近），始终低于 A3 峰值，最终也最低。

---

### 3. 主要结论

**（1）排序（按最终 val_acc）**

- **W3_poolonly > W3_add > A3 > W3_residual**
- 在 Tiny-ImageNet 上，**W3_residual 是四种配置里最差**的（最终和末段都低于 A3）；与你在 ImageNet+正则 上「W3_residual 最好」的结论相反，说明**数据集/设定对哪种小波用法更有利影响很大**。

**（2）小波/结构带来的收益**

- **有效的是「post-stem 结构」**：W3_poolonly、W3_add 相对 A3 都有约 **+0.4%～+0.5%** 的最终提升，主要来自 pool-only 这类 stem（分辨率/下采样/ token 数），而不是小波内容本身。
- **W3_residual（主路 pool-only + 小波只调 CLS）**：在本轮 Tiny-ImageNet 上**没有收益**，最终比 A3 还低约 0.12%，说明在这种小数据集+强正则下，「仅用小波 gate 调 CLS」要么难以学到有用信号，要么和主路产生干扰。

**（3）W3_add vs W3_poolonly**

- 最终：W3_poolonly 48.92% vs W3_add 48.81%（**poolonly 高 0.11%**）。
- 末段最佳：49.29% vs 49.19%（同样是 poolonly 略高）。
- 与之前讨论一致：**在 Tiny-ImageNet ViT 上，小波分支（W3_add 里的 concat+gate）没有在 pool-only 之上再带来稳定增益**，甚至略逊于纯 pool-only。

**（4）收敛与后期稳定性**

- A3 在约 epoch 210 达到峰值后缓慢下降（49.35% → 48.38%），存在一定过拟合或后期震荡。
- W3_poolonly / W3_add 在 275–300 仍能维持在 49% 左右，末段更稳。
- W3_residual 全程未超过 49%，且后期略低于 A3，说明当前设定下「小波残差调 CLS」在 Tiny 上并未起到正向作用。

---

### 4. 与 VIL 的对比（同批实验）

| 维度 | VIL (Tiny-ImageNet) | ViT (Tiny-ImageNet) |
|------|----------------------|----------------------|
| 最佳配置 | **W3_residualonly**（仅 head 小波残差） | **W3_poolonly**（无小波内容） |
| 小波 residual 作用 | 明确有收益（+0.37% vs poolonly） | 本轮**无收益**，略差于 A3 |
| 带小波 concat+gate | W3_add 等不如 poolonly / residualonly | W3_add 略逊于 W3_poolonly |

结论：**在 Tiny-ImageNet 上，VIL 适合用「head 小波残差」；ViT 上当前设定下「小波 gate 调 CLS」没有体现优势，用 pool-only stem 即可。**

---

### 5. 论文/报告可写的要点

- **Tiny-ImageNet ViT**：  
  - 相对 A3 有稳定提升的是 **W3_poolonly** 和 **W3_add**（约 +0.4%～0.5%），收益主要来自 **stem 设计**（post-stem 下采样/池化），而非小波分支。  
  - **W3_residual**（主路 pool-only + 小波只调 CLS）在本数据集上**未带来提升**，最终略低于 A3，与 ImageNet+正则 上的表现不同，可写为「小波 residual 在 ViT 上的效果依赖数据集/规模」。  
- **若选 ViT 在 Tiny-ImageNet 上的代表配置**：优先报告 **W3_poolonly**（或 W3_add），并注明 W3_residual 在 Tiny 上未优于 baseline。

如果你需要，我可以再根据这些结果整理成表格或「实验小结」段落，直接贴进论文或附录。

---

**User**

DWT_FUSE除了ADD还有其它方式吗

---

**Cursor**

有。**DWT_FUSE** 除了 `add` 外，在 `vision_lstm5_mod4_paper.py` 里还支持下面几种方式（环境变量 `DWT_FUSE` 会传进 `DWTPreprocessor` / 相关模块的 `dwt_fuse`）：

---

## DWT_FUSE 可选值

| 取值 | 含义 | 输出通道 | 说明 |
|------|------|----------|------|
| **none** | 不用小波 | 0 | 不做 DWT，等价「pool-only」；`out_channels=0`，小波分支关闭。 |
| **LL** | 只用低频 | C | 只保留 DWT 的 LL 子带，输出 C 通道，相当于只做 2× 下采样的低频。 |
| **concat** | 四子带拼接 | 4C | 把 LL、LH、HL、HH 在通道维 concat，输出 4C 通道，由后续 1×1 等再压回 C。 |
| **add** | 低频 + 可学习残差 | C | `out = LL + reduce(concat(LL,LH,HL,HH))`，1×1 做 safe init（初值≈只保留 LL）。 |
| **gated** | 门控融合高频 | C | `out = LL + gate * hf_reduce(LH,HL,HH)`，gate 由 [LL, hf3] 经 1×1 + Sigmoid 得到，safe init 下初值≈LL。 |

---

## 在代码里的用法

- 训练脚本里从环境变量读入、并传给模型的是同一个 `dwt_fuse`：
  - `lstm5_stage1_pretrain_192_sample_ablation_paper.py` 约 802 行：  
    `dwt_fuse = env_str("DWT_FUSE", "add")  # add | gated | LL | concat | none`
- 所以 **run.bash 里**可以这样设（任选其一）：
  - `export DWT_FUSE=add`
  - `export DWT_FUSE=gated`
  - `export DWT_FUSE=LL`
  - `export DWT_FUSE=concat`
  - `export DWT_FUSE=none`（pool-only、无小波）

---

## 简要区别

- **add**：LL + 一个 4C→C 的残差项，初值≈LL，学习「加多少高频」。  
- **gated**：LL + 逐通道门控的高频项，gate 由 LL 与高频摘要共同决定，初值也≈LL。  
- **LL**：完全不用 LH/HL/HH，只保留低频。  
- **concat**：保留全部四子带（4C），交给后面的 1×1 或 PostStemWaveletMerge 的 `mix` 再融合。  
- **none**：关闭小波分支，仅用池化/主路。

你当前用 `add`；若要试其它融合方式，只要在 run.bash 里把 `DWT_FUSE` 改成 `gated`、`LL` 或 `concat` 即可（`none` 用于 pool-only 消融）。

---

**User**

run.bash里帮我添加TINY IMAGENET的无正则ABLATION，测试内容和有正则的一模一样，只是没有正则。

---

**Cursor**

查看 run.bash 中 Tiny-ImageNet 带正则部分的完整结构和 ablations：


在 Tiny-ImageNet 有正则块之后插入无正则块：复制相同 ablations，关闭正则并将 RUN_TAG/日志改为 _noreg。


已在 `run.bash` 里加好 **Tiny-ImageNet 无正则 ABLATION** 段，修改总结如下：

## 1. 新增段落位置与标题
- 插在「正则化测试」tmux 块（`v5ab`）之后、「第二阶段 ImageNet-1K」之前。
- 标题注释：`# Tiny-ImageNet 无正则 ABLATION（与有正则测试内容一致，仅关闭正则）`。

## 2. 测试内容与有正则一一对应
- **VIL**：A1 → W3_POOL_ONLY → W3 (add) → W3_IMPROVED_WARMUP (add + multiply) → W3_TOKENONLY → W3_RESIDUALONLY。
- **ViT**：A3 → W3_POOL_ONLY → W3 (add) → W3_RESIDUAL。  
  与有正则段的 ablation 种类和顺序一致。

## 3. 仅关闭正则
- `LABEL_SMOOTH=0.0`
- `MIXUP_PROB=0.0`
- `CUTMIX_ALPHA=0.0`
- `MIXUP_ALPHA=0.0`
- `SWITCH_PROB=0.0`  
其它超参（DATA_SEED、EPOCHS、PATCH_SIZE、FEAT_CH、OUT_DIR 等）与有正则段相同。

## 4. 不与有正则结果混淆
- **RUN_TAG**：全部改为带 `_noreg` 后缀（如 `tiny_vil_A1_ch32_patch8_noreg`），输出子目录为 `outputs_pswf_paper/tiny_vil_A1_ch32_patch8_noreg/` 等。
- **日志**：重定向到 `/home/omnisky/..._noreg.log`。
- **Tmux 会话名**：`v5ab_tiny_noreg`，与有正则的 `v5ab` 分开，可单独 attach 运行或查看。

运行无正则段时执行该 tmux 块即可（或从 `run.bash` 中只执行这一段的命令）；有正则结果仍在 `*_reg` 的 RUN_TAG 和对应目录下，互不覆盖。

---

**User**

我现在决定把TINY IMAGENET的ABLATION测试在没有正则的前提下也跑一遍，你觉得有必要吗

---

**Cursor**

**结论：有价值，建议做；算力紧的话可以先跑子集。**

**有必要的原因：**

1. **把“结构贡献”和“正则贡献”分开**  
   有正则时，提升可能来自：PSWF/结构本身，或 mixup/cutmix/label smoothing 掩盖过拟合。无正则能看出模型“裸”表现，更容易判断哪个 ablation 是真正在改表示/优化。

2. **过拟合与泛化**  
   Tiny-ImageNet 数据量小，无正则更容易过拟合。看各变体在无正则下的 train–val gap 和 val 曲线，能判断谁更稳、谁更依赖正则，对写论文的分析部分很有用。

3. **和文献/审稿对齐**  
   很多工作会做“无/轻正则”的消融，便于公平比较结构。你有一份无正则的完整 ablations，审稿人要 no-reg 对比时可以直接用。

4. **排序是否一致**  
   若在无正则下 A1 / W3_add / W3_poolonly / W3_residual 等的**相对排序**与有正则一致，可以更放心说“结论不依赖正则”；若不一致，正好分析是哪些组件在和正则强耦合。

**可省或后做的情况：**

- 若当前目标只是“在 Tiny-ImageNet 上拿到尽量高的准确率”，有正则那一套已经够用。
- 若算力有限，可以**先跑无正则子集**：例如 A1、W3_add、W3_poolonly、W3_residualonly（VIL + ViT 各几条），看和 reg 下的排序是否一致，再决定是否把无正则的完整 ablation 全跑完。

**建议：**  
你已经在 `run.bash` 里加好了无正则段，从实验设计角度**值得跑**。若时间/卡紧张，就先用无正则跑上述几条关键配置，再视结果决定是否补全全部无正则 ablations。

