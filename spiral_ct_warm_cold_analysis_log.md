# Spiral-CT Warm/Cold 分析实验日志

按 `spiral_ct_warm_cold_analysis_plan.md` §21 的固定格式记录每次实验。
数据/图输出目录：`output/real/ldctc002/spiral/ntrain1000/grad_compare/`。

---

## Experiment A：volume-outside Gaussian 时间序列 + 六方向拆解（plan §7）

日期：2026-08-30。脚本：`experiments/helpers/analyze_outside_growth.py`
（图 `outside_growth.png`、`outside_direction_scatter_yz.png`；表 `outside_growth_table.csv`）
时间点仅 init + eval 步（5k..30k）；事件级（500..4900）待记录型重跑补充。

| step | N_out cold | R_out cold | N_out warm | R_out warm |
|---|---|---|---|---|
| 0 (init) | 207 | 0.41% | 3080 | 6.16% |
| 5000 | 76802 | 15.35% | 50562 | 9.58% |
| 10000 | 77354 | 15.46% | 51119 | 9.69% |
| 30000 | 78053 | 15.60% | 51847 | 9.83% |

（注：N_out 为体积外并集；六方向计数对同时跨多方向的角点高斯会重复计数，
两者不可直接相加。）六方向（step 5000，cold / warm）：x_low 12179/4959，
x_high 208/1098，y_low 11478/7378，y_high 8230/4805，z_low 25131/19179，
z_high 28349/17342。step 5000 之后各方向均基本冻结（30k 只比 5k 多 ~1.3k/1.3k）。

**Observation:**
cold 的 outside 比例在 0→5k 从 0.41% 暴涨到 15.35%（+14.9pp，绝对量 +76.6k），
warm 同期只从 6.16% 涨到 9.58%（+3.4pp，+47.5k）；5k 后两者都冻结。
方向以 z 为主（cold z_low+z_high=53.5k 占 62%；warm 36.5k 占 67%），
但 cold 在 x_low（12.2k vs 4.9k）和 y（19.7k vs 12.2k）也有显著外溢；
x_high 两边都极少（0.2k/1.1k），说明 +x 方向不是泄漏通道。
cold init 几乎全在体积内（207 点全在 z_low），outside 是训练早期产生的。

**Supports:**
cold/warm 差异在 0–5k 快速形成（plan §15 Step 1 的"early emergence"分支）→
进入 migration / densification 机制分析。
z 方向是主要泄漏通道，与 z-edge occupancy deficit（Fact 5）方向一致，
可以继续把 z-outside 增长与 z-edge NG 赤字放在同一机制框架下检查
（cold z-outside 比 warm 多 17k，同时 z-edge 体积内少 ~20%）。

**Rejects:**
"cold 的 outside 高是因为初始化就在体积外"——cold init outside 仅 0.41%（< warm 的 6.6%）。
"主要在 x/y 泄漏、与 z-edge deficit 无关"——x/y 也有显著量（cold 32k），
z-edge deficit 与 x/y leakage 是否同一机制仍有待区分。

**Remaining ambiguity:**
1. 事件级时间分辨：0→5k 内部的增长曲线（哪段最陡、是否集中在 budget 打满前后）未知，
   需要 500..4900 的事件快照。
2. 无法区分 outside 增长来源：迁移（optimizer 推出）vs 出生（densify 直接在体积外创建）。
3. z_high（体积上方）28.3k 远大于 z_low 的 25.1k——是否与扫描 z 范围向 +z 侧的
   不对称（z_shift 上界 -1.0mm 超出体积顶 -2.3mm）有关未确认。

**Next:**
1. 搭建 record_population 基础设施（事件级 xyz 快照 + birth/prune 记录 + world 梯度）；
2. 验证性短跑复现 cold 0→5k 轨迹后，跑 cold/warm 完整 0→30k 记录跑；
3. 用事件日志做 Experiment B（真实位移，plan §17 优先）/ C（梯度方向）/ D（人口记账）。

---

## 记录型重跑基础设施 + 轨迹复现验证（plan §17、§20-1/3）

新配置：`eval.record_population`（每个 densification 事件存
`population_events/step_*.npz`：xyz_pre/xyz_post 快照、world 梯度 xyz_grad_pre、
birth_xyz/birth_type(1=clone,2=split)/birth_parent_xyz、prune_xyz/prune_tag/
prune_reason(1=density,2=bbox,3=screen,4=scale)）、`optim.early_stop_after_steps`。
对齐不变量已验证：事件间无结构变化 → xyz_post(e) 与 xyz_pre(e+1) 逐行同 id；
人口守恒逐事件成立（ΔN = B − P + M 误差 max=0）。

复现验证（0→5k，对比原 run step_005000）：
cold：psnr3d 27.096 vs 27.114、ssim3d 0.6640 vs 0.6647、N=499958 vs 500448、
R_out=15.30% vs 15.35% ✓；warm：psnr3d 27.730 vs 27.725、ssim3d 0.6803 vs 0.6809、
N=502316 vs 527615（差 5%，见下）、R_out=8.65% vs 9.58%。
warm 的 N 差异来自 500k 预算边界事件的一次性超配（原 run 单事件超配到 527k，
复现 run 到 502k；psnr/ssim 一致，轨迹统计等价）。完整 0→30k 记录跑已后台启动。

---

## Experiment B：真实位移（plan §8、§17）

事件间窗口（100 步）幸存高斯位移，id 对齐，cold/warm 0→5k：

**Observation:**
|Δp| 每 100 步窗口：cold mean=0.00188（0.018mm）p99=0.0106；warm mean=0.00179
p99=0.0104——两者几乎相同。Δz 对称（P(Δz>0)=0.498/0.499，mean≈0）。
图：displacement_hists_{cold,warm}_05k.png。

**Supports:**
位移量级与分布 cold/warm 无系统性差异 → "optimizer 把高斯推出体积"的
mass-outward-migration 假说被削弱；outside 差异不是通过幸存者漂移产生的。

**Rejects:**
以迁移为主导通道的 capacity leakage 解释（至少对幸存者均值层面）。

**Remaining ambiguity:**
1. 只统计了幸存者——被 prune 的高斯（split parent）的生前轨迹未计入；
2. 逐 bin 的净迁移（z-accounting 已算，见 D）比全局分布更相关。

**Next:**
结合 C（梯度方向）与 D（出生/剪枝记账）判断 outside 增长的真实来源。

---

## Experiment C：边界梯度方向（plan §9）

边界带（体积面内 0.05 缩放单位 ≈0.5mm）高斯的 world 梯度 g=∂L/∂xyz，
outward 分量 d=−g·n（n 为外法向；d>0 = 优化器在向外推）：

**Observation:**
两模型、所有边界、所有事件：P(d_out>0) 几乎都 ≤ 0.4，多数 ≤ 0.1。
cold z_low：step600 P=0.01，step5000 P=0.40（E[d_out]=−2.4e-6）；
cold z_high：P≤0.06；x_low P=0.18→0.10。warm 各边界 P≤0.35。
图：boundary_gradient_direction_{cold,warm}_05k.png。

**Supports:**
"cold 的梯度方向更容易把高斯推出体积"假说被否定——边界更新方向总体向内/中性。
与 B 的位移对称性互相印证。

**Rejects:**
gradient-direction-outward 机制（至少对边界带内的既有高斯）。

**Remaining ambiguity:**
事件日的梯度是单视角的瞬时梯度（每 100 步采样一次），跨视角平均行为未直接验证；
但 B 的真实位移（跨 100 步累积）是对"实际去向"的更可靠度量，同样不支持外向迁移。

**Next:**
转向 D：outside 增长更可能来自出生侧（densification 直接在体积外创建）。

---

## Experiment D：人口记账（plan §10）

0→5k 逐事件、逐 z-bin 的 ΔN = B − P + M 分解（守恒误差 0）。
图：z_accounting_{cold,warm}_05k.png；事件分辨率 R_out 曲线
nout_fine_curve_{cold,warm}_05k.png。

**Observation:**
1. R_out 事件曲线：cold 2.99%@600 → 10.2%@1600 → 15.4%@5000（单调、0→2k 为主）；
   warm 6.26%@600 → 8.44%@1600 → 8.65%@5000（+2.4pp，基本原地）。
2. 出生是 outside 增长的主通道：cold 0→5k 总出生 462474，其中 83183（18.0%）
   直接出生在体积外；warm 453200 中 38880（8.6%）。
3. 出生构成：cold 体积外出生中 clone 75.7% / split 24.3%（split 子代 20200，
   其 parent 96% 本身已在体积外）；warm 体积外出生中 clone 99.0% / split 1.0%。
   全量 split 次数：cold 22958 子代（11479 次 parent split）vs warm 1766 子代
   ——cold 的 split 触发频率是 warm 的 13 倍。
4. split 子代相对 parent 的偏移（≈parent scale）：cold mean 2.03mm/p90 3.81mm/
   max 9.8mm；warm mean 1.83mm/p90 3.65mm——两模型的 split parent 尺度本身相近，
   差别在 cold 触发 split 的高斯数量多一个量级（scale>1.9mm 的群体大得多）。
5. 剪枝：cold 总 11479 = 全部为 split_parent（密度剪枝≈0）；warm 883。
6. 预算锁定时刻：warm 在 step≈1600 就达到 500k 上限（其后出生≈0）；
   cold 到 step≈4900 才触及上限，0→5k 全程持续出生。
7. 出生 z 分布（下外/8bin/上外）：cold 25070/…/29220（上外 6.3%）；
   warm 12873/…/14687（上外 3.2%）。顶部 bin：cold 46119 vs warm 51818
   ——warm 把更多出生留在顶部 bin 内，cold 更多溢到体积上方。

**Supports:**
1. "densification 直接在体积外创建 Gaussian"是 outside 增长的主通道
   （Q2 的第二个选项），尤其通过 clone 对体积外 parent 的就地翻倍 +
   边界带 split 的 ±2-4mm 散射；
2. 机制链：cold 早期优化在 FOV 边界（重建框切穿物体处）把部分高斯长成大尺度
   （>1.9mm，跨边界覆盖框外物质）→ densify 时触发 split → 子代散射出框 →
   体积外 clone 放大 → 雪球（体积外出生占比 3%→41%）。
   warm 先验拟合已把边界表示好，边界高斯尺度紧凑 → split 罕见 → 体积外出生
   只维持在 ~6-8% → 预算更早、更有效地锁定在体积内。
3. z-edge 体积内赤字（Q3）的来源不是迁移、不是剪枝，而是"出生分配"：
   cold 的边界出生更多落在体积外（上外 29220 vs 14687），
   warm 更多留在框内顶部 bin。

**Rejects:**
1. 全局预算竞争（Experiment E 假说）作为 cold 泄漏主因：cold 的 outside 增长
   发生在预算仍有大量余量的阶段（0→2k，N 才 10 万→44 万），不是被 cap 逼出去的；
2. 迁移主导泄漏（与 B/C 一致）；
3. 剪枝主导（cold 密度剪枝≈0）。

**Remaining ambiguity:**
1. 为什么 cold 会在边界长成大尺度高斯（scale>1.9mm）而 warm 不会——需要
   事件级 scale 记录或渲染诊断（现有数据只能从 split 子代偏移间接推断）；
2. out-of-box 高斯是否真的"必需"（框外物质对投影渲染的贡献）——若必需，
   clamp 干预会伤害 2D 投影损失，需要同时看 2D/3D 指标；
3. 事件日单视角梯度 vs 跨视角平均的差异（见 C）。

**Next:**
Experiment F（position clamp 到重建框，4 组：cold/cold+bound/warm/warm+bound，
各 0→5k）：若 cold+bound 的 R_out 显著下降且 3D 指标不降（或升），则泄漏是因果的；
若 2D 损失明显变差，则说明 out-of-box 表示对投影拟合是必需的，泄漏是"被迫"的。

---

## Experiment F（part 1）：cold + position clamp（plan §12）

新配置 `model.position_clamp_box`：每步 optimizer.step 后把高斯中心 clamp 到
重建体积框 [offOrigin∓sVoxel/2]。cold+bound 0→5k（GPU 2，~6 分钟）。

**Observation:**
1. 指标（step 5000）：psnr2d 30.163 vs cold 33.333（**−3.17 dB**），
   ssim2d 0.941 vs 0.961；psnr3d 26.634 vs 27.096（−0.46 dB），
   ssim3d 0.652 vs 0.664。N=461963（vs 499958）。
2. 出生记录（clamp 前的本欲出生位置）：450315 出生中 71398（15.9%）本欲
   出生在体积外——与未 clamp 的 18.0% 几乎相同。densify 的出生意向没有被
   clamp 改变。
3. clamp 有效：eval xyz 全部在框内/框面上（z 面 pin 住 24621 个，x/y 面同样
   有 pin 但 float32 舍入使严格比较只检出 z 面；用容差判定则 outside≈0）。

**Supports:**
1. out-of-box 高斯对投影拟合是**功能必需**的：不允许它们存在 → 2D PSNR 暴跌
   3.2 dB。重建框把物体截断（框上方/下方有真实物质穿过投影线），模型必须用
   框外质量解释这些投影。因此"capacity leakage"不能简单视为纯浪费——
   至少一部分是边界表示的必然代价。
2. 出生意向由边界区域的梯度/尺度动力学决定，与 clamp 无关（15.9% vs 18.0%）。

**Rejects:**
1. "clamp 一下就能把泄漏预算收回、改善重建"——3D 反而 −0.46 dB（边界堆积的
   pin 高斯是退化表示，且 2D 损失上升把优化拉向坏方向）；
2. 泄漏纯属浪费、可无代价消除的假设。

**Remaining ambiguity:**
1. cold 与 warm 的差别现在更精确地表述为"边界表示的代价不同"：cold 用 15.4%
   的框外预算 + 边界大尺度高斯来满足投影；warm 只用 9.6%。为什么 cold 需要
   更多？——需要 scale/密度随事件的记录（当前只记了 xyz）。
2. warm+bound 的结果尚未出（预计 warm 的 2D 损失会更小，因为它的框外质量少）。

**Next:**
1. warm+bound 跑完对比；
2. 若确认"边界表示"是根因，做针对边界的干预变体（如限制边界区 scale 增长 /
   split 子代框内化），或记录事件级 scale/density 补充机制证据。

---

## Experiment F（part 2）：warm + position clamp

warm+bound 0→5k（GPU 2）。

**Observation:**
step 5000：warm+bound psnr2d 31.101 vs warm 34.048（**−2.95 dB**），
psnr3d 27.171 vs 27.730（−0.56 dB），ssim3d 0.669 vs 0.680；N=504598 vs 502316。
2D 损失幅度与 cold+bound 的 −3.09 dB 基本相当。

**Supports:**
1. 两模型对 out-of-box 质量的**功能需求相同**（clamp 的 2D 代价几乎一样）；
   cold/warm 的差别不在"需不需要框外质量"，而在"用多少预算满足同一需求"
   （cold 15.4% vs warm 9.6%）——cold 的边界表示效率更低。
2. warm 的框外质量是"量少质高"：同样禁掉，warm 少损失一点 2D（−2.95 vs −3.09），
   但 3D 一样掉 ~0.5 dB，说明两模型的边界表示都不是零成本的。

**Rejects:**
"框外质量可无代价消除"（两组都打脸）；"warm 的框外高斯是纯冗余"（warm 也需要）。

**Remaining ambiguity:**
cold 多出来的 ~6pp 框外预算是否边际有用（超出 warm 水平的部分是冗余还是
质量补偿）——需要"限制但不断绝"的干预（如上限 9% 的 clamp 区域放宽）才能区分。

**Next:**
1. 事件级 scale/density 记录跑（cold/warm 0→5k，已启动）——验证"cold 边界
   大尺度"假说并给出尺度演化曲线；
2. 5k-15k 窗口补充（见下）。

---

## Experiment D 补充：5k→15k 窗口（cap 后的纯优化期）

来自已完成的 0→30k 记录跑（144 个事件，600..14900）。

**Observation:**
两模型在 step≈5k 都已达到 500k cap（warm 实为 ~1600 就锁定），5k→14900 的
所有事件出生/剪枝数 = **0**（densify 被 cap 完全关闭），N 精确冻结
（cold 500226、warm 502317）。R_out 仅 +0.16pp 漂移。幸存者位移
|Δp|/100步 mean=0.0010（0-5k 是 0.0019，慢一半）。

**Supports:**
Fact 3（early lock-in）的最强版本：5k 之后不仅空间分布冻结，**人口结构完全
冻结**，后期 25k 步只能做参数微调。warm/cold 的一切差距在 0→5k 就已铸成。

**Rejects:**
后期优化重新组织表示的可能（结构上不可能——没有出生/剪枝通道）。

**Next:**
把 0→5k 的机制（split 散射 + clone 放大）与 scale 演化证据拼齐，然后设计
针对边界表示的干预（限制边界 scale 或 split 区域）。

---

## 边界 scale 动力学（事件级 scale/density 记录跑，cold/warm 0→5k）

新记录：population_events 增加 scale_pre/post、density_pre/post
（脚本分析产物 boundary_scale_dynamics.png）。

**Observation:**
边界带（距任一盒面 <1.9mm）高斯的 mean max-scale 演化：
cold：0.303mm@600 → **0.711mm@5000**（×2.3）；interior 0.277→0.331（×1.2）。
warm：0.504→0.613；interior 0.483→0.522（均温和）。
P(max-scale>1.9mm=split 阈值)：cold 边界带 0%→**4.7%**（interior 0%）；
warm 边界带 1.2%→0.7%（无大尺度尾部形成）。

**Supports:**
机制链的关键一环得到直接证据：cold 在边界带长出一个大尺度子群体（~5%、
>1.9mm、均值 ×2.3），它们正是 split 的触发者（split 需 scale>1.9mm 且
grad≥5e-5）；warm 的边界尺度温和（先验拟合已定好边界表示，无需撑大）。
结合 D 的发现（cold split 是 warm 的 13 倍、体积外出生 18% vs 8.6%、
子代散射 ±2-4mm），完整链条闭合：
cold 均匀弱先验 → 边界区域优化被迫用大尺度高斯跨盒覆盖框外物质 →
densify 时 split 散射出框 + 框外 clone 放大 → 15.4% 预算在框外、
z 边缘框内载体少 ~20% → 500k 预算 5k 锁定 → 最终质量落后。

**Rejects:**
"cold/warm 的 densify 触发差异来自梯度大小"——需要继续排除（梯度阈值相同，
但触发 split 还需要大尺度；大尺度子群体是 cold 特有，与边界投影需求相关）。

**Remaining ambiguity:**
1. 大尺度边界高斯为什么在 cold 形成：是"框外物质的投影需求"（功能性）还是
   "弱初始密度导致优化不稳"（病理性）？clamp 实验（2D −3dB）支持功能性
   解释占主导。
2. warm prior pickle 的 scale 列量纲（log 空间？）存疑，不影响 warm 轨迹复现
   （事件级 scale 自洽）。
3. split 散射是否因果导致质量差距——需要"无 split"干预验证。

**Next:**
max_scale=0.1（=1.9mm=split 阈值）干预：cold/warm 各 0→5k（已启动）。
若 cold_maxscale 的框外出生骤降且 3D 质量持平或上升 → split 散射是因果通道；
warm_maxscale 作对照（预计几乎不变）。

---

## 干预：max_scale=0.1（=1.9mm，split 阈值，禁大尺度）

cold/warm + max_scale 0→5k。

**Observation:**
1. 指标（step 5000）：cold+ms psnr2d 32.997/psnr3d 27.048/ssim3d 0.663
   （vs cold 33.253/27.114/0.665，全面略降）；warm+ms 33.587/27.482/0.676
   （vs warm 34.048/27.730/0.680，也略降）。
2. 人口：cold+ms R_out 15.4%→**12.1%**（框外预算下降），但框外出生占比反而
   升到 20.5%（18.0%），split 子代仍 20120（scale 帽与 split 阈值同值，事件时
   先 split 再剪），另产生 **46390 次 scale 剪枝**（churn）。warm+ms R_out
   9.6%→9.0%，同样略降。

**Supports:**
大尺度高斯对两模型都是功能性的（禁掉 → 双双略降）。cold 的框外人口确实部分
依赖大尺度/split 通道（12.1% vs 15.4%），符合"split 散射是框外出生通道之一"。

**Rejects:**
"split 散射是 cold 落后的因果瓶颈"——机械上减少了框外人口，但质量没有改善
（反而微降）。因此泄漏只是下游症状；真正的约束在 cold 初始化的边界表示能力
本身。warm 不需要大尺度/框外质量去"赢"，但也不能白白移除它们。

**Remaining ambiguity:**
1. cold 与 warm 的差距是否本质是"表示效率"——即同样质量需要的预算不同？
   → 提高预算（cap=1M）可以直接检验：若 cold 拿到更多预算后追平 warm，
   说明是 capacity 约束；若依然落后，说明优化动力学本身有差异。
2. scale 剪枝造成的 46k churn 与质量损失无法完全分开。

**Next:**
Experiment E 变体：cap=1M，cold/warm 各 0→5k（已启动）。

---

## Experiment E 变体：cap=1M（global budget competition 检验）

cold/warm + max_num_gaussians_absolute=1e6，0→5k。

**Observation:**
1. 指标（step 5000）：cold+1M psnr3d 27.102/ssim3d 0.664/psnr2d 33.314
   （vs cold 27.114/0.665/33.253——**完全无改善**）；warm+1M 27.588/0.682/33.688
   （vs warm 27.730/0.680/34.048——也基本不变，2D 略降）。
2. 人口：cold+1M N=499346@5k——**cold 根本没吃下额外预算**（与 500k 时几乎
   相同的出生数 462k，densify 由自身的 grad 触发决定，不由 cap 限制）；
   warm+1M N=660848（+160k，吃下了），R_out 升到 12.9%（多出的预算部分流到
   框外），质量却略降。

**Supports:**
cold 的落后**不是** global capacity competition：预算翻倍后 cold 既不增人、
也不提质。cold 的约束在其自身（初始基底的表示效率 / densify 触发分布），
不在 500k 上限。warm 吃下额外预算但边际收益为负——500k 已接近该算法/数据的
有效表示容量。

**Rejects:**
"提高 N_max 可消除 cold/warm 差距"（plan §11 的判定分支：完全没改善 →
cap 不是主因，不再围绕 cap 展开）。

**Remaining ambiguity:**
无（cap 假说已关闭）。

**Next:**
实验链收束（见"结论汇总"）。可选后续：G/H（关 densify / freeze xyz）作
机制拆分补充；plan §19 的 circular-cone control 是"spiral 特异性"主张的
必要条件，暂不声称 spiral 特有。

---

# 结论汇总（截至 2026-08-30 的实验链）

**核心机制链（全部环节均有直接证据）：**

cold 的 FDK-前景均匀初始化在重建框边界（框切穿物体处）只有弱基底 →
0→5k 早期优化被迫在边界带长成大尺度高斯（mean max-scale ×2.3，5% 超过
1.9mm split 阈值；warm 无此现象）→ densify 事件中这些大高斯触发 split
（cold 是 warm 的 13 倍），子代 ±2-4mm 散射、约半数落在框外，框外 parent
再被 clone 放大（框外出生占 cold 全部出生的 18%，warm 仅 8.6%；出生占比
从 3% 雪球到 41%）→ step≈5k 双方达到 500k 上限、densify 完全关闭、人口
结构永久锁定（5k→30k 出生/剪枝=0）→ cold 最终 15.4% 预算在框外（warm
9.6%）、z 边缘框内载体少 ~20%、总体有效表示容量低 ~13% → 最终 3D PSNR
落后 0.35-0.4 dB、结构更糊。

**被否定/关闭的假说：**
1. z 边缘投影视角缺失（Nv 沿 z 平坦）；
2. 初始 z 边缘载体饥饿（init 均匀）；
3. 梯度幅值饥饿（G/NG 两者接近）；
4. 梯度方向外推（边界更新方向向内/中性；位移对称）；
5. 迁移主导泄漏（实际位移 cold/warm 同分布；泄漏靠出生）；
6. global budget competition（1M cap 无效）；
7. 泄漏是纯浪费、可移除换质量（clamp −3dB 2D；maxscale 也略降）。

**干预实验的教训：** 泄漏与大尺度是 cold 在弱初始基底下满足投影需求的最优
应对（功能性），不能廉价修复；warm 的优势来自 prior-fit 提供的边界表示本身。

**回答 plan §6 的四个问题：**
Q1（何时）：0→5k，单调，0→2k 为主。Q2（为何出去）：出生直接生在框外
（split 散射 + clone 放大），非迁移、非剪枝。Q3（z 边缘赤字来源）：出生分配
——cold 的边界出生更多落在框外（上外 29220 vs warm 14687），框内顶部 bin
反而更少。Q4（是否因果）：泄漏是功能性下游症状；cold 的表示效率差异才是
上游约束，已通过 clamp/maxscale/1M-cap 三个干预排除简单修复路径。

**下一步（按 plan §19-§20）：**
1. circular-cone control（相同 init/cap/densify、匹配投影数）检验
   "spiral 几何是否放大该现象"——目前只敢说"当前 spiral 数据 + R2Gaussian
   风格优化下出现"；
2. 可选 G/H 机制拆分（关 densify / freeze xyz）补充早期动力学的最后拼图；
3. 若之后要设计算法，方向应是"边界感知的初始化/先验"或"split 子代位置
   约束"，而非 visibility 加权或 z 均衡采样（证据不支持后者针对根因）。

---

# 算法修复路线与下一轮实验计划（2026-08-30 晚）

## 候选路线（按机制环节分类，详见与用户的讨论）

1. 打"功能需求"：
   - 1a ROI 投影掩码：只保留与重建框相交的射线的像素参与损失，消灭框外
     质量需求（最彻底，损失目标改为"只拟合 ROI"）；
   - 1b 扩展重建框 + 评估裁剪：把框外物质变成扩大域的合法物质。
2. 打"出生放大"（cold 15.4% vs warm 9.6% 的超配部分）：
   - 2a 框外预算上限：框外高斯数 ≥ 目标比例（如 10%）后，densify 选择
     排除框外高斯；
   - 2b 只 clamp 出生位置：split/clone 子代出生时 clamp 回框内，优化器
     仍可自由移动（区别于已测的全位置 clamp F）；
   - 2c 边界感知 split（子代偏移按到边界距离缩放）。
3. 打"边界大尺度"：3a 边界种子层（用扩展 FDK 在框外薄层预置高斯）；
   3b 框外位置软惩罚（λ·Σ dist_outside²）。
4. 锚定最终指标：4a 训练中加框内随机子体积 3D L1 监督。

## 本轮实验（并行）

- E1: cold + 2a（optim.outside_gaussian_budget=0.10，0→5k，记录跑）
- E2: cold + 2b（model.clamp_child_positions=true，0→5k，记录跑）
- E3: warm + 2b（对照组，0→5k，记录跑）
- E4（离线预计算）：1a 的 ROI 投影掩码（每视角每像素"射线是否与框相交"，
  1000 视角，736×64）——为下一轮损失掩码实验备好
判定（对照基线 cold 15.4%/warm 9.6% 框外、psnr3d 27.11/27.73@5k）：
- 2a/2b 应把框外占比压到 ~10% 而 2D 损失基本不涨（区别于 clamp 的 −3dB）；
  若 3D 或边缘质量受损，则结合 3a 边界种子层再测。

---

## E4 结果：ROI 投影掩码对当前几何是空集 → 路线 1a 作废

**Observation:**
对全部 1000 个训练视角、全部 736×64 像素计算"射线是否与重建框相交"：
hit_frac = **1.000**（每个视角每个像素的射线都穿过框）。
原因：重建框 z 跨度 14mm 远大于探测器 v 方向在等中心的板宽（±0.96mm），
任何射线的 z 参数区间 (0, ~20mm] 都会与框的相机系 z 区间相交；侧向
（u 最大 0.466 rad）也在框角范围内。掩码文件已存
（roi_proj_mask.npz，全 1，47MB），但无损失掩码价值。

**Supports:**
无（负结果）。

**Rejects:**
路线 1a（"只保留与框相交射线的像素"）：不存在与框不相交的像素——框外物质
对投影的贡献与框内贡献纠缠在**每一条**射线的线积分里，像素掩码无法分离。
框外质量需求来自"射线在框外段穿过的物质"，不是来自"某些只看到框外的像素"。

**Remaining ambiguity:**
无（1a 对该几何不成立，关闭该路线）。

**Next:**
需求侧改走 1b（扩展重建框覆盖整个扫描区域 + 评估时裁剪）；放大侧 2a/2b
（E1/E2/E3 运行中）。

---

## 干预 E1/E2/E3 运行状态

- E1 cold+2a：首跑因 split 的 exclude mask 未随 clone 扩容对齐而崩
  （已修复：exclude pad 到当前 N），已重跑；
- E2 cold+2b、E3 warm+2b：运行中（Child-position clamp 已生效）。
- 用户要求：每次实验保存重建切片图 → 已写 save_recon_slices.py 并对全部
  既有实验模型生成 recon_slices_{model}_{step}.png（pred + |pred−GT| 三平面）。

---

## E1/E2/E3 结果：2a 框外预算 vs 2b 出生 clamp（step 5000）

指标与人口（对照基线）：

| run | psnr2d | psnr3d | ssim3d | N | R_out |
|---|---|---|---|---|---|
| cold | 33.253 | 27.114 | 0.665 | 499958 | 15.30% |
| **cold+2a(预算10%)** | **33.456** | 27.110 | 0.664 | 468543 | **10.02%** |
| cold+2b(出生clamp) | 31.309 | 26.908 | 0.660 | 470333 | 12.91% |
| warm | 34.048 | 27.730 | 0.680 | 502316 | 8.65% |
| warm+2b | 33.502 | 27.652 | 0.680 | 510940 | 8.39% |

**Observation:**
1. 2a：框外占比 15.3%→10.0%（出生框外率 18.0%→10.4%），总高斯 −31k；
   3D 完全不变（27.110 vs 27.114），2D 还略升 +0.2 dB。体积内 z 分布与
   baseline 逐片一致（±1%）——被砍掉的 ~5pp 框外预算**没有**回流到框内，
   是纯冗余。
2. 2b：cold 框外只降到 12.9%（出生通道被掐但已有框外 parent 保留、优化器
   仍可迁出），且 **2D −1.9 dB、3D −0.21**，框内上下边缘片反而更少
   （45.7k→38.6k、46.8k→42.1k）。warm+2b 同样受损（2D −0.55、3D −0.08，
   边缘片 54.9k→46.8k）。

**Supports:**
1. 2a 是干净的算法修复：cold 超出 warm 水平的 ~5pp 框外预算纯属冗余
   （可无代价移除），框外预算上限 10% 即达标，且不损害任何指标；
2. 出生通道是框外质量的**功能主通道**（2b 掐出生 → 2D 受损，与 F 的 clamp
   一致但更温和），不能简单禁用；
3. z-edge 框内赤字与框外超配**相互独立**：2a 把框外砍到 10% 后框内 z
   分布纹丝不动——edge 赤字另有来源（cold 在框内边缘的出生本身就少，
   见 Experiment D 出生 z 分布），不是"框外抢了预算"。

**Rejects:**
"框外超配是 z-edge 赤字的原因"；"出生 clamp 是可行修复"。

**Remaining ambiguity:**
cold 框内边缘出生为什么少（边缘带梯度/尺度动力学的哪个环节抑制了 densify
触发）——warm 边缘每高斯梯度略高（G/NG 数据），但触发还需要 scale/结构条件。

**Next:**
1. 2a 作为候选算法修复记入结论（并可测 2a 在 30k 全程的效果 + warm 对照）；
2. z-edge 框内赤字单独追查：比较 cold/warm 边缘带"满足 densify 触发条件的
   高斯数"随事件演化（已有事件级 grad[记录]/scale 数据可算）；
3. 需求侧改测 1b（扩展重建框覆盖扫描区域 + 评估裁剪）——1a 已证无意义。

---

## z-edge 框内赤字来源分析（触发率 + 出生分区，offline，基于既有事件/评估数据）

**Observation:**
1. 出生分区（0→5k，4 类）：cold in-edge 76744 / in-interior 302547 /
   out-z 54290 / out-xy 28893（in-edge 占 16.6%，out-z 占 11.7%）；
   warm in-edge **91774** / in-interior 322546 / out-z 27560 / out-xy 11320
   （in-edge 占 20.3%，out-z 占 6.1%）。warm 的框内边缘出生多 15k、
   out-z 少 27k，且逐事件前段更高更持续（22k/23k vs cold 的衰减型 15k→6.7k）。
2. densify 触发率（step5000，grads=accum/denom ≥ 5e-5）：
   cold 边缘带 **0.06%** / 内部 0.03%；warm 边缘带 **0.99%** / 内部 0.53%
   ——warm 边缘带的触发率是 cold 的 ~16 倍（accum/denom 是均值，窗口长度
   不影响可比性；但 warm 自 1600 起因 cap 停 densify、无 reset，需注明）。
3. 边缘带大尺度高斯（scale>1.9mm）的梯度：cold 大尺度 grad_norm 5.7e-6 ≈
   小尺度 4.4e-6（大高斯不在结构上）；warm 大尺度 **3.17e-5** ≫ 小尺度
   6.1e-6（大高斯在真实结构上被强拉）。

**Supports:**
z-edge 框内赤字的完整机制（回答 plan Q3）：
cold 的均匀初始化把边缘高斯放在低内容/低梯度区 → 边缘触发率只有 warm 的
1/16 → 框内边缘出生少 15k；且 cold 边缘触发的高斯是大尺度（且不在结构上）
→ 走 split 通道、子代散射出框（out-z 出生 54k vs warm 27.6k）。
warm 的先验拟合把边缘高斯放在真实结构上（大尺度者梯度 5 倍于小尺度）→
高触发率、clone 就地复制 → 框内边缘出生多、out-z 少。

**Rejects:**
边缘赤字与触发阈值本身无关（同一 5e-5）；与全局预算无关（2a 已证）。

**Remaining ambiguity:**
warm 自 1600 停 densify 后其 5k 触发率是否系统性偏高（无 reset 的窗口
差异）——可用 600..1600 事件窗口内的 compact 决策梯度记录做补充校准。

**Next:**
等 2a-30k 全程验证（cold/warm 并行运行中）；完成后汇总本轮结论。

---

## E1 全程验证：2a 框外预算 10%（0→30k，cold + warm 对照）

**Observation（step 30000）:**

| run | psnr3d | ssim3d | psnr2d | N | R_out | 框内高斯 |
|---|---|---|---|---|---|---|
| cold 基线 | 29.365 | 0.7146 | 36.79 | 500458 | 15.60% | 422.4k |
| **cold+2a** | **29.365** | **0.715** | **36.93** | 500019 | **10.48%** | **447.6k** |
| warm 基线 | 29.722 | 0.7247 | 37.05 | 527615 | 9.83% | 475.8k |
| warm+2a | 29.732 | 0.724 | 37.04 | 500257 | 8.94% | 455.5k |

**Supports:**
1. 2a 在 30k 全程是**纯赢**的算法修复：cold 的 3D 质量与基线完全一致
   （29.365 vs 29.365）、2D 略升 +0.15 dB，同时框外预算 15.6%→10.5%、
   框内有效容量 +25k。被砍掉的 ~5pp 框外质量全程无功能贡献（纯冗余）。
2. warm+2a 对照干净：预算不触发时干预无副作用（29.732 vs 29.722）。
3. 与 cap1m 结论自洽：cold+2a 多出 25k 框内容量但 3D 不变——容量不是
   cold/warm 差距的约束（表示/优化动力学才是）。

**Rejects:**
"框外预算对后期优化有不可替代作用"——30k 全程无损失。

**Remaining ambiguity:**
无（2a 已验证完毕，作为候选修复收进结论）。

**Next:**
1. 算法修复候选定案：`optim.outside_gaussian_budget`（框外预算上限）+ 完整
   机制链已可成文；
2. 若追求进一步逼近 warm：z-edge 框内赤字仍需初始化侧修复（3a 边界种子层
   或 prior-fit 边界补丁）；1b 扩展框为可选的大改路线（需数据侧重建扩展 GT）。

重建切片（pred + |pred−GT| 三平面）：`recon_slices_{实验名}_step{步}.png`
- base_cold / base_warm（原始基线，5k+30k）
- E1_cold_2a_budget10、E2_cold_2b_childclamp、E3_warm_2b_childclamp
- F_cold_boxclamp、F_warm_boxclamp
- maxscale_cold、maxscale_warm、cap1m_cold、cap1m_warm（均为 5k）

事件分析图（实验字母 + 机制）：
- A：`A_outside_growth.png`、`A_outside_direction_scatter_yz.png`、
  `outside_growth_table.csv`、`factors_*.png`（Nv/NG/梯度因子）、
  `ng_time_series_yz.png`、`grad_per_gaussian_time_series_yz.png`
- B：`B_displacement_hists_{cold,warm}_05k.png`
- C：`C_boundary_grad_direction_{cold,warm}_05k.png`、
  `boundary_scale_dynamics.png`
- D：`D_nout_fine_curve_{cold,warm}_05k.png`、`D_z_accounting_{cold,warm}_05k.png`
- E1/E2/E3：`{nout_fine_curve,displacement_hists,boundary_gradient_direction,z_accounting}_{E1_2a_budget10,E2_2b_childclamp,E3_2b_childclamp}.png`
  与对应 `recon_slices_E*`

---

## Improved-GS 移植实验：G0–G3（Growth Control / LAS / EAS，0→30k，improved-densification 分支）

**背景**：移植论文 "Improving Densification in 3D Gaussian Splatting for High-Fidelity
Rendering"（CVPR 2026 Findings，arXiv:2508.12313，官方实现 XiaoBin2001/Improved-GS）。
组件：Growth Control（√progress 预算爬坡，budget=总人口上限，去 clone）、LAS（长轴分裂，
子代 ±3·0.45·σ_long 确定性放置，尺度 ×0.55/×0.893、密度 ×0.6）、EAS（边缘感知得分；
**代理版**，见下）。与冷启动机制链的对应：去 clone 关闭框外 clone 放大通道、LAS 替代
随机三维散射（±2-4mm）、EAS qualify 补边缘带低梯度缺口。模型：
`factgs_det_native_cold_rec30`（G0 legacy 基线）、`factgs_det_native_cold_G{1,2,3}_rec`。
框外口径：scene_scale=2/19 缩放重建 box（与 2a 实验一致；G0 的 17.9% 与既有 18% 吻合）。

**Observation（step 30000）:**

| run | psnr3d | ssim3d | psnr2d | N | 出生构成 | 框外出生 | z 边缘带人口 |
|---|---|---|---|---|---|---|---|
| G0 legacy | 29.365 | 0.7146 | 36.79 | 500226 | 95% clone | 17.9% | 10.9% |
| G1 growthctl（无 LAS） | 29.35 | 0.701 | 35.91 | **51.6k** | 全 split | 89.3% | 15.5% |
| **G2 +LAS** | **30.15** | **0.723** | **37.89** | 500000 | 全 split | 25.0% | 14.0% |
| G3 +EAS qualify | 30.02 | 0.720 | 37.52 | 500000 | 全 split | **11.1%** | 10.5% |

**Supports:**
1. **G2（LAS）全面最优**：psnr3d +0.78 dB、psnr2d +1.10 dB vs G0。LAS 直接修复冷启动
   机制链的第 2 环（split 随机散射）：确定性长轴分裂、无尺度资格（小高斯也有分裂入口）。
2. **G3（EAS qualify）泄漏最小**：框外出生 11.1%（比 G0 −38%）、z 边缘出生最少；质量仍
   高于 G0（+0.65 dB），代价 vs G2 −0.13 dB——EAS 把预算导向投影边缘、牺牲螺旋 z 端部
   低覆盖区表示。
3. **G1 揭示机制根源**：去 clone 后只有 >1.9mm 大尺度高斯可 split（它们集中在边界），
   随机 split 子代散射使框外出生占比飙到 89.3%——原版 split 散射通道在冷启动被放大。
   但 G1 仅 51.6k 人口追平 G0 的 3D 质量 → legacy 500k 人口大部分是低效冗余，去 clone +
   预算制本身无质量损失。
4. Growth Control 行为正确：√progress 爬坡、500k 上限精确锁定（G1 因候选不足未触顶）。

**Remaining ambiguity:**
1. G3 vs G2 的 −0.13 dB 是否可由超参回收（qualify 分位、sample_cams）——超参实验进行中。
2. EAS 为**代理版**：∂ℓ/∂density（ℓ=Σ w·I，w=GT 投影拉普拉斯边缘图）近似官方内核版
   accum_weights（Σ w·T·α）；误差来自颜色/softplus 导数调制，且需额外前向+反向
   （+10 视角/事件 ≈ +5% 计算）。彻底版需改 gs_ct_rasterizer 子模块内核。

**Next:**
1. G3 超参变体（eas_qualify_quantile 0.90/0.99、G2+2a 组合，运行中）；
2. 若需彻底 EAS：评估 fork 子模块内核的成本与收益。

---

## Improved-GS 超参变体：G3 分位扫描 + G4（LAS+2a 组合，0→30k）

**Observation（step 30000）:**

| run | psnr3d | ssim3d | psnr2d | 框外出生 | z 边缘带人口 |
|---|---|---|---|---|---|
| G2 +LAS | 30.15 | 0.723 | 37.89 | 25.0% | 14.0% |
| G3 q90 | 30.02 | 0.721 | 37.51 | 11.1% | 10.6% |
| G3 q95（原 G3） | 30.02 | 0.720 | 37.52 | 11.1% | 10.5% |
| G3 q99 | 30.03 | 0.721 | 37.53 | 11.7% | 10.7% |
| **G4 LAS+2a** | **30.18** | **0.724** | **37.91** | **10.1%** | **15.4%** |

模型：`factgs_det_native_cold_G3_q{90,99}_rec`、`factgs_det_native_cold_G4_las2a_rec`
（G4 = improved + LAS + `optim.outside_gaussian_budget=0.1`）。

**Supports:**
1. **G4 是全场最优**：psnr3d +0.81 dB vs G0、框外出生 10.1%（比 G2 −60%、比 G0 −44%）、
   z 边缘人口 15.4%（全场最高）。2a 与 LAS 正交叠加：2a 管"选择"（候选排除框外）、
   LAS 管"放置"（确定性长轴分裂）——机制链第 2 环（散射）与第 3 环（选择放大）同时修复。
2. **qualify 分位 [0.90, 0.99] 完全无敏感**（30.02/30.02/30.03，框外 11.1/11.1/11.7%）：
   预算（√爬坡到 500k）在大部分窗口超过候选数 → 资格带内候选几乎全被选中，分位只改
   资格带宽度、不改选中结果。G3 相对 G2 的 −0.13 dB 是 EAS 重排出生分布的本质代价，
   非超参可回收。
3. G3 的 z 边缘人口（10.5%）全场最低而 G4（15.4%）最高、质量也最高——螺旋 z 端部
   人口是**有效**覆盖（z_shift 端部物质投影稀疏但必需）；EAS"导向投影可见边缘"策略在
   此数据集方向上不利（投影边缘集中在中心视野，与 z 端部稀缺覆盖竞争预算）。

**Rejects:**
"EAS 边缘资格带宽度是有效旋钮"——此数据集上分位无关；代理版 EAS 的边际价值已被
零开销的 2a 取代（G4 同时拿到更低泄漏与更高质量）。

**Remaining ambiguity:**
EAS 的 z 端部竞争是代理版伪影还是官方版固有（官方 accum_weights 也加权投影边缘，
同样会偏中心）——若日后研究可改 rank 模式或内核版复核；当前无实际必要性。

**Next:**
定案推荐配置：`optim.densification_method=improved optim.use_las=true
optim.outside_gaussian_budget=0.1`（G4）。内核版 EAS 不推进。

---

## warm（FDK prior）启动验证：G2 / G4（0→30k）

**背景**：G4 定案前需在 warm 侧确认无副作用（对照 E1 的 warm+2a 验证思路）。
warm 基线 = `factgs_det_native_warm_fdk1000`（init_mode=prior，
vol_fit_fdk1000/step_500 点云，29.7225/0.7247/37.03，N=527615，R_out=9.83%）。
模型：`factgs_det_native_warm_fdk1000_G{2,4}_rec`。

**Observation（step 30000）:**

| run | psnr3d | ssim3d | psnr2d | N | 框外出生 | 最终 R_out |
|---|---|---|---|---|---|---|
| warm 基线 | 29.722 | 0.7247 | 37.03 | 527615 | — | 9.83% |
| warm G2（LAS） | 30.283 | 0.723 | 37.96 | 500000 | 23.2% | **21.87%** |
| **warm G4（LAS+2a）** | **30.289** | **0.723** | **38.01** | 500000 | 9.9% | 9.98% |

**Supports:**
1. **G4 在 warm 下同样定案**：质量 +0.57 dB 3D / +0.97 dB 2D，框外出生 9.9%、最终
   R_out 9.98% 与基线（9.83%）持平——2a 在 warm 无副作用（与 E1 结论一致），G4 是
   cold/warm 通吃的推荐配置。
2. **warm G2 揭示 LAS 的框外语义与 legacy clone 不同**：框外占比翻倍（21.9%）但质量
   反升 +0.56 dB——LAS 子代沿长轴确定性放置、无随机散射，框外人口是受控且功能性的
   （呼应既有结论"out-of-box 表示对投影拟合是必需"）；legacy 的框外冗余才是浪费。
3. LAS 增益不限于冷启动（warm 也 +0.56 dB）——确定性长轴分裂普遍优于随机散射 split。

**Rejects:**
"LAS 会加剧框外泄漏而伤质量"——warm 下框外翻倍但质量上升；泄漏定性取决于放置
确定性而非框外数量。

**Remaining ambiguity:**
warm_G2 的 R_out 21.9% 若叠加 2a 会掉回 10%（G4 已证）——2a 上限 10% 是否也会砍掉
"功能性框外"部分（G4 质量 30.289 > G2 30.283 说明未砍掉或砍掉无碍）。

**Next:**
Improved-GS 移植全矩阵定案（cold+warm）。收尾：提交分支或按需清理实验产物。


## 螺旋 z-θ 耦合量化分析：per-z 覆盖度（2026-09-01）

**问题**：螺旋扫描中投影的 z 与旋转角 θ 严格耦合（z_shift vs unwrap(θ) 相关系数 1.0000，
dz/dθ = 0.274 mm/rad，全程一致；注意 samples_per_rotation=1152 为原始采样率，本数据集实际
降采样到每圈 ~104 投影，相邻角步长 0.011–0.033 rad 不均匀）。相比 circular 的"每层全角度
覆盖"，想量化螺旋每个 z 层的有效覆盖，判断"coverage deficit"是否存在、能否据此改进。

**方法**（纯离线，用 meta_data.json 几何；场景坐标 ss=2/max(sVoxel)=0.1053）：
对 100 个等距 z 层（box z∈[−1.713, −0.240]），对每个投影求其射线族（u 方向逐列 + v 方向
连续行带）与该层 xy∈[−1,1]² 的交叠，统计 ①被至少一条射线照到的投影数、②xy 覆盖面积、
③覆盖投影的角跨度（递减 unwrap 后）。探测器位于源对侧（DO−DD，初版曾错放同侧导致全零）。

**结果（每层，N=1000 投影）：**

| z 位置 | 覆盖投影数 | xy 面积覆盖 | 角跨度 |
|---|---|---|---|
| 端部 10% (z≈−1.7~−1.6) | 210 | 90.7% | ~705° |
| 中心 10% (z≈−0.98) | 213 | 90.2% | ~729° |
| **端部:中心** | **0.99** | **1.01** | **0.97** |

**关键发现：**
1. **没有 z 向覆盖 deficit**：每层被 ~210 个投影（21%）照射，xy 覆盖 ~90%，端部:中心 ≈ 1.0
   三个指标全部。此前"z-end 人口是 effective coverage"的推断不成立——G3 的 −0.13 dB
   不是端部缺覆盖所致，而是 EAS 把出生从 z-end 挪走后的本质代价（z-end 有独立质量贡献，
   但机制不是几何覆盖）。
2. **每层角跨度 ~730° > 360°**：螺旋的 v 行带（探测器 z 向 3.5mm 全高）使每个 z 层被
   超过一整圈的角度照射——层间信息冗余而非缺口。circular 每层恰好 360°，螺旋反而更足。
3. **真正的信息差在"确定性可预测"而非"更多覆盖"**：每个投影的 z_shift 精确界定其视野
   z 窗口（[z_s−sV/2, z_s+sV/2]，±1.75mm），这是 circular 没有的显式先验——同一点只在
   一组可精确枚举的投影中可见，可用于覆盖加权、初始化、或正则。

**Implications for spiral 方向：**
- 方向 A（coverage-aware training）动机修正：不是补 deficit，而是利用**已知的 per-projection
  z 视野**做显式先验（如按可见投影数加权 quality/gradient，或 z 层感知的 density 预算分配）。
- 每层 21% 投影覆盖、~90% 面积 → 螺旋信息量确实低于 circular（每层有效角度冗余 2 圈但
  投影数 1/5），但端部与中心完全一致 → 若再出现 z 端部质量问题，归因不应是覆盖。
- 数据存于 /tmp/z_coverage.npy（zgrid, area, n_proj）。

### 图（2026-09-01 追加，英文标签版）

存于 `analysis/z_coverage/`：
- `nvis_spiral_xy.png` — spiral 每体素可见投影数 xy 热图（端部/1/4/中心/3/4 四个 z 层）
- `nvis_circular_xy.png` — 对照：同一窄探测器 circular（z_shift 固定）——端部层全 0，中心层全 1000
- `nvis_three_compare_xz.png` — 三合一（x=0 平面 y-z 剖面）：spiral / 窄探测器 circular / 理想宽探测器 circular
- `nvis_maps.npz` — 原始数据（zgrid, nvis_sp, nvis_ci, nvis_ideal）
- `z_coverage.npy` — 前期 per-z 区域覆盖数据（zgrid, area, n_proj）

补充修正：此前"xz 剖面 (y=0)"标注有误——实际取的是 x=0 平面（y×z），英文版图已改为 "plane x=0: y vs z"。每层 n_visible 梯度主要在 y 方向（端部层 81~143，1.8×；中心层 83~142，1.7×），x 方向近乎均匀。
