# Spiral-CT Gaussian Warm/Cold Initialization 差异分析与后续实验任务

## 0. 总目标

当前任务不是简单比较 warm start 和 cold start 谁更好，而是解释：

> **为什么在相同投影数据、相同训练算法和相同 Gaussian 数量上限下，warm initialization 最终获得更高的重建质量、更锐利的结构，而 cold initialization 明显更差？**

重点分析：

\[
\text{Initialization}
\rightarrow
\text{early optimization dynamics}
\rightarrow
\text{densification / movement / pruning}
\rightarrow
\text{Gaussian spatial allocation}
\rightarrow
\text{final reconstruction quality}
\]

尤其关注：

\[
\boxed{\text{0--5000 step}}
\]

因为现有实验表明 Gaussian 空间分布的主要差异几乎全部在这一阶段形成。

---

# 1. 实验背景

## 1.1 Cold initialization

cold 使用约 50k Gaussian 初始化。

经过修正后确认：

- 并不是按照 FDK gradient magnitude 直接造成 z 边缘缺点；
- 初始化 Gaussian 在 reconstruction volume 的 z 方向基本均匀；
- 每个 z 区间大约 5k–7k Gaussian；
- z 上下边缘并不存在明显初始 occupancy starvation。

因此之前的：

\[
\text{cold init edge sparse}
\]

假说已经被否定。

---

## 1.2 Warm initialization

warm initialization 来自 prior volume 的 Gaussian volume fitting。

初始 Gaussian 数同样约 50k。

虽然 warm 的位置、scale、density 等参数质量更高，但从单纯的 z Gaussian count 来看：

\[
N_G^{warm}(z,0)
\approx
N_G^{cold}(z,0)
\]

至少不存在数量级上的 z coverage 差异。

---

# 2. 必须废弃的旧结论

之前 gradient map 使用：

```text
histogramdd
```

但没有固定 histogram range。

由于训练后存在：

- cold：约 14.8% Gaussian 漂移出 volume；
- warm：约 9% Gaussian 漂移出 volume；

这些 outlier 把直方图范围撑大。

结果 reconstruction volume 只占 histogram 中央一部分，导致旧图看起来：

> cold 的 z 上下边缘没有 gradient。

实际上旧图的所谓“上下边缘”有相当部分已经是 volume 外区域。

修正方式：

- histogram range 固定为 reconstruction volume；
- volume 外 Gaussian 单独统计；
- 所有 occupancy / gradient 图重新生成。

因此以下旧结论全部作废：

1. cold z 边缘从初始化开始 Gaussian 数严重不足；
2. cold z 边缘单 Gaussian gradient 比 warm 小 4–11 倍；
3. cold 出现严重 edge gradient starvation；
4. warm 是因为初始 edge Gaussian 更多才建立正反馈。

这些不能继续作为后续推理基础。

---

# 3. 当前已经可靠确认的事实

这是后续所有实验的基础。

## Fact 1：z 方向投影视角覆盖基本均匀

逐 voxel 几何可见视角数：

\[
N_v(x)
\]

沿 z 大约：

\[
112\sim120
\]

波动仅约：

\[
7\%
\]

并不存在明显上下边缘骤降。

因此当前数据下：

\[
\boxed{
\text{z-edge reconstruction difference}
\not\approx
\text{z view-count deficiency}
}
\]

至少不能简单归因于螺旋扫描上下边缘少视角。

---

## Fact 2：cold/warm 初始化的 z Gaussian 数基本一致

init 约 50k Gaussian。

z 分成 8 个 bin 后：

\[
N_G(z,0)
\approx5000\sim7000
\]

cold / warm 均如此。

因此：

\[
\boxed{
\text{initial Gaussian quantity starvation}
}
\]

不是主要原因。

---

## Fact 3：Gaussian 数量变化几乎全部发生在 0–5000 step

两个模型：

\[
50k
\rightarrow
\sim500k
\]

主要发生在 early densification stage。

到 step 5000 后，各个 z bin 的 Gaussian 数基本已经确定。

例如：

\[
N_G(z,5000)
\approx
N_G(z,15000)
\approx
N_G(z,30000)
\]

因此存在明显：

\[
\boxed{
\text{early densification lock-in}
}
\]

即：

> early stage 决定 Gaussian capacity 如何分布，后期训练几乎不会再重新分配这种空间容量。

这是目前最重要的实验观察之一。

---

## Fact 4：warm 最终 volume 内 Gaussian 更多

step 30000：

cold：

\[
N_{\text{inside}}\approx426k
\]

warm：

\[
N_{\text{inside}}\approx480k
\]

volume 外：

cold：

\[
N_{\text{outside}}\approx74k
\]

warm：

\[
N_{\text{outside}}\approx48k
\]

即：

\[
N_{\text{outside}}^{cold}
>
N_{\text{outside}}^{warm}
\]

cold 大约多出：

\[
26k
\]

volume-outside Gaussian。

因此 cold 存在更明显的：

\[
\boxed{\text{effective capacity leakage}}
\]

虽然总 Gaussian budget 都约为：

\[
500k
\]

但真正位于 reconstruction volume 内的有效表示容量存在约：

\[
13\%
\]

差异。

---

## Fact 5：z 边缘 Gaussian deficit 是训练产生的，而不是初始化决定的

初始化时：

\[
N_G^{cold}(z_{\rm edge})
\approx
N_G^{warm}(z_{\rm edge})
\]

甚至部分位置 cold 更多。

但 step 5000 后：

\[
N_G^{cold}(z_{\rm edge})
<
N_G^{warm}(z_{\rm edge})
\]

差约：

\[
18\%-24\%
\]

而且之后一直维持到 step 30000。

说明：

\[
\boxed{
\text{edge deficit emerges during early training}
}
\]

而不是：

\[
\text{edge deficit inherited from initialization count}
\]

---

## Fact 6：单 Gaussian gradient norm 没有明显崩溃

修正后的：

\[
\frac{G(x)}{N_G(x)}
\]

显示 cold / warm 每 Gaussian 平均位置梯度在各 z 区域比较接近。

约：

\[
1.0\times10^{-5}
\sim
1.2\times10^{-5}
\]

warm 通常只高：

\[
10\%-15\%
\]

左右。

因此不存在此前猜测的：

\[
\bar g^{cold}_{edge}
\ll
\bar g^{warm}_{edge}
\]

数量级差异。

所以：

\[
\boxed{
\text{gradient magnitude starvation}
}
\]

目前不是主要解释。

---

# 4. 当前最合理的核心假说

基于以上事实，目前的问题应重新定义为：

> **warm/cold 的差异主要不是由“有没有梯度”决定，而更可能由 early-stage Gaussian population dynamics 决定。**

即：

\[
\boxed{
\text{Initialization quality}
\rightarrow
\text{early position optimization}
\rightarrow
\text{densification / migration / pruning}
\rightarrow
\text{spatial capacity allocation}
\rightarrow
\text{lock-in}
}
\]

cold 可能在 early stage 出现：

\[
\text{more migration outside volume}
\]

或者：

\[
\text{different densification allocation}
\]

或者：

\[
\text{different pruning}
\]

或者三者共同作用。

最终造成：

\[
N_{\text{inside}}^{cold}
<
N_{\text{inside}}^{warm}
\]

并进一步造成：

\[
\text{effective representation capacity}^{cold}
<
\text{warm}
\]

最终表现为更差的高频细节和更低 PSNR / SSIM。

---

# 5. 核心问题已经从 gradient magnitude 转向 population dynamics

后续分析不要只看：

\[
\left\|
\frac{\partial L}{\partial xyz}
\right\|
\]

因为它只告诉我们：

> Gaussian 受到多强的推动。

现在更需要回答：

> Gaussian 被推动到哪里？

即需要分析：

\[
\frac{\partial L}{\partial x},
\quad
\frac{\partial L}{\partial y},
\quad
\frac{\partial L}{\partial z}
\]

的方向。

尤其是：

\[
g_z
\]

在上下边界处是否存在 outward bias。

同时需要分解 Gaussian 数量变化：

\[
\boxed{
N(t+\Delta t)
=
N(t)
+
Birth
-
Prune
+
Migration_{in}
-
Migration_{out}
}
\]

这将成为后续实验主框架。

---

# 6. 后续实验总体目标

后续实验需要回答四个问题。

### Q1

cold 多出来的 volume-outside Gaussian：

\[
74k-48k\approx26k
\]

是什么时候产生的？

### Q2

它们为什么跑出去？

是 gradient direction 问题，还是 densification 直接在 volume 外创建？

### Q3

cold z 边缘 Gaussian 少约 20%，到底是：

- 少 densify；
- 更多移出去；
- 更多被 prune；
- 还是被 global Gaussian budget 抢占？

### Q4

这种 early spatial allocation difference 是否真正导致 warm reconstruction 更好？

最终必须通过 intervention experiment 建立因果关系，而不仅仅是相关性。

---

# 7. Experiment A：Volume-outside Gaussian 时间序列

## 目的

确认：

\[
N_{\text{out}}^{cold}
>
N_{\text{out}}^{warm}
\]

究竟在什么时候形成。

## 统计

推荐 step：

\[
0,
500,
1000,
1500,
2000,
3000,
4000,
5000,
10000,
30000
\]

如果 checkpoint 不够密，可以使用现有 densification event 数据。

统计：

\[
N_{\text{out}}(t)
\]

以及：

\[
R_{\text{out}}(t)
=
\frac{N_{\text{out}}(t)}
{N_{\text{total}}(t)}
\]

cold / warm 同图。

## 同时分方向统计

不能只统计：

```text
outside / inside
```

需要区分：

\[
x<x_{min}
\]

\[
x>x_{max}
\]

\[
y<y_{min}
\]

\[
y>y_{max}
\]

\[
z<z_{min}
\]

\[
z>z_{max}
\]

最好输出：

```text
outside_x_low
outside_x_high
outside_y_low
outside_y_high
outside_z_low
outside_z_high
```

## 需要回答

如果主要是：

\[
z_{\rm outside}
\]

增加，那么与当前 z-edge deficit 直接相关。

如果主要是 x/y，则：

> overall capacity leakage 和 z-edge deficit 可能是两个不同机制。

不要提前把二者混为一谈。

---

# 8. Experiment B：Gaussian displacement / migration

## 目的

判断 volume-outside Gaussian 是：

> 原本在 volume 内，训练后跑出去；

还是：

> densification 时直接在 volume 外出生。

## 对已有 Gaussian 追踪

如果 Gaussian ID 可以追踪：

\[
p_i(t_0)
\rightarrow
p_i(t_1)
\]

统计：

\[
\Delta p_i
=
p_i(t_1)-p_i(t_0)
\]

以及：

\[
\Delta x_i,\Delta y_i,\Delta z_i
\]

尤其关注最终 outlier：

\[
p_i(5000)\notin V
\]

回看：

\[
p_i(0)
\]

或 parent Gaussian 的位置。

## 推荐图

### displacement magnitude

\[
\|\Delta p_i\|
\]

cold vs warm histogram。

### z displacement

\[
\Delta z_i
\]

cold vs warm。

### initial-position map

对最终 outlier，画它们的：

\[
p_i(t_{\text{initial}})
\]

看看是否集中在：

\[
z_{\rm edge}
\]

---

# 9. Experiment C：Gradient direction，而不是 norm

## 目的

目前已经知道：

\[
\|g\|_{cold}
\approx
\|g\|_{warm}
\]

因此必须检查：

\[
\text{direction}(g)
\]

是否不同。

## 对边界定义 outward normal

例如上 z boundary：

\[
n=(0,0,+1)
\]

下 z boundary：

\[
n=(0,0,-1)
\]

对每个 Gaussian：

\[
g_i=
\frac{\partial L}{\partial p_i}
\]

但注意 optimizer 更新方向通常是：

\[
-\nabla L
\]

因此真正的位置移动趋势应该分析：

\[
d_i=-g_i
\]

定义 outward component：

\[
d_{\text{out}}
=
d_i\cdot n
\]

若：

\[
d_{\text{out}}>0
\]

表示 optimizer 正在把 Gaussian 往 volume 外推。

## 推荐统计

分别 cold/warm：

\[
E[d_{\text{out}}]
\]

\[
P(d_{\text{out}}>0)
\]

\[
E[d_{\text{out}}\mid d_{\text{out}}>0]
\]

在：

- z lower boundary；
- z upper boundary；
- x/y boundary；

分别统计。

## 如果发现

\[
P_{cold}(d_{\rm out}>0)
>
P_{warm}(d_{\rm out}>0)
\]

则说明：

> cold 的问题不是 gradient 太弱，而是 gradient direction 更容易把 Gaussian 推出有效 volume。

这是非常重要的机制证据。

---

# 10. Experiment D：Gaussian population accounting

这是后续最重要的实验之一。

## 目标

对每个 spatial bin 建立：

\[
N(z,t+\Delta t)
-
N(z,t)
\]

的来源分解。

## 定义

对于 bin \(b\)：

\[
\Delta N_b
=
B_b
-
P_b
+
M_{in,b}
-
M_{out,b}
\]

其中：

### Birth

\[
B_b
\]

在 densification 时新创建、且落入 bin \(b\) 的 Gaussian 数。

### Prune

\[
P_b
\]

从 bin \(b\) 被删除的 Gaussian 数。

### Migration-in

\[
M_{in,b}
\]

从其他 bin 移入 \(b\)。

### Migration-out

\[
M_{out,b}
\]

从 \(b\) 移出到其他 bin 或 volume 外。

## 时间范围

重点：

\[
0\rightarrow5000
\]

最好按照每次 densification event 做统计。

而不是只看 eval checkpoint。

## 最终图

横轴：

\[
t
\]

纵轴：

\[
z
\]

分别画：

1. Birth map；
2. Prune map；
3. Migration-out map；
4. Migration-in map；
5. Net change map。

cold / warm 对照。

---

# 11. Experiment E：Global Gaussian budget competition

## 动机

当前：

\[
N_{\max}=500k
\]

两模型都很快达到上限。

因此可能发生：

> early high-gradient region 优先 densify，把 global Gaussian budget 吃满。

导致后续空间区域无法继续补充 Gaussian。

## 实验

保持其他所有条件不变，只改变：

\[
N_{\max}
\]

比如：

\[
500k,\quad750k,\quad1000k
\]

只训练到：

\[
5000
\]

即可。

不需要完整 30k。

## 观察指标

不要先看 PSNR。

重点看：

\[
N_G(z,5000)
\]

\[
N_{\rm out}(5000)
\]

\[
Birth(z)
\]

## 判定

如果提高 cap 后 cold：

\[
N_G^{cold}(z_{\rm edge})
\uparrow
\]

明显追近 warm：

说明：

\[
\boxed{
\text{global capacity competition}
}
\]

是主要因素。

如果完全没改善：

说明 cap 不是主因，应继续关注 migration / pruning。

---

# 12. Experiment F：限制 Gaussian 移出 volume

这是非常重要的 intervention experiment。

## 目的

判断：

\[
\text{capacity leakage}
\]

是不是最终 reconstruction gap 的因果原因。

## 简单版本

训练过程中对 Gaussian center：

\[
p_i
\]

限制在 reconstruction bounding box：

\[
p_i\in V
\]

可以采用：

### clamp

最简单诊断版本：

\[
x=\operatorname{clip}(x,x_{min},x_{max})
\]

等。

虽然不是最终算法设计，但足够做因果实验。

## 比较

四组：

\[
cold
\]

\[
cold+bound
\]

\[
warm
\]

\[
warm+bound
\]

## 如果

\[
cold+bound
\]

明显：

- volume 内 Gaussian 增多；
- z-edge occupancy 恢复；
- reconstruction PSNR / sharpness 接近 warm；

那么：

\[
\boxed{
\text{Gaussian leakage is causal}
}
\]

如果 occupancy 改善但 reconstruction 几乎不变：

说明 out-of-volume Gaussian 并不是主要性能瓶颈。

这一点必须靠干预确认，不能只凭数量相关性下结论。

---

# 13. Experiment G：禁止 densification，单独研究 position optimization

可以作为机制拆分实验。

## 目的

区分：

\[
\text{initialization}
\rightarrow
\text{position drift}
\]

和：

\[
\text{initialization}
\rightarrow
\text{densification difference}
\]

哪个先发生。

## 设置

cold / warm 都：

- 50k initialization；
- 关闭 densification；
- 关闭 prune 或保持一致；
- 只优化 Gaussian parameters。

运行：

\[
0\rightarrow5000
\]

观察：

\[
N_{\rm out}(t)
\]

虽然 Gaussian 数固定，但位置可移动。

## 判定

如果没有 densification 时：

\[
N_{\rm out}^{cold}
\gg
N_{\rm out}^{warm}
\]

已经出现，

说明：

> initialization 直接改变了 center optimization dynamics。

如果关闭 densification 后 cold/warm 几乎一样：

说明：

> 差异主要由 densification mechanism 放大。

---

# 14. Experiment H：只 densify，不优化 center

这是 G 的互补实验。

可以：

- freeze xyz；
- 允许 density/scale 优化；
- 保留 densification。

观察 cold/warm 的：

\[
Birth(z,t)
\]

如果仍存在明显不同：

说明：

> densification trigger 本身就与 warm/cold parameter state 有关。

如果差异基本消失：

说明：

> center optimization / migration 是重要前置因素。

---

# 15. 推荐的实验决策树

Agent 不要机械地把所有实验一次性跑完。

应该根据结果动态推进。

### Step 1

先跑：

\[
N_{\rm out}(t)
\]

如果 cold/warm 的差异在 0–5k 快速出现：

↓

进入 migration 分析。

如果差异直到很晚才产生：

↓

重新检查 prune / later optimization。

### Step 2

分析 outside direction。

如果主要是 z：

↓

重点检查 z boundary gradient direction。

如果主要 x/y：

↓

不要把 z-edge occupancy deficit 和 outside Gaussian 强行绑定。

### Step 3

做 population accounting。

如果：

\[
Migration_{out}
\]

cold 显著更多：

↓

优先做 bounding-box intervention。

如果：

\[
Birth_{edge}^{cold}
\]

明显更少：

↓

优先研究 densification trigger / global budget。

如果：

\[
Prune_{edge}^{cold}
\]

更多：

↓

检查 pruning criterion 是否对 initialization state 有偏置。

### Step 4

如果发现 global budget competition：

↓

提高 \(N_{\max}\) 做诊断。

如果提高 cap 可以消除差异：

↓

研究 spatially balanced densification。

如果无效：

↓

不要继续围绕 cap 展开。

---

# 16. Agent 需要避免的几个分析错误

## 不要再从 final map 直接推原因

例如：

\[
N_G(z,30000)\text{ 少}
\]

不能直接解释成：

> 这里没 densify。

因为它可能是：

- birth 少；
- migration out；
- prune 多。

必须拆 population accounting。

## 不要把 histogram 空白自动解释为 gradient 缺失

所有空间统计必须：

- 固定 reconstruction volume range；
- separately count out-of-range Gaussian；
- 不允许 outlier 自动扩张 histogram range。

## 不要混淆 gradient norm 和 update direction

optimizer 实际移动：

\[
-\nabla L
\]

不是：

\[
\nabla L
\]

分析 outward migration 时必须使用实际 parameter update direction。

如果 optimizer 是 Adam，最好进一步读取实际：

\[
\Delta xyz
\]

因为：

\[
\Delta xyz
\neq
-\eta\nabla L
\]

严格成立。

Adam 还有 momentum / normalization。

所以**最可靠的是直接统计真实 xyz update**：

\[
\Delta p_i
=
p_i(t+1)-p_i(t)
\]

gradient direction 只是辅助解释。

---

# 17. 一个非常重要的改进：优先分析真实 displacement，而不是 raw gradient

考虑 Adam：

\[
m_t
=
\beta_1m_{t-1}
+
(1-\beta_1)g_t
\]

\[
v_t
=
\beta_2v_{t-1}
+
(1-\beta_2)g_t^2
\]

最终：

\[
\Delta p_t
=
-\eta
\frac{\hat m_t}
{\sqrt{\hat v_t}+\epsilon}
\]

所以即便：

\[
\|g\|_{cold}
\approx
\|g\|_{warm}
\]

实际：

\[
\|\Delta p\|
\]

仍然可能不同。

因此后续优先级应该是：

\[
\boxed{
\text{actual displacement}
>
\text{optimizer update}
>
\text{raw gradient}
}
\]

如果代码允许，最好直接记录：

```python
xyz_before
optimizer.step()
xyz_after
delta_xyz = xyz_after - xyz_before
```

这比分析 raw gradient 更可靠。

---

# 18. 最终希望建立的机制链

目前最值得验证的目标模型是：

\[
\text{Warm prior}
\]

↓

更合理的 Gaussian parameters / spatial correspondence

↓

early optimization 时：

\[
\Delta xyz
\]

更稳定、更少无效漂移

↓

更多 Gaussian 留在 reconstruction support

↓

densification budget 更有效地分配到真实结构区域

↓

\[
N_{\text{effective}}
\uparrow
\]

↓

early spatial allocation 被锁定

↓

后期优化拥有更好的 representation basis

↓

最终：

\[
\boxed{
\text{higher sharpness + better PSNR/SSIM}
}
\]

cold 则可能：

\[
\text{poor initial parameter state}
\]

↓

early center optimization / densification 不稳定

↓

capacity leakage / misallocation

↓

500k budget 被提前锁死

↓

后期无法重新组织 representation

↓

最终细节恢复不足。

---

# 19. 对“螺旋 CT 特性”的态度

目前不要声称：

> 这个机制是 spiral CT 特有。

现有实验只能证明：

> 在当前 spiral fan-beam 数据和 R²Gaussian-style optimization 下出现这种现象。

最终必须增加 circular-cone control：

使用：

- 相同 initialization；
- 相同 Gaussian cap；
- 相同 densification；
- 尽量匹配投影数量和体素分辨率。

比较：

\[
R_{\rm out}
\]

\[
N_G(z,t)
\]

\[
Migration
\]

\[
Birth
\]

如果：

\[
\text{spiral cold/warm gap}
\gg
\text{circular cold/warm gap}
\]

才能进一步说：

\[
\boxed{
\text{spiral geometry amplifies initialization sensitivity}
}
\]

然后再研究为什么。

---

# 20. Agent 的近期最高优先级

按顺序：

1. **验证 0–5k 的 volume-outside Gaussian growth。**
2. **按 x/y/z 六个方向拆解 outside Gaussian。**
3. **直接记录/分析 Gaussian actual displacement。**
4. **建立 birth / prune / migration population accounting。**
5. **找到 cold z-edge deficit 的真正来源。**
6. **用 bounding-box constraint 或其他 intervention 验证因果。**
7. **再测试 Gaussian cap competition。**
8. **最后才考虑设计新算法。**

暂时不要急着设计：

- visibility-aware densification；
- z-balanced sampling；
- edge补点；
- gradient normalization。

因为现在没有足够证据证明它们针对真正根因。

---

# 21. Agent 每完成一次实验应自动输出的内容

不要只生成图。

每次需要固定输出：

### Observation

客观现象和数值。

### Supports

它支持哪个假说。

### Rejects

它削弱或否定哪个假说。

### Remaining ambiguity

结果仍然无法区分哪些机制。

### Next experiment

下一项最有信息增益的实验是什么。

推荐格式：

```text
Observation:
cold 在 step 0→3000 期间 z_high outside ratio 从 1.2% 增至 6.5%，
warm 仅从 1.1% 增至 2.4%。

Supports:
cold early-stage outward migration hypothesis.

Rejects:
pure global Gaussian-cap competition cannot alone explain this phenomenon.

Remaining ambiguity:
尚不能判断 Gaussian 是被 optimizer 移出 volume，
还是 densification child 直接出生在 volume 外。

Next:
追踪 parent-child birth position，并统计 real xyz displacement。
```

这会让整个实验链自己推进，而不是“画一张图再回来问人”。

---

# 22. Agent 的总原则

> **不要追求证明已有猜测，而要优先设计能区分竞争性解释的实验。**

当前最重要的三个竞争机制就是：

\[
\boxed{
\text{Densification allocation}
}
\]

vs.

\[
\boxed{
\text{Gaussian migration}
}
\]

vs.

\[
\boxed{
\text{Pruning / budget competition}
}
\]

先把 cold 0–5k 的 Gaussian population deficit 分解清楚，再谈算法创新。

如果这条链最终能够证明：

\[
\text{warm prior}
\rightarrow
\text{less early capacity leakage}
\rightarrow
\text{better spatial allocation}
\rightarrow
\text{better reconstruction}
\]

那就已经是一个相当完整的机制分析了。

之后再做 circular control，才有资格讨论它是不是：

\[
\boxed{
\text{spiral CT 特有或被 spiral geometry 显著放大的 optimization pathology}
}
\]
