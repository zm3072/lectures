# Lecture 16 

如何利用可以自动判定对错的奖励，把后训练从主观偏好对齐扩展到可规模化的推理、代码与智能体能力训练。

## 知识路线

RLHF 的代理奖励瓶颈 → 可验证奖励 RL（RLVR） → 复习 PPO → 去掉 Critic 得到 GRPO → 分析 GRPO 的无偏性与长度偏差 → DeepSeek-R1：纯 RL、SFT 冷启动与蒸馏 → Kimi K1.5：数据难度、长度控制与 RL 系统 → Qwen3：低数据 RL、思考模式融合与测试时扩展 → Agentic 

#### RLVR

让模型生成回答，再通过程序、规则、答案检查器等客观方式判断回答是否正确，并把验证结果作为 reward 训练模型。

### PPO 从公式到语言模型实现

参考 minimind 项目中的实现。

### GRPO

GRPO 用同题多样本的相对表现替代 Critic，换来实现简单；代价是 advantage 不再是标准价值估计，并引入组归一化与长度偏差。

#### GRPO 定义

GRPO 保留 PPO 式采样、概率比裁剪和 KL 约束，但去掉 value model。对同一 prompt 采样一组回答，用组内 reward 的 z-score 作为 advantage。

#### GRPO 流程

1. 对每个 prompt 生成多条 rollout；
2. 逐条计算 reward；
3. 在组内计算均值/方差并标准化；
4. 加入 KL 约束；
5. 对 policy loss 做梯度更新。

#### GRPO 并非严格无偏

策略梯度可以减去只依赖状态、不依赖当前动作的 baseline 而保持期望梯度不变。合法 baseline 的目的在于降方差，而不是改变优化目标。

组内标准差依赖同组采样动作，除以它会改变梯度的期望，因此不是传统意义上保持无偏的 baseline。leave-one-out 等方法可构造更接近无偏的组相对估计，同时还可修改长度归一化。

```text
原始目标：最大化期望奖励
        ↓
真实策略梯度无法直接计算
        ↓
使用有限 rollout 构造随机梯度
        ↓
合法 baseline：
只依赖状态，不依赖当前动作
→ 降低方差，不改变期望
        ↓
GRPO 组均值：
包含当前回答自己的 reward
→ 严格来说产生常数缩放偏差
        ↓
GRPO 组标准差：
依赖整组采样动作
→ 随机重加权不同轨迹和 prompt
→ 一般会改变梯度方向
        ↓
回答长度归一化：
分母依赖生成长度
→ 不同长度轨迹被重新加权
→ 产生 response-level length bias
        ↓
改进：
LOO baseline
+ 移除组标准差归一化
+ 固定长度分母
```

策略梯度允许减去只依赖状态、而不依赖当前动作的 baseline：

$$
\nabla_\theta J
= \mathbb{E}
\left[
(R-b(x))
\nabla_\theta\log\pi_\theta(y\mid x)
\right]
$$

因为：

$$
\mathbb{E}_{y\sim\pi_\theta}
\left[
\nabla_\theta\log\pi_\theta(y\mid x)
\right]
=0
$$

所以合法 baseline 只降低方差，不改变梯度期望。

GRPO 对同一个 prompt 采样多个回答，并使用组内 reward 均值和标准差构造 advantage：

$$
\hat A_i
=
\frac{R_i-\bar R}
{\hat\sigma_R+\epsilon}
$$

组均值 $\bar R$ 包含当前样本自身的 $R_i$，所以它依赖当前动作。只减组均值时，梯度期望会被缩放为：

$$
\mathbb{E}[\hat g]
=
\frac{G-1}{G}\nabla J
$$

因此不是严格无偏，但在理想条件下只改变整体尺度。

组内标准差 $\hat\sigma_R$ 同样依赖整组采样动作，并与分子相关。一般有：

$$
\mathbb{E}
\left[
\frac{X}{\hat\sigma_R}
\right]
\neq
\frac{\mathbb{E}[X]}
{\mathbb{E}[\hat\sigma_R]}
$$

因此，标准差归一化会重新加权不同轨迹和不同 prompt，通常不仅改变梯度大小，还可能改变梯度方向。它不是传统意义上保持无偏的 baseline。

Leave-one-out 方法为第 $i$ 条回答使用其他回答的均值：

$$
b_{-i}
= \frac{1}{G-1}
\sum_{j\ne i}R_j
$$

由于 $b_{-i}$ 不依赖当前回答 $y_i$，在条件独立假设下可以构造无偏的组相对策略梯度。

此外，GRPO 按回答自身长度归一化：

$$
\frac{1}{L_i}\sum_t\ell_{i,t}
$$

也会因为 $L_i$ 依赖当前轨迹而引入长度偏差：短的正优势回答被更强强化，短的负优势回答被更强惩罚，使长错误回答被相对较少地惩罚。使用统一固定分母可以减轻这种 response-level length bias。

最终应当理解为：

> **GRPO 通过引入一定偏差换取了更低方差和更稳定的训练，但组标准化与长度归一化可能改变原始期望奖励目标。**


#### GRPO Length bias

由于 GRPO 把每条回答的梯度除以自身长度，短的好回答被更强强化，短的坏回答被更强惩罚。因此好回答趋向简短，而坏回答相对趋向冗长。	​

关于这一点的修复 Dr.GRPO

```
补充1：DAPO

主要改动：
1. Clip-Higher

2. 动态采样(The More the Merrier: Dynamic Sampling)

过滤掉准确性为 1 和 0 的

3. Token-Level策略梯度损失

传统的GRPO算法采用样本级损失计算，导致长响应中的token对整体损失的贡献较低。（如何理解？ PPO DPO GRPO 都是样本级损失计算？）

4. 过长奖励整形(Hide and Seek: Overlong Reward Shaping)

软过长惩罚机制，通过长度感知的惩罚区间，逐步增加对过长响应的惩罚，从而减少奖励噪声并稳定训练。


简单对比 GRPO 和 DAPO

对比DeepSeek提出的GRPO（Group Relative Policy Optimization）和DAPO（Decoupled Clip and Dynamic sAmpling Policy Optimization）。

(1) 策略优化机制

GRPO：通过群组相对奖励归一化估计优势，消除对价值函数的依赖。其目标函数包含KL散度惩罚项，以限制策略偏离参考模型，但这一设计在长链式推理（CoT）场景中可能限制模型探索空间。
DAPO：移除了KL散度惩罚，允许模型在长推理任务中自由探索。通过解耦裁剪（Decoupled Clip）调整上下裁剪范围（ 
 =0.2， 
 =0.28），提升低概率token的探索能力，有效缓解熵崩溃问题。
(2) 训练效率优化

GRPO：在提示样本准确率趋近1时，梯度信号减弱，导致训练效率下降。此外，样本级损失计算可能忽略长序列中关键token的影响。
DAPO：引入动态采样策略，过滤准确率为0或1的无效样本，确保批次内梯度有效性；采用Token级策略梯度损失，强化长序列中每个token的贡献，避免低质量模式（如重复生成）的干扰。
(3) 奖励机制设计

GRPO：依赖传统奖励模型，存在Reward Hacking风险，可能导致模型过度优化局部奖励而非全局推理质量。
DAPO：采用基于规则的奖励建模，直接以任务最终准确率作为奖励信号（如AIME答案正确性），并结合过长奖励整形（Soft Overlong Punishment），对超长响应施加动态惩罚，减少噪声干扰。

```

### DeepSeek-R1


### Kimi-1.5


### Qwen3


### Agentic RL














# KL 散度

先约定记号：

* $P(x)$：目标分布、真实数据分布；
* $Q(x)$：模型分布、近似分布。

KL 散度定义为：

$$
D_{\mathrm{KL}}(P \,\|\, Q)
= \sum_x P(x)\log\frac{P(x)}{Q(x)}
$$

连续情况把求和换成积分。

所谓正向 KL 和反向 KL，就是交换 $P$ 与 $Q$ 的位置：

* 正向 KL：$D_{\mathrm{KL}}(P \,\|\, Q)$
* 反向 KL：$D_{\mathrm{KL}}(Q \,\|\, P)$

注意，KL 不对称：

$$
D_{\mathrm{KL}}(P \,\|\, Q) \neq D_{\mathrm{KL}}(Q \,\|\, P)
$$


## 正向 KL：$D_{\mathrm{KL}}(P \,\|\, Q)$

展开为：

$$
D_{\mathrm{KL}}(P \,\|\, Q)
= \mathbb{E}_{x\sim P}
\left[
\log \frac{P(x)}{Q(x)}
\right]
$$

关键点在于：

> 它是在真实分布 $P$ 中采样，然后检查模型 $Q$ 是否给这些真实样本足够高的概率。

正向 KL 会要求模型：

> 真实数据可能出现的地方，你都必须覆盖。

如果某个位置满足：

$$
P(x)>0,\qquad Q(x)\approx 0
$$

那么：

$$
P(x)\log\frac{P(x)}{Q(x)}
$$

会非常大，会给模型很大的惩罚项。

也就是说，当真实分布中有概率发生的点，model 的分布没有给概率就会收到很大惩罚，正向 KL 非常不喜欢模型漏掉真实分布中的某个模式。

因此正向 KL 通常具有：

$$
\boxed{\text{mode-covering，模式覆盖}}
$$

的倾向。


## 反向 KL：$D_{\mathrm{KL}}(Q \,\|\, P)$

展开为：

$$
D_{\mathrm{KL}}(Q \,\|\, P)
= \mathbb{E}_{x\sim Q}
\left[
\log\frac{Q(x)}{P(x)}
\right]
$$

关键点变成：

> 从模型 $Q$ 自己的分布中采样，然后检查这些样本在真实分布 $P$ 中是否合理。

如果某个位置满足：

$$
Q(x)>0,\qquad P(x)\approx 0
$$

那么：

$$
Q(x)\log\frac{Q(x)}{P(x)}
$$

会非常大。

也就是说，当真实分布的点出现概率很低时，model 会受到很大惩罚。

因此，反向 KL 非常不喜欢模型把概率放在真实分布的低概率区域。

它更倾向于：

> 找到一个安全的高概率区域，然后集中在那里。

因此反向 KL 通常具有：

$$
\boxed{\text{mode-seeking，模式寻找}}
$$

的倾向。

## 和最大似然训练的关系

语言模型预训练通常使用最大似然：

$$
\max_\theta
\mathbb{E}_{x\sim P_{\text{data}}}
\left[\log Q_\theta(x)\right]
$$

等价于最小化：

$$
D_{\mathrm{KL}}
\left(
P_{\text{data}} \,\|\, Q_\theta
\right)
$$

忽略与模型参数无关的常数即可。

因此，最大似然训练本质上对应正向 KL。

这解释了为什么最大似然模型通常倾向于覆盖训练数据的各种模式：

* 不同写作风格；
* 不同答案表达；
* 不同语义可能性；
* 不同数据群体。

如果模型给真实数据很低概率，就会受到直接惩罚。


## 在强化学习和 LLM 对齐中的含义

在 LLM 强化学习中，经常加入参考模型 KL 约束。

设：

* $\pi_\theta$：当前策略模型；
* $\pi_{\mathrm{ref}}$：参考模型，通常是 SFT 模型。

常见形式是：

$$
D_{\mathrm{KL}}
\left(
\pi_\theta \,\|\, \pi_{\mathrm{ref}}
\right)
$$

即：

$$
\mathbb{E}_{y\sim\pi_\theta}
\left[
\log\frac{\pi_\theta(y\mid x)}
{\pi_{\mathrm{ref}}(y\mid x)}
\right]
$$

这是相对于参考模型的反向 KL。

它关注的是：

> 当前策略实际会生成的回答，在参考模型看来是不是过于离谱。

如果当前策略把概率放到参考模型几乎不支持的输出上，就会受到较大惩罚。

它的作用包括：

* 防止策略偏离 SFT 模型太远；
* 减少奖励 hacking；
* 保持语言流畅性；
* 防止输出分布发生剧烈坍缩。


如果使用另一方向

$$
D_{\mathrm{KL}}
\left(
\pi_{\mathrm{ref}} \,\|\, \pi_\theta
\right)
$$

则更关注：

> 参考模型原来支持的各种回答，当前模型是否仍然覆盖。

它更倾向于保留参考模型的多样性，但实际计算通常更麻烦，因为需要从参考模型分布中采样或完整估计。