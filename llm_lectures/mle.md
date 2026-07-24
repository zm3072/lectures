我已根据你提供的材料，去除重复内容并重新串联为一套从**统计学定义 → 优化目标 → 分布学习 → 大语言模型应用**的知识体系。

# 最大似然估计（MLE）知识体系

## 1. MLE 要解决什么问题？

现实中的数据由某个未知的真实分布产生：

$$
x \sim q(x)
$$

但我们无法直接知道真实分布 $q(x)$，只能观察到有限的数据集：

$$
D={x_1,x_2,\dots,x_N}
$$

因此，我们先选择一个参数化模型：

$$
p_\theta(x)
$$

其中：

* $p_\theta(x)$：模型假设的数据分布；
* $\theta$：需要从数据中学习的参数。

最大似然估计的目标是：

> 在给定的模型分布族中，找到一组参数，使模型能够最好地解释已经观察到的数据。

---

## 2. 似然函数

假设样本独立同分布，即：

$$
x_i \overset{\text{i.i.d.}}{\sim} p_\theta(x)
$$

那么整批数据出现的联合概率为：

$$
p_\theta(D)
= p_\theta(x_1,x_2,\dots,x_N)
= \prod_{i=1}^{N}p_\theta(x_i)
$$

当数据 $D$ 已经固定，并把它看作参数 $\theta$ 的函数时，称为**似然函数**：

$$
L(\theta;D)
= \prod_{i=1}^{N}p_\theta(x_i)
$$

最大似然估计为：

$$
\boxed{
\hat{\theta}_{\mathrm{MLE}}
= \arg\max_\theta L(\theta;D)
= \arg\max_\theta
\prod_{i=1}^{N}p_\theta(x_i)
}
$$

直观上：

> 哪一组参数更容易产生我们实际观察到的数据，数据就更支持哪一组参数。

---

## 3. 概率与似然的区别

二者可能使用同一个数学表达式：

$$
p(D\mid\theta)
$$

区别在于观察角度不同。

| 概念 | 固定的量 | 变化的量 | 研究的问题 |
| --- | --- | --- | --- |
| 概率 | 参数 $\theta$ | 数据 $D$ | 给定参数，哪些数据容易出现？ |
| 似然 | 数据 $D$ | 参数 $\theta$ | 给定数据，哪组参数更合理？ |

可以简单记为：

> **概率：参数固定，看数据。**
> **似然：数据固定，看参数。**

MLE 比较的是：

$$
p(D_{\mathrm{observed}}\mid\theta_1),
\quad
p(D_{\mathrm{observed}}\mid\theta_2),
\quad \dots
$$

“最大”是沿着**参数维度**进行比较，而不是说当前数据比其他所有数据更容易出现。

---

## 4. 硬币例子

假设硬币正面朝上的概率为 $\theta$：

$$
P(X=1)=\theta,
\qquad
P(X=0)=1-\theta
$$

抛硬币 10 次，观察到：

* 7 次正面；
* 3 次反面。

这组数据的似然函数为：

$$
L(\theta)=\theta^7(1-\theta)^3
$$

取对数：

$$
\ell(\theta)
= \log L(\theta)
= 7\log\theta+3\log(1-\theta)
$$

求导并令其为零：

$$
\frac{\mathrm d\ell}{\mathrm d\theta}
= \frac{7}{\theta}
- \frac{3}{1-\theta}
= 0
$$

得到：

$$
7(1-\theta)=3\theta
$$

因此：

$$
\boxed{
\hat{\theta}_{\mathrm{MLE}}=0.7
}
$$

这说明伯努利分布参数的最大似然估计，就是样本中正例出现的比例。

---

## 5. 为什么使用对数似然？

原始似然是许多概率的乘积：

$$
L(\theta)
= \prod_{i=1}^{N}p_\theta(x_i)
$$

直接优化存在两个问题：

1. 大量小概率相乘，容易发生数值下溢；
2. 乘积不方便求导和优化。

因此通常对似然取对数：

$$
\log L(\theta)
= \sum_{i=1}^{N}\log p_\theta(x_i)
$$

由于对数函数单调递增：

$$
\arg\max_\theta L(\theta)
= \arg\max_\theta \log L(\theta)
$$

训练通常使用梯度下降，需要最小化损失，因此定义负对数似然：

$$
\mathcal{L}_{\mathrm{NLL}}
= -\sum_{i=1}^{N}\log p_\theta(x_i)
$$

所以：

$$
\boxed{
\text{最大化似然}
\Longleftrightarrow
\text{最大化对数似然}
\Longleftrightarrow
\text{最小化负对数似然}
}
$$

---

## 6. MLE 如何学习数据分布？

真实数据来自未知分布：

$$
x\sim q(x)
$$

我们选择一个模型分布族：

$$
{p_\theta(x):\theta\in\Theta}
$$

然后利用有限样本求解：

$$
\hat{\theta}
= \arg\max_\theta
\sum_{i=1}^{N}\log p_\theta(x_i)
$$

最终得到：

$$
p_{\hat{\theta}}(x)
$$

我们希望：

$$
p_{\hat{\theta}}(x)\approx q(x)
$$

因此，更准确地说：

> MLE 不是直接学习真实分布，而是在选定的模型分布族中，找到最能解释观测样本的参数。

---

## 7. MLE 与 KL 散度的关系

当样本数量足够多时，平均对数似然：

$$
\frac{1}{N}
\sum_{i=1}^{N}
\log p_\theta(x_i)
$$

会接近真实分布下的期望：

$$
\mathbb{E}_{x\sim q}
\left[
\log p_\theta(x)
\right]
$$

前向 KL 散度为：

$$
D_{\mathrm{KL}}(q\|p_\theta)
= \mathbb{E}_{x\sim q}
\left[
\log\frac{q(x)}{p_\theta(x)}
\right]
$$

展开得到：

$$
D_{\mathrm{KL}}(q\|p_\theta)
= \mathbb{E}_{q}[\log q(x)]
- \mathbb{E}_{q}[\log p_\theta(x)]
$$

其中第一项与参数 $\theta$ 无关，因此：

$$
\boxed{
\max_\theta
\mathbb{E}_{q}[\log p_\theta(x)]
\Longleftrightarrow
\min_\theta
D_{\mathrm{KL}}(q\|p_\theta)
}
$$

所以从分布角度看：

> 最大似然估计是在最小化真实数据分布 $q$ 到模型分布 $p_\theta$ 的前向 KL 散度。

---

## 8. MLE 什么时候能逼近真实分布？

MLE 能否学好，主要取决于以下条件。

### 8.1 模型分布族是否合适

如果真实分布 $q(x)$ 位于模型分布族中，并且样本足够多，MLE 有机会恢复真实参数。

如果：

$$
q(x)\notin\{p_\theta(x):\theta\in\Theta\}
$$

那么无论数据有多少，模型都无法完全表示真实分布，只能找到模型族中最接近的分布。

### 8.2 样本是否足够

有限样本存在随机误差。

例如真实硬币正面概率是 $0.7$，但只抛两次并且都是正面，MLE 会得到：

$$
\hat{\theta}=1
$$

随着样本增多，样本比例通常逐渐接近真实概率：

$$
\hat{\theta}_{\mathrm{MLE}}
\rightarrow
	heta_{\mathrm{true}}
$$

这就是 MLE 的**一致性直觉**。

### 8.3 优化是否充分

即使目标函数正确，复杂神经网络也可能由于优化困难，只找到局部或近似解。

---

## 9. MLE 与大语言模型

## 9.1 自回归语言模型

给定一个 token 序列：

$$
x_1,x_2,\dots,x_T
$$

根据概率链式法则：

$$
p_\theta(x_1,x_2,\dots,x_T)
= \prod_{t=1}^{T}
p_\theta(x_t\mid x_{<t})
$$

其中：

$$
x_{<t}=(x_1,\dots,x_{t-1})
$$

语言模型通过提高训练文本中每个真实 token 的条件概率，提高整个序列的概率。

---

## 9.2 语言模型的 MLE 目标

语言模型的最大似然目标为：

$$
\max_\theta
\prod_{t=1}^{T}
p_\theta(x_t\mid x_{<t})
$$

取对数后：

$$
\max_\theta
\sum_{t=1}^{T}
\log p_\theta(x_t\mid x_{<t})
$$

转换为训练损失：

$$
\boxed{
\mathcal{L}_{\mathrm{LM}}
= -\sum_{t=1}^{T}
\log p_\theta(x_t\mid x_{<t})
}
$$

它表示：

> 给定正确的前文，提高真实下一个 token 的预测概率。

---

## 9.3 一个具体例子

训练文本为：

```text
我 喜欢 吃 苹果
```

可以构造以下预测任务：

| 输入前缀              | 目标 token |
| ----------------- | -------- |
| `<BOS>`           | 我        |
| `<BOS> 我`         | 喜欢       |
| `<BOS> 我 喜欢`      | 吃        |
| `<BOS> 我 喜欢 吃`    | 苹果       |
| `<BOS> 我 喜欢 吃 苹果` | `<EOS>`  |

假设在前缀“我喜欢吃”下：

$$
p_\theta(\text{苹果})=0.6
$$

这个位置的损失为：

$$
\mathcal{L}_t
= -\log 0.6
$$

如果模型只给真实 token 分配了 $0.01$ 的概率：

$$
\mathcal{L}_t
= -\log 0.01
$$

损失会明显更大，因此训练会更强烈地调整参数，提高真实 token 的概率。

---

## 10. MLE、NLL 与交叉熵的关系

假设词表大小为 $V$，真实 token 是第 $k$ 个。

真实标签是 one-hot 分布：

$$
y_i=
\begin{cases}
1, & i=k \\
0, & i\neq k
\end{cases}
$$

模型预测分布为 $p_i$，交叉熵为：

$$
\mathcal{L}_{\mathrm{CE}}
= -\sum_{i=1}^{V}y_i\log p_i
$$

由于只有 $y_k=1$：

$$
\mathcal{L}_{\mathrm{CE}}
= -\log p_k
$$

这正是正确 token 的负对数似然。因此，在标准分类和语言模型训练中：

$$
\boxed{
\text{最大似然估计}
\Longleftrightarrow
\text{最小化 NLL}
\Longleftrightarrow
\text{最小化交叉熵}
}
$$

需要注意：

* **MLE** 是参数估计原则；
* **NLL** 是将 MLE 写成最小化目标后的损失；
* **交叉熵** 是从两个分布差异的角度表达同一个目标。

---

## 11. 预训练和 SFT 都属于 MLE

## 11.1 预训练

预训练让模型预测自然文本中的下一个 token：

$$
\mathcal{L}_{\mathrm{pretrain}}
= -\sum_t
\log p_\theta(x_t\mid x_{<t})
$$

它学习的是：

> 自然文本通常按照什么规律出现？

---

## 11.2 SFT

给定用户指令 $x$ 和示范回答：

$$
y=(y_1,\dots,y_T)
$$

SFT 通常优化：

$$
\mathcal{L}_{\mathrm{SFT}}
= -\sum_{t\in\mathrm{answer}}
\log p_\theta(y_t\mid x,y_{<t})
$$

很多实现只对回答部分计算 loss，指令部分通过 mask 忽略。

因此，SFT 本质上是**条件最大似然估计**：

> 给定用户指令，提高人工示范回答出现的概率。

---

## 12. MLE 与偏好优化、强化学习的关系

大模型训练流程可以概括为：

```text
预训练 MLE
    ↓
SFT 条件 MLE
    ↓
偏好优化或强化学习
```

| 方法         | 主要学习目标                       |
| ---------- | ---------------------------- |
| MLE / 预训练  | 模仿自然文本的数据分布                  |
| SFT        | 模仿人工示范回答                     |
| DPO        | 提高 chosen 相对于 rejected 的偏好概率 |
| PPO / GRPO | 提高能够获得高奖励的回答概率               |

MLE 需要一个明确的目标序列，并逐 token 模仿。

PPO、GRPO 等方法则允许模型先生成回答，再根据整个回答的奖励更新策略，不要求存在唯一的标准答案。

---

## 13. MLE 的主要局限

## 13.1 只能模仿训练数据

MLE 会提高训练样本的概率。因此，数据中的错误、偏见和低质量模式也可能被模型学习。

## 13.2 似然高不等于回答质量高

MLE 优化的是：

$$
\log p_\theta(y\mid x)
$$

但用户真正关心的可能是：

* 回答是否正确；
* 是否有帮助；
* 是否安全；
* 推理是否可靠；
* 工具调用是否成功。

这些目标不一定能由 next-token loss 完整表达。

## 13.3 Exposure Bias

训练时通常使用真实前缀：

$$
p_\theta(x_t\mid x_{<t}^{\mathrm{true}})
$$

这称为 **teacher forcing**。

推理时只能使用模型自己生成的前缀：

$$
p_\theta(\hat{x}_t\mid \hat{x}_{<t})
$$

一旦前面生成错误，后续输入就可能偏离训练数据分布，错误可能逐步累积。

## 13.4 MLE 不决定解码策略

MLE 负责学习概率分布，而生成时如何从分布中选择 token，是另一个问题，例如：

* Greedy decoding；
* Temperature sampling；
* Top-k；
* Top-p。

因此：

> MLE 决定模型学到怎样的概率分布，解码策略决定如何从该分布中生成文本。

---

## 14. 最核心的知识主线

```text
未知真实分布 q(x)
        ↓
观察有限样本 D
        ↓
选择参数化模型 pθ(x)
        ↓
计算观测数据的似然
        ↓
最大化似然 / 对数似然
        ↓
最小化 NLL / 交叉熵
        ↓
得到模型分布 pθ(x)
        ↓
希望逼近真实分布 q(x)
```

对应到大语言模型：

```text
真实文本序列
    ↓
预测每个位置的下一个 token
    ↓
计算真实 token 的概率
    ↓
最小化 -log p(真实 token)
    ↓
提高整段真实文本的概率
```

---

## 15. 一句话总结

$$
\boxed{
\hat{\theta}_{\mathrm{MLE}}
= \arg\max_\theta
p(D_{\mathrm{observed}}\mid\theta)
}
$$

最大似然估计的核心思想是：

> 给定已经观察到的数据，在选定的模型分布族中，寻找最容易产生这些数据的参数。

在大语言模型中，它具体表现为：

> 给定真实前文，不断提高真实下一个 token 的预测概率；对海量文本重复这一过程，就构成了语言模型的预训练。

后续材料也可以继续按这套结构整理：**核心问题 → 数学定义 → 直觉解释 → 推导关系 → LLM 应用 → 局限与总结**。
