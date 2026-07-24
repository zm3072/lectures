
# MLE 

# KL 散度、交叉熵与 MLE

设：

* $P(x)$：真实数据分布；
* $Q_\theta(x)$：模型学习到的分布。

三者的共同目标是：

> 让模型分布 $Q_\theta$ 尽量接近真实分布 $P$。

---

## 1. MLE：确定优化目标

最大似然估计（MLE）希望模型给训练数据尽可能高的概率：

$$
\max_\theta \prod_{i=1}^{N} Q_\theta(x_i)
$$

取对数后，等价于：

$$
\max_\theta \sum_{i=1}^{N}\log Q_\theta(x_i)
$$

也就是最小化负对数似然：

$$
\min_\theta -\frac{1}{N}\sum_{i=1}^{N}\log Q_\theta(x_i)
$$

因此：

$$
\boxed{\text{MLE} \Longleftrightarrow \text{最小化 NLL}}
$$

---

## 2. 交叉熵：实际使用的损失

交叉熵定义为：

$$
H(P, Q_\theta)
= -\mathbb{E}_{x\sim P}
\left[\log Q_\theta(x)\right]
$$

训练集可以看作从真实分布 $P$ 中采样得到，因此：

$$
-\frac{1}{N}\sum_{i=1}^{N}\log Q_\theta(x_i)
\approx
H(P, Q_\theta)
$$

Q 告诉我们模型给每个样本多少概率；训练数据的出现频率告诉我们这些样本应该被计算多少次。

所以：

$$
\boxed{\text{MLE} \Longleftrightarrow \text{最小化交叉熵}}
$$

在多分类任务中，如果正确类别为 $y$，交叉熵简化为：

$$
L_{\mathrm{CE}}
= -\log Q_\theta(y\mid x)
$$

即提高模型对正确类别分配的概率。

---

## 3. KL 散度：衡量两个分布的差异

正向 KL 散度定义为：

$$
D_{\mathrm{KL}}(P \,\|\, Q_\theta)
=
\sum_x P(x)\log\frac{P(x)}{Q_\theta(x)}
$$

它与交叉熵的关系是：

$$
H(P, Q_\theta)
=
H(P)
+
D_{\mathrm{KL}}(P \,\|\, Q_\theta)
$$

其中 $H(P)$ 只由真实分布决定，与模型参数无关。

因此：

$$
\boxed{
\min_\theta H(P,Q_\theta)
\Longleftrightarrow
\min_\theta D_{\mathrm{KL}}(P \,\|\, Q_\theta)
}
$$

---

## 4. 三者的关系

$$
\boxed{
\text{最大化似然}
\Longleftrightarrow
\text{最小化负对数似然}
\Longleftrightarrow
\text{最小化交叉熵}
\Longleftrightarrow
\text{最小化正向 KL}
}
$$

可以简单理解为：

* **MLE**：统计目标，让真实数据出现的概率最大；
* **交叉熵**：训练时实际计算的损失；
* **KL 散度**：衡量模型分布与真实分布之间的差异。

> MLE、交叉熵和正向 KL，本质上是在从不同角度描述同一个优化过程。
