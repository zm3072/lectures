**kvdivergence · KV Cache 量化对 LLM 贪心输出的行为学评估** — *独立研究项目，Python / llama.cpp*

- **研究动机**：业界以困惑度为依据将 8-bit KV cache 当作"近乎无损"的默认部署方案，但对于依赖输出确定性的下游场景（KV 缓存复用、单元测试、审计），关心的是 token 级文本等价而非分布距离。项目要回答：KV cache 量化是否会改变贪心解码下的实际输出文本？——这一问题聚合指标无法回答。
- **度量方案设计**：提出三个字符级、确定性的行为学度量——首次分歧位置（first-divergence）、翻转比例（flip-fraction，首次分歧位置占参考文本长度的比例）、改变比例（change-rate），替代困惑度作为"是否无损"的判据；度量精确、无需 token 对齐或 LLM 打分。
- **实验方案设计**：搭建三台仅 KV 精度不同的 llama.cpp 服务（fp16 / q8_0 / q4_0），共用权重、种子、上下文与 Flash Attention，将 `-ctk / -ctv` 隔离为唯一自变量；加入同服务 fp16 两次请求的自洽对照（30/30 字节一致）排除运行抖动，关闭跨服务 prompt cache 复用避免污染 KV 状态。
- **主要发现**：30 条固定提示、128 token 贪心续写下，q8_0 在 **83%** 的提示上改变输出（中位翻转点 41%），q4_0 在 **100%** 上改变（中位翻转点 0%）；由于 fp16 与 q8_0 共用 Flash Attention 实现，分歧可归因于 KV 精度而非注意力实现——**8-bit KV 在 token 级并非无损**。

### 高效深度学习模型压缩与硬件感知优化｜MIT

**技术栈：** Python、PyTorch、Transformers、Pruning、Quantization、NAS、AWQ

* 实现 VGG 的细粒度剪枝、通道剪枝、K-Means 量化及 INT8 整数推理；将模型由 **35.20 MiB 压缩至 7.79 MiB（4.5×）**并保持 **92.79%** 准确率，结构化剪枝将计算量降低 **49.7%**、推理延迟降低 **15.5%**，INT8 模型准确率达到 **92.82%**。
* 基于 OFA/MCUNetV2 构建硬件感知神经架构搜索流程，实现 MACs、峰值内存分析器和三层 MLP 精度预测器，并完成随机搜索与进化搜索；在 **60M MACs、250 KB** 约束下获得 **92.56%** 准确率的 VWW 子网络。
* 在 OPT-1.3B 上实现 **3-bit AWQ 权重量化**，通过激活统计、显著通道缩放及网格搜索降低量化误差，将困惑度由朴素量化的 **121.90 降至 17.93**，同时将权重存储由 **5043.73 MiB 降至 495.06 MiB（10.2×）**。



### nano-vLLM 推理引擎与 Qwen3-32B 部署

* 优化 Scheduler、BlockManager，实现跨请求 Prefix Cache 复用，多轮对话缓存命中率提升 50%。
* 重构 GQA Decode PagedAttention Kernel，延迟从 3.11 ms 降至 40.19 μs，提升了 77 倍，并行度（Waves / SM）提升 6 倍。
* 4卡 L20 部署 Qwen3-32B 并进行压力测试：16 并发下 Prefill 吞吐达 5954 tokens/s，Decode 聚合吞吐达 435 tokens/s，16K 共享前缀场景缓存命中率达 98.69%。


L1 · 通用基础

Transformer 注意力机制、KV cache 的作用与结构
贪心解码 vs 采样、logits 与 argmax
fp32/fp16/bf16 数值范围与精度、量化误差如何在注意力里传播
困惑度定义及其盲区（分布级期望损失为什么掩盖 token 级 argmax 翻转）
L2 · Efficient AI 专门

权重量化：GPTQ、AWQ、SmoothQuant、bitsandbytes(llm.int8)、QLoRA
KV / 激活量化：KIVI、KVQuant、llama.cpp 的 q8_0 / q4_0 KV
PTQ vs QAT 的差异与选择
其他推理加速：Speculative Decoding、Token Compression / KV Eviction（H2O、StreamingLLM、SnapKV）、Flash Attention v1/v2/v3
视觉侧对应技术（桥接老师方向）：Q-Diffusion、PTQ4DM、扩散蒸馏（LCM、SDXL-Turbo）、视频生成的 KV / feature cache
L3 · 评估与度量设计（重心调整为"如何设计 metric 与实验"，不再讲科研范式）

聚合级 vs 行为学度量的差异：perplexity / FID / CLIPScore 各自遮蔽了什么
确定性度量：为什么字符级/像素级精确比较优于 LLM-as-judge
控变量实验设计：怎么把一个多因素系统隔离到单一自变量、正/负对照、self-consistency check
生成模型的确定性：温度、种子、cache、注意力实现各自如何引入不确定性
CV 侧类比：确定性采样下扩散模型的行为一致性怎么测
L4 · 工程栈（简历不写，但准备一下，被问到时能答）

Python 项目结构、类型系统、pytest
LLM 服务栈：llama.cpp、vLLM、HuggingFace Transformers、GGUF
CV 侧对应：PyTorch、diffusers、xFormers
L5 · 阅读清单

KV 量化：KIVI (2024)、KVQuant (2024)
权重量化：GPTQ、AWQ、LLM.int8()
推测解码：Fast Inference from Transformers via Speculative Decoding (Leviathan et al., 2023)
Flash Attention 的数值性：v1/v2 原文关于数值差异的段落
CV 侧迁移：Q-Diffusion、PTQ4DM、LCM —— 面试时可直接提"想把 kvdivergence 的度量与控变量迁移到这里"
