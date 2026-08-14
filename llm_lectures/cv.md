### 高效深度学习模型压缩与硬件感知优化｜MIT

**技术栈：** Python、PyTorch、Transformers、Pruning、Quantization、NAS、AWQ

* 实现 VGG 的细粒度剪枝、通道剪枝、K-Means 量化及 INT8 整数推理；将模型由 **35.20 MiB 压缩至 7.79 MiB（4.5×）**并保持 **92.79%** 准确率，结构化剪枝将计算量降低 **49.7%**、推理延迟降低 **15.5%**，INT8 模型准确率达到 **92.82%**。
* 基于 OFA/MCUNetV2 构建硬件感知神经架构搜索流程，实现 MACs、峰值内存分析器和三层 MLP 精度预测器，并完成随机搜索与进化搜索；在 **60M MACs、250 KB** 约束下获得 **92.56%** 准确率的 VWW 子网络。
* 在 OPT-1.3B 上实现 **3-bit AWQ 权重量化**，通过激活统计、显著通道缩放及网格搜索降低量化误差，将困惑度由朴素量化的 **121.90 降至 17.93**，同时将权重存储由 **5043.73 MiB 降至 495.06 MiB（10.2×）**。



### nano-vLLM 推理引擎与 Qwen3-32B 部署

* 优化 Scheduler、BlockManager，实现跨请求 Prefix Cache 复用，多轮对话缓存命中率提升 50%。
* 重构 GQA Decode PagedAttention Kernel，延迟从 3.11 ms 降至 40.19 μs，提升了 77 倍，并行度（Waves / SM）提升 6 倍。
* 4卡 L20 部署 Qwen3-32B 并进行压力测试：16 并发下 Prefill 吞吐达 5954 tokens/s，Decode 聚合吞吐达 435 tokens/s，16K 共享前缀场景缓存命中率达 98.69%。
