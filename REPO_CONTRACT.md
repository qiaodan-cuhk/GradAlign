# GradAlign 实验约定

## 动手前三句话

1. 列出本方法改动的文件，包括实验 YAML 和必要的回归用例。
2. 写明注册名、输入信号和输出形式，以便核对要运行哪些阶段。
3. 说明与对照共享的候选池、随机种子、GRPO 预算、留出集和计算成本口径。

## 唯一的在线实验入口

- 从仓库根目录运行 `python automated/dynamic_selection.py --config configs/webinstruct_math.yaml`。
- 新选题方法在 `automated/select_data.py` 写选择函数，向 `SELECTOR_REGISTRY` 登记一个 `Selector`。
- 函数读取 `SelectionInputs`，在注册项声明候选回答、准确率、梯度模式和参考集依赖。
- 在 `configs/` 增加方法 YAML，设置 `mode` 和独立的 `prefix`。命令行参数优先。
- 全量 GRPO 对照直接走 `automated/launch_verl_training.py`，使用同一份留出集和判题器。

## 数据与训练的交接

- `extra_info.index` 标识当前数据集的原题，进程间与断点都使用这个 ID。
- 裁剪型选择器返回本轮候选 ID 列表，每题最多一次，公共写出器生成 JSONL 与 `train.parquet`。
- 路由型选择器可返回 `RoutedSelection`，为当前候选窗口的每道题给出一个标签，并可附加分数。
- 整池 `sim_route` 使用 `k: 1`，写出每道题的浮点 `align_score` 和整数 `route_tag`。0 表示双分支，1 表示仅保留分支，2 表示仅遗忘分支。
- 原版 GRPO 忽略选题元数据。C 组的 `RoutedRLHFDataset` 将标签作为样本张量传给独立的 `RoutedPPOActor`。
- `n_samples_train` 和 `n_samples_val` 是选题侧每题回答数，`rollout_n` 是正式训练每题新生成的回答数。
- `train_batch_size` 计题目，`iters_per_select` 计 GRPO step。比较方法时单独记录两者和每题回答数。

## 对照组共用的代码

- 方法 PR 沿用 `prepare_data.py`、分块顺序、尾块处理、轮次调度和断点规则。
- 参考集 `selection_reference` 用于选题，`heldout_eval` 用于评测。文件哈希与行数见 `experiment.json`。
- 沿用公共数学判题器、留出集采样配置及评测频率。基础设施修复单独提交并对两组生效。
- 沿用 `automated/timing.py`，每组同时报告留出集指标对应的 GRPO step 和累计分配 GPU 小时。
- 两组均显式给出作业独占的 `--allocated_gpus` 数目。没有卡数时，账本只报告耗时。
- 原版选题、随机对照和全量 GRPO 共用训练入口。修改选题方法的 PR 保持主循环和 veRL 训练实现只读。

## CPU 冒烟

每个改动和 PR 上 GPU 前，在装有本仓库 veRL 及 pytest 的环境中从仓库根目录执行同一条命令。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false python -m pytest -q tests/test_selection_baseline.py tests/test_graft_*.py verl/tests/trainer/ppo/test_core_algos_on_cpu.py::test_grpo_and_vectorized_equivalence
```

用例覆盖 14 道合成候选题、3 组 GRPO 优势、四进程 CPU FSDP 路由与优化器续训、双态 HF 导出，以及三份 YAML 的训练入口。GPU rollout 与 Oracle 分析在调度测试中使用替身。
