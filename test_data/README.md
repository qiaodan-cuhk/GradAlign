# 四卡 GradAlign / GRAFT 实验数据

从仓库根目录运行 `export GRADALIGN_DATA_ROOT="$PWD/test_data"`，三份 `configs/graft_*.yaml` 即可直接读取下面的本地数据。每个目录含 JSONL、Parquet 和记录上游版本、样本 ID、随机种子的 `pilot_source.json`。

| 目录 | 用途 | 题数 |
| --- | --- | ---: |
| `webinstruct_math80/` | WebInstruct 数学训练候选池 | 80 |
| `amc22_ref16/` | AMC 2022 选题参考集，用于算 Oracle 梯度 | 16 |
| `amc23/` | AMC 2023 跨题留出评测 | 40 |

## 已有离线 Docker

在能访问 GitHub 的电脑上下载一个文件并传入 Docker。压缩包包含上表中的数据、启动脚本和供 Ray 2.43 环境选用的原生 vLLM Oracle 补丁，适配 `feat/graft-oracle-routing` 上的基础提交 `788aeff`。[下载完整离线实验包](gradalign-ppu-toy-788aeff.tar.gz?raw=true)

在 Docker 中进入压缩包所在目录，解压并运行 CPU 选题检查。将 `GA_REPO` 和 `OUT` 设为实际的仓库和结果目录。

```bash
tar -xzf gradalign-ppu-toy-788aeff.tar.gz
cd gradalign-ppu-toy
sha256sum -c SHA256SUMS
export GA_REPO=/ossfs/workspace/GradAlign
export GRADALIGN_DATA_ROOT="$PWD/data"
export OUT=/ossfs/workspace/results/gradalign-ppu-toy-01
bash run-toy.sh select
```

这条命令从前 8 题中按种子 42 选 2 题，结果写入 `$OUT/selector/train.parquet`。模型路径、四卡 B/C 一步训练和输出说明见压缩包内的 `README.md`。

## 数据来源

三份样本由仓库的 `automated/prepare_graft_pilot.py` 使用 seed 42 筛选，并转换为 veRL 的数学问答格式。WebInstruct 来自 TIGER-Lab，AMC 参考与评测来自 AI-MO。两份数据卡均标注 Apache-2.0。[WebInstruct 数据卡](https://huggingface.co/datasets/TIGER-Lab/WebInstruct-verified-unfiltered/blob/ac48a536b31d3cabb346e1139fb6c69ae631abd7/README.md) · [AMC 数据卡](https://huggingface.co/datasets/AI-MO/aimo-validation-amc/blob/69d78a4a2c840e82d69af6bc742bda09005f6316/README.md) · [数据许可证](LICENSE)
