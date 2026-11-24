# GradAlign: Gradient-Aligned Data Selection for LLM Reinforcement Learning

Official implementation for the paper:

>**GradAlign: Gradient-Aligned Data Selection for LLM Reinforcement Learning**  
>Ningyuan Yang*, Weihua Du*, Weiwei Sun, Sean Welleck, Yiming Yang  
>Preprint

GradAlign is an online RL data-selection method that ranks candidate training problems by alignment between:
- candidate GRPO gradients and
- an aggregated gradient from a small trusted validation set.

The method is designed for non-stationary LLM RL and targets three practical regimes from the paper:
- unreliable reward signals,
- distribution imbalance, and
- low-utility large-scale web corpora.

## Repository layout

```text
.
|-- automated/                   # End-to-end orchestration scripts
|   |-- prepare_data.py          # Dataset download + conversion to VERL-style JSONL/Parquet
|   |-- dynamic_selection.py     # Main loop: infer -> analyze -> aggregate -> select -> train
|   |-- run_inference_local.py   # vLLM/Ray rollout generation
|   |-- run_parallel_analysis.py # Distributed gradient alignment driver
|   |-- aggregate.py             # Per-problem similarity/accuracy aggregation
|   |-- select_data.py           # Selection policies (sim, dot, accgreedy, align, rand, ...)
|   |-- launch_verl_training.py  # GRPO training launcher (VERL backend)
|   |-- merge_model.py           # Merge distributed checkpoints to HF format
|   `-- config.py                # Local model/data path config
|-- select/                      # Gradient/similarity analysis implementation
|-- verl/                        # Modified VERL training framework
`-- scripts/                     # Utility scripts
```

## Selection modes

| Paper name | `--mode` | Description |
|---|---|---|
| GradAlign (cosine) | `sim` | Cosine similarity between candidate and validation gradients (default in paper) |
| GradAlign (dot) | `dot` | Dot product similarity |
| LearnAlign | `align` | Gradient alignment without external validation set |
| AccGreedy | `accgreedy` | Prioritize problems with pass rate closest to 50% |
| Random | `rand_in_chunk` (alias `rand`) | Sample actual IDs from the current chunk without replacement, using seed + round index. Skips rollout generation and gradient scoring. |

## 1) Environment setup

Follow VERL installation instructions: https://verl.readthedocs.io/en/latest/start/install.html

Typical local setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r verl/requirements.txt
pip install -e ./verl
```

Notes:
- `select/inference_ray_batch.py` requires `ray>=2.44.1`.
- Multi-GPU analysis/training depends on your local CUDA, NCCL, and vLLM stack.

## 2) Configuration

### 2.1 Edit `automated/config.py`

Set model aliases and data roots:

```python
MODELS = {
    "qwen2.5-1.5b-math": "/path/to/Qwen2.5-1.5B-Math-Instruct",
    "qwen3-8b-base": "/path/to/Qwen3-8B-Base",
}

BASE_DATA_DIR = "/path/to/data"
BASE_RESPONSES_DIR = "/path/to/responses"
```

### 2.2 Configure the model-judge reward

`verl/verl/utils/reward_score/model_reward.py` contains the async model-judge path used by this codebase. Configure your API client/service there before running experiments that enable model judging.

### 2.3 Experiment inputs

Use `--selection_reference amc22` to guide gradient-based selection and `--heldout_eval amc23` for evaluation.
The runner checks normalized user prompts for holdout overlap and records the resolved datasets and settings in
`experiment.json`. Use a separate prefix for each method and seed. For legacy runs without a manifest,
`--adopt_legacy_run` records the settings you supply after you verify the original run.
Rollout caches carry a fingerprint of the selection grader. An unmarked cache or a changed grader
triggers fresh rollouts for the affected selection round. Use a fresh prefix for comparisons with runs
whose earlier training steps used the old grades.
The launcher resolves the bundled veRL configuration and reward paths relative to this checkout.

From the repository root, run either arm of the 16-training-rollout example with the same YAML.
Use the actual number of dedicated GPUs for the timing ledger.

```bash
python automated/dynamic_selection.py --config configs/webinstruct_math.yaml --allocated_gpus 8
python automated/dynamic_selection.py --config configs/webinstruct_math.yaml --mode rand_in_chunk --prefix webinstruct_rand_s42 --allocated_gpus 8
```

To add a baseline, register its choice function and required signals in `automated/select_data.py`,
then select it with `mode` in a new experiment YAML. The shared runner writes its selected Parquet
and sends it to the same training launcher and held-out evaluation. See [REPO_CONTRACT.md](REPO_CONTRACT.md).

## 3) Data preparation

Prepare datasets to unified VERL-style records (`train.jsonl`, `train.parquet`, `val.jsonl`, `val.parquet`):

```bash
cd automated
python prepare_data.py --dataset dapo
python prepare_data.py --dataset webinstruct
python prepare_data.py --dataset amc22
python prepare_data.py --dataset amc23
python prepare_data.py --dataset aime
```

Supported dataset keys in `prepare_data.py` include:
`aime24`, `aime25`, `aime`, `math`, `metamath`, `strategyqa`, `webinstruct`, `deepscaler`, `mmlupro`, `gsm8k`, `theoremqa`, `supergpqa`, `amc22`, `amc23`, `math500`, `dapo`.

Optional dataset mixing:

```bash
cd automated
python mix.py \
  --datasets dapo webinstruct \
  --nums 10000 20000 \
  --ls 0 0 \
  --output_name dapo_webinstruct_mix
```

## 4) Run GradAlign end-to-end

`dynamic_selection.py` orchestrates:
1. chunk candidate data,
2. rollout generation,
3. gradient alignment analysis,
4. top-k selection,
5. GRPO training on selected subsets.

Example:

```bash
cd automated
python dynamic_selection.py \
  --prefix gradalign_exp \
  --model qwen3-8b-base \
  --train_dataset webinstruct \
  --selection_reference amc22 \
  --chunk_size 5120 \
  --k 20 \
  --mode sim \
  --num_selections 100 \
  --train_batch_size 128 \
  --iters_per_select 10 \
  --n_samples_train 128 \
  --n_samples_val 16 \
  --rollout_n 16 \
  --minibatch_size 8 \
  --max_tokens 3072 \
  --max_model_len 4096 \
  --ckpt_root /path/to/checkpoints \
  --heldout_eval amc23
```

Important arguments:
- `--chunk_size`: number of candidate problems per round (`M` in paper).
- `--k`: selection ratio (`q` in paper), selecting `chunk_size / k` each round.
- `--mode`: selection policy (`sim`, `dot`, `accgreedy`, `align`, `rand`, ...).
- `--n_samples_train`, `--n_samples_val`: rollout counts used in gradient estimation (`k_r`, `k_v`).
- `--iters_per_select`: GRPO update steps between selection rounds.
- `--num_selections`: number of selection rounds.

The round selects up to `floor(eligible candidate count / k)` prompts. The selected data must fill at
least one training batch. The sample above selects 256 of 5,120 candidates and takes 10 GRPO steps of
128 prompts each. With all selected prompts usable, that is five passes over those 256 prompts.
Each training appearance generates 16 fresh answers. In this example, the 128 candidate answers per
question are a separate expense. The YAML example uses 16 candidate answers per question.
The source is shuffled once with the chunk seed, drops the incomplete tail when full chunks exist,
and cycles through those chunks, so a question can be selected again on later visits.

## 5) Baselines

Random baseline through the same loop:

```bash
cd automated
python dynamic_selection.py \
  --prefix random_baseline \
  --model qwen3-8b-base \
  --train_dataset webinstruct \
  --selection_reference amc22 \
  --chunk_size 5120 \
  --k 20 \
  --mode rand_in_chunk \
  --num_selections 100 \
  --train_batch_size 128 \
  --iters_per_select 10 \
  --rollout_n 16 \
  --max_tokens 3072 \
  --ckpt_root /path/to/checkpoints \
  --heldout_eval amc23
```

Direct GRPO training without data selection:

```bash
cd automated
python launch_verl_training.py \
  --prefix grpo_baseline \
  --model qwen3-8b-base \
  --train_dataset webinstruct \
  --heldout_eval amc23 \
  --train_batch_size 128 \
  --max_response_length 3072 \
  --rollout_n 16 --test_freq 5 --rollout_val_n 16 \
  --total_training_steps 1000 --total_epochs 1000 \
  --ckpt_root /path/to/checkpoints --allocated_gpus 8
```

## 6) Export merged model

`merge_model.py` merges checkpoints from:
`<ckpt_root>/<experiment_name>/global_step_<step>/actor`

Manual merge example:

```bash
cd automated
python merge_model.py \
  --experiment_name <experiment_name> \
  --step <global_step> \
  --output_model_name <merged_name> \
  --backend megatron \
  --ckpt_root /path/to/checkpoints \
  --dest_root /path/to/merged_models
```

## 7) Output artifacts

Typical artifacts under `ckpt_root/<experiment_name>/`:
- `chunks/chunk_*/train.jsonl` and `chunks/chunk_*/train.parquet`: chunked candidate pools.
- `global_step_*/train_split/part_0/responses.json` and `responses_sorted.json`: train rollouts.
- `global_step_*/train_split/part_0/similarity_results_cosine_real.jsonl`: one score per candidate problem.
- `global_step_*/train_split/similarity_results_aggregated.jsonl`: per-problem aggregated similarity.
- `global_step_*/train_split/accuracy_by_problem.jsonl`: per-problem accuracy (and similarity when available).
- `global_step_*/val_responses/responses.json` and `responses_sorted.json`: validation rollouts (for similarity-based modes).
- `global_step_*/merged/`: merged model when the actor checkpoint lacks a complete Hugging Face export.
- `selected/iter_<i>_<mode>_<n>/train.jsonl` and `train.parquet`: selected training subset for that iteration.
- `global_step_*/actor/` (from VERL): training checkpoint shards used as merge input.
- `global_step_*/gradalign_complete.json` records a verified round boundary.
- `timings.jsonl` records chunk preparation, selection (including candidate and reference work),
  and GRPO with held-out evaluation. GPU-hours equal wall seconds times the declared dedicated GPU count
  divided by 3,600. The cumulative value includes recorded failed attempts. A hard-killed process,
  time between stages and remote judge costs have no recorded duration in this ledger.

With a positive `--test_freq`, online runs evaluate before the initial update, at interval multiples,
and at the end of each round. A boundary that is also an interval multiple triggers one evaluation.
For 100 uninterrupted rounds of 10 steps with frequency 5 this gives 201 evaluation calls including step 0.

## Citation

```bibtex
@article{yang2026gradalign,
  title={GradAlign: Gradient-Aligned Data Selection for LLM Reinforcement Learning},
  author={Yang, Ningyuan and Du, Weihua and Sun, Weiwei and Welleck, Sean and Yang, Yiming},
  journal={arXiv preprint arXiv:2602.21492},
  year={2026}
}
```

## Acknowledgment

This repository is built on top of VERL (`verl/`) and includes project-specific modifications for GradAlign data selection and analysis.

## Four-GPU Oracle routing pilot

The three recipes share the frozen `Qwen/Qwen2.5-Math-1.5B-Instruct` base, two rank-4 LoRA adapters on `q_proj` and `v_proj`, FSDP1, and the existing math reward. Use a single-node CUDA environment with the checkout's veRL, Ray Data LLM, Accelerate and synchronous vLLM LoRA support. Install this checkout with `python -m pip install --no-deps -e ./verl` in that environment. The sync vLLM backend uses the V0 engine interface.

| Recipe | Data used for ten GRPO steps |
| --- | --- |
| `configs/graft_a_hard.yaml` | Highest 16 of 80 by reference-gradient similarity, vanilla GRPO |
| `configs/graft_b_all.yaml` | Same pool of 80, vanilla GRPO, skips the Oracle |
| `configs/graft_c_route.yaml` | All 80, with 16 retain-only, 48 both, 16 forget-only, forget-only gradient scale 2 |

The prepared pilot datasets ship with this branch. To use them offline, set `GRADALIGN_DATA_ROOT="$PWD/test_data"` at the repository root. An existing Docker checkout can use the standalone archive containing the same data and one-step launch scripts. [Offline pilot data and download](test_data/README.md)

Mount the same safetensors base snapshot for all arms (model revision `aafeb0fc6f22cbf0eaeed126eff8be45b0360a35` was used to verify tokenization). The preparation command pins the upstream WebInstruct and AMC revisions, records source IDs, checks the chat prompt length, and writes three disjoint datasets in a dedicated directory. It also accepts the original parquet files via `--web_parquet` and `--amc_parquet` for an offline machine.

```bash
export MODEL_DIR=/models/Qwen2.5-Math-1.5B-Instruct
export GRADALIGN_DATA_ROOT=/data/gradalign-graft-pilot
export CKPT_DIR=/checkpoints/gradalign-graft-pilot
python automated/prepare_graft_pilot.py --model_path "$MODEL_DIR"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export RAY_ADDRESS=local
export VLLM_USE_V1=0
(
    set -e
    for arm in a_hard b_all c_route
    do
        python automated/dynamic_selection.py --config "configs/graft_${arm}.yaml" \
            --model_path "$MODEL_DIR" --ckpt_root "$CKPT_DIR"
    done
)
```

Each arm uses 8 prompts per step and 16 fresh answers per prompt. The Oracle uses 4 answers per candidate and 8 per AMC22 reference prompt. Greedy pass@1 is measured on all 40 AMC23 prompts at steps 0, 5 and 10. In an 8-GPU container, parallel jobs can use distinct four-device visibility masks and their own local Ray instances. Give each arm/seed a distinct prefix and share the checkpoint root to reuse round-zero Oracle scores.

- `selected/iter_*/train.parquet` carries the actual training prompts. C also writes `routes.json` with group sizes and the fraction of candidate groups with constant rewards.
- `heldout_metrics.jsonl` records `val_full`, C's `val_retain_only`, GRPO steps and cumulative allocated GPU hours. The standalone estimate replaces a shared Oracle wait or cache hit with its measured build cost. `timings.jsonl` includes final export time. Training metrics include the response fraction and the global pre-clip parameter-gradient norm contributed by each route.
- `global_step_*/actor/` stores sharded model and optimizer state for resuming. At a round boundary `merged/full/` is a standard HF policy for the next Oracle, and C also exports `merged/retain_only/`. Evaluation packs an inference copy of the adapters and leaves the training parameters active. Later rounds score with their own full policies.

To check the actual GPU environment first, run C with the same recipe and append `--prefix graft_gpu_check --iters_per_select 1 --num_selections 2 --test_freq 1`. This exercises one checkpoint handoff, the second Oracle, and both evaluation states. The CPU suite and its exact command are in [REPO_CONTRACT.md](REPO_CONTRACT.md).
