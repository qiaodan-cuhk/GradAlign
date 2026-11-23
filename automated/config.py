"""
Centralized configuration for model paths and common constants.

Users should update `MODELS` to reflect their local/remote model store.
"""

from typing import Dict


# Map a short, stable model key to its absolute model path.
# Update these paths to match your environment.
MODELS: Dict[str, str] = {
    # MODEL NAME: MODEL PATH
}


# Reward style used across datasets for consistency
REWARD_STYLE: str = "rule-lighteval/MATH_v2"


# Base directory for auto-selected dataset outputs
BASE_DATA_DIR: str = "/path/to/data" #SET YOUR OWN PATH

# Base directory for API responses
BASE_RESPONSES_DIR: str = "/path/to/responses" #SET YOUR OWN PATH


# Map model keys to short model type identifiers used in dataset paths
MODEL_TYPES: Dict[str, str] = {
    # Meta Llama
    # NAME: Model_type (e.g. qwen3: qwen)
}

def get_model_path(model_key: str) -> str:
    """Return the absolute model path for a given key, or raise with guidance."""
    if model_key not in MODELS:
        known = ", ".join(sorted(MODELS.keys()))
        raise KeyError(
            f"Unknown model key '{model_key}'. Known keys: {known}. "
            "Update config.MODELS if you need a new mapping."
        )
    return MODELS[model_key]


def get_model_type(model_key: str) -> str:
    """Return a short model type string for use in dataset paths.

    Falls back to simple heuristics if the key is not explicitly mapped.
    """
    if model_key in MODEL_TYPES:
        return MODEL_TYPES[model_key]
    key = model_key.lower()
    if "qwen" in key:
        return "qwen"
    if "llama" in key:
        return "llama"
    if "deepseek" in key:
        return "deepseek"
    if "mistral" in key or "mixtral" in key:
        return "mistral"
    return key.split("-")[0]


def get_dataset_dir(dataset_label: str, model_key: str = None) -> str:
    """Return dataset dir as {BASE_DATA_DIR}/{dataset}/{model_type}."""
    safe_dataset = dataset_label.replace("/", "-")
    return f"{BASE_DATA_DIR}/{safe_dataset}"


def get_response_dir(dataset_label: str, model_name: str) -> str:
    """Return responses dir as {BASE_RESPONSES_DIR}/{model_name}/{dataset}."""
    safe_dataset = dataset_label.replace("/", "-")
    safe_model_name = model_name.replace("/", "-")
    return f"{BASE_RESPONSES_DIR}/{safe_model_name}/{safe_dataset}"


# Backward-compat alias (to avoid breakage while migrating code)
get_output_dir = get_dataset_dir


def add_dataset_roles(parser):
    """The historical argument aliases remain valid in existing experiment scripts."""
    parser.add_argument("--selection_reference", "--val_dataset", dest="selection_reference",
                        help="Dataset used to estimate selection gradients")
    parser.add_argument("--heldout_eval", "--verl_val_set", dest="heldout_eval", required=True,
                        help="Disjoint dataset used to report model performance")
    parser.add_argument("--heldout_dir", help="Override the directory containing the evaluation Parquet")
    parser.add_argument("--heldout_parquet", default="train.parquet")


def parse_experiment_args(parser, argv=None):
    """Apply a flat YAML experiment through the same types and choices as the CLI."""
    import argparse
    import sys
    from pathlib import Path

    argv = sys.argv[1:] if argv is None else argv
    lookup = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    lookup.add_argument("--config")
    path = lookup.parse_known_args(argv)[0].config
    if path is None:
        return parser.parse_args(argv)

    import yaml

    try:
        values = yaml.safe_load(Path(path).expanduser().read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        parser.error(f"Cannot read experiment YAML {path}: {exc}")
    if not isinstance(values, dict):
        parser.error("Experiment YAML must be a mapping of argument names to values")
    actions = {action.dest: action for action in parser._actions if action.dest not in ("help", "config")}
    options = []
    for key, value in values.items():
        if key not in actions:
            parser.error(f"Unknown experiment setting: {key}")
        option = actions[key].option_strings[0]
        if actions[key].nargs == 0:
            if not isinstance(value, bool):
                parser.error(f"{key} expects a YAML boolean")
            if value:
                options.append(option)
        elif value is not None:
            if isinstance(value, bool) or not isinstance(value, (str, int, float)):
                parser.error(f"{key} expects a scalar value")
            options.append(f"{option}={value}")
    return parser.parse_args([*options, *argv])
