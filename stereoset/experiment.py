import json
import os

import transformers

from stereoset.stereoset import StereoSetRunner
from models import colake


thisdir = os.path.dirname(os.path.realpath(__file__))

def generate_experiment_id(
    name,
    model=None,
    model_name_or_path=None,
    bias_type=None,
    seed=None,
):
    experiment_id = f"{name}"

    # Build the experiment ID.
    if isinstance(model, str):
        experiment_id += f"_m-{model}"
    if isinstance(model_name_or_path, str):
        experiment_id += f"_c-{model_name_or_path.replace('/', '-')}"
    if isinstance(bias_type, str):
        experiment_id += f"_t-{bias_type}"
    if isinstance(seed, int):
        experiment_id += f"_s-{seed}"

    return experiment_id


def run_experiment(cfg):
    experiment_id = generate_experiment_id(
        name="stereoset",
        model=cfg.model.name,
        model_name_or_path=cfg.model.model_name_or_path,
        seed=cfg.benchmark.seed,
    )

    print("Running StereoSet:")
    print(f" - persistent_dir: {cfg.benchmark.persistent_dir}")
    print(f" - model: {cfg.model.model}")
    print(f" - model_name_or_path: {cfg.model.model_name_or_path}")
    print(f" - batch_size: {cfg.benchmark.batch_size}")
    print(f" - seed: {cfg.benchmark.seed}")

    if "luke" in cfg.model.model_name_or_path:
        model = transformers.LukeForMaskedLM.from_pretrained(cfg.model.model_name_or_path)
    elif "colake" in cfg.model.model_name_or_path:
        model = colake.ColakeForMaskedLM()
    else:
        model = transformers.AutoModelForMaskedLM.from_pretrained(cfg.model.model_name_or_path)
    model.eval()

    if "colake" in cfg.model.model_name_or_path:
        tokenizer = colake.ColakeTokenizer()
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(cfg.model.model_name_or_path)

    runner = StereoSetRunner(
        intrasentence_model=model,
        tokenizer=tokenizer,
        input_file=f"{cfg.benchmark.persistent_dir}/test.json",
        model_name_or_path=cfg.model.model_name_or_path,
        batch_size=cfg.benchmark.batch_size,
    )
    results = runner()

    os.makedirs(f"{cfg.benchmark.persistent_dir}/results/stereoset", exist_ok=True)
    with open(
        f"{cfg.benchmark.persistent_dir}/results/stereoset/{experiment_id}.json", "w"
    ) as f:
        json.dump(results, f, indent=2)


