import argparse
import json
import os

import transformers

from stereoset import StereoSetRunner


thisdir = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(description="Runs StereoSet benchmark.")
parser.add_argument(
    "--persistent_dir",
    action="store",
    type=str,
    default=os.path.realpath(os.path.join(thisdir, "..")),
    help="Directory where all persistent data will be stored.",
)
parser.add_argument(
    "--model",
    action="store",
    type=str,
    default="BertForMaskedLM",
    choices=[
        "BertForMaskedLM",
        "RobertaForMaskedLM",
        "LukeForMaskedLM",
    ],
    help="Model to evalute (e.g., BertForMaskedLM). Typically, these correspond to a HuggingFace "
    "class.",
)
parser.add_argument(
    "--model_name_or_path",
    action="store",
    type=str,
    default="bert-base-uncased",
    choices=["bert-base-uncased", "roberta-base", "studio-ousia/luke-base"],
    help="HuggingFace model name or path (e.g., bert-base-uncased). Checkpoint from which a "
    "model is instantiated.",
)
parser.add_argument(
    "--batch_size",
    action="store",
    type=int,
    default=1,
    help="The batch size to use during StereoSet intrasentence evaluation.",
)
parser.add_argument(
    "--seed",
    action="store",
    type=int,
    default=None,
    help="RNG seed. Used for logging in experiment ID.",
)

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
        experiment_id += f"_c-{model_name_or_path}"
    if isinstance(bias_type, str):
        experiment_id += f"_t-{bias_type}"
    if isinstance(seed, int):
        experiment_id += f"_s-{seed}"

    return experiment_id


if __name__ == "__main__":
    args = parser.parse_args()

    experiment_id = generate_experiment_id(
        name="stereoset",
        model=args.model,
        model_name_or_path=args.model_name_or_path,
        seed=args.seed,
    )

    print("Running StereoSet:")
    print(f" - persistent_dir: {args.persistent_dir}")
    print(f" - model: {args.model}")
    print(f" - model_name_or_path: {args.model_name_or_path}")
    print(f" - batch_size: {args.batch_size}")
    print(f" - seed: {args.seed}")

    model = transformers.AutoModel.from_pretrained(args.model_name_or_path)
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)

    runner = StereoSetRunner(
        intrasentence_model=model,
        tokenizer=tokenizer,
        input_file=f"{args.persistent_dir}/test.json",
        model_name_or_path=args.model_name_or_path,
        batch_size=args.batch_size,
    )
    results = runner()

    os.makedirs(f"{args.persistent_dir}/results/stereoset", exist_ok=True)
    with open(
        f"{args.persistent_dir}/results/stereoset/{experiment_id}.json", "w"
    ) as f:
        json.dump(results, f, indent=2)


