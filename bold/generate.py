import os
import json
import logging
from tqdm import tqdm

import transformers
import hydra
import torch

logging.basicConfig(level=logging.ERROR)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def generate_from_prompt_dict(cfg, model, tokenizer, input_path):
    batch_size = cfg.benchmark.input_batch_size
    torch.manual_seed(0)

    with open(input_path) as jsonfile:
        prompts_dict = json.load(jsonfile)
    category = os.path.basename(input_path).split('_prompt')[0]
    print(f"Generate sentences for category {category}.")

    if cfg.benchmark.output_path != "":
        out_path = hydra.utils.to_absolute_path(cfg.benchmark.output_path)
    else:
        out_path = f"generated_sentences/{cfg.model.name}"

    for sub_category, sub_category_prompts in tqdm(prompts_dict.items()):
        print(f" - Processing subcategory {sub_category} ...")
        new_subdir = os.path.join(out_path, category)
        os.makedirs(new_subdir, exist_ok=True)
        output = []
        # combine prompt list per entity to combined list per sub category
        sub_category_prompt_list = [p for p_list in sub_category_prompts.values() for p in p_list ]

        start_idx = 0
        pbar = tqdm(desc="Batch processing prompts")
        while start_idx < len(sub_category_prompt_list):
            if start_idx + batch_size <= len(sub_category_prompt_list):
                batch = sub_category_prompt_list[start_idx:start_idx + batch_size]
                step_size = batch_size
            else:
                batch = sub_category_prompt_list[start_idx:]
                step_size = len(batch)

            inputs = tokenizer(batch, return_tensors="pt", padding=True)
            try:
                generated = model.generate(
                    input_ids=inputs["input_ids"].to(DEVICE),
                    attention_mask=inputs["attention_mask"].to(DEVICE),
                    do_sample=cfg.benchmark.do_sample,
                    max_new_tokens=cfg.benchmark.max_new_tokens,
                    top_p=cfg.benchmark.top_p,
                    top_k=cfg.benchmark.top_k,
                    temperature=cfg.benchmark.temperature,
                    #num_return_sequences=cfg.benchmark.num_return_sequences,
                )
            except:
                print(f"Item in category: {category}, subcategory: {sub_category} not tokenizable")
                output += ["None"]
            else:
                batch_out_sentence = tokenizer.batch_decode(generated, skip_special_tokens=True)
                batch_out_sentence = [s.replace("\n", "").replace("\t", "") for s in batch_out_sentence]
                output += batch_out_sentence
            pbar.update(step_size)
            start_idx += batch_size
        pbar.close()
        if len(output) != len(sub_category_prompt_list):
            print("Number of input prompts not equal to number of outputs.\nGenerated ", len(output), " Input ",
                  len(sub_category_prompt_list))
        print("\n--- Writing generated sentences to file.")
        with open(os.path.join(new_subdir, f"{sub_category}.txt"), 'w') as f:
            f.write("\n".join(output))


def generate_bold_sentences(cfg):
    model = transformers.AutoModelForCausalLM.from_pretrained(cfg.model.model_name_or_path)
    model.to(DEVICE)
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(cfg.model.tokenizer_path, padding_side="left")

    # Define PAD Token = EOS Token = 50256
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    if cfg.benchmark.name == "bold":
        print("Generate sentence with BOLD prompts.")
        for category in cfg.benchmark.prompt_categories:
            input_path = hydra.utils.to_absolute_path(os.path.join(cfg.benchmark.prompt_path, f"{category}_prompt.json"))
            generate_from_prompt_dict(cfg, model, tokenizer, input_path)
