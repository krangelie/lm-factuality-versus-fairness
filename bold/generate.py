import os
import json
import csv

import transformers
import hydra


def generate_from_prompt_dict(cfg, model, tokenizer, input_path):
    with open(input_path) as jsonfile:
        prompts_dict = json.load(jsonfile)
    category = os.path.basename(input_path).split('_prompt')[0]
    print(f"Generate sentences for category {category}.")

    if cfg.benchmark.output_path != "":
        out_path = hydra.utils.to_absolute_path(cfg.benchmark.output_path)
    else:
        out_path = f"generated_sentences/{cfg.model.name}"

    for sub_category, prompts in prompts_dict.items():
        print(f"Processing subcategory {sub_category} ...")
        new_subdir = os.path.join(out_path, "by_category", category)
        os.makedirs(new_subdir, exist_ok=True)

        with open(os.path.join(new_subdir, f"{sub_category}.tsv"), 'w') as outfile:
            tsv_writer = csv.writer(outfile, delimiter='\t')
            for prompt_list in prompts.values():
                for i, prompt in enumerate(prompt_list):
                    input_ids = tokenizer.encode(prompt, return_tensors="pt")
                    generated_sentences = model.generate(
                        input_ids,
                        do_sample=cfg.benchmark.do_sample,
                        max_length=cfg.benchmark.max_length,
                        top_p=cfg.benchmark.top_p,
                        top_k=cfg.benchmark.top_k,
                        temperature=cfg.benchmark.temperature,
                        num_return_sequences=cfg.benchmark.num_return_sequences,
                    )
                    for sentence in generated_sentences:
                        sentence = tokenizer.decode(sentence)
                        print(sentence)
                        sentence = sentence.replace("<s>", "").replace("</s>", "").replace("\n", "")
                        tsv_writer.writerow([sentence])

def generate_bold_sentences(cfg):
    model = transformers.AutoModelForCausalLM.from_pretrained(cfg.model.model_name_or_path)
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(cfg.model.hf_path)

    if cfg.benchmark.name == "bold":
        print("Generate sentence with BOLD prompts.")
        for category in cfg.benchmark.prompt_categories:
            input_path = hydra.utils.to_absolute_path(os.path.join(cfg.benchmark.prompt_path, f"{category}_prompt.json"))
            generate_from_prompt_dict(cfg, model, tokenizer, input_path)
