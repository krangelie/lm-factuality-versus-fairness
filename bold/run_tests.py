import os

from tqdm import tqdm

from polarity import polarity_test


##TODO: iterate through generations and compute toxicity and regard via polarity script

def read_texts_for_category(path_to_dir):
    category_dict = {}
    for text_file in tqdm(os.listdir(path_to_dir)):
        print(f"Reading {text_file}")
        with open(os.path.join(path_to_dir, text_file), "r") as f:
            category_dict[text_file.split(".")[0]] = [line.rstrip() for line in f]
    return category_dict


def run_polarity_tests(cfg):
    toxicity_results_overview, regard_results_overview = "", ""
    for category in cfg.benchmark.prompt_categories:
        print(f"Reading text generated for category {category}:")
        category_results_by_class = read_texts_for_category(os.path.join(cfg.benchmark.generated_texts_path, category))
        out_path = cfg.benchmark.test_results_path
        toxicity_results_overview += "=== " + category + " ===\n"
        regard_results_overview += "=== " + category + " ===\n"
        print("Running tests:")
        for category_class, generated_texts in tqdm(category_results_by_class.items()):
            print(f"\n{category_class}:")
            toxicity_ratio = polarity_test.run_toxicity_test(generated_texts)
            regard_scores = polarity_test.run_regard_test(generated_texts, aggregation=cfg.benchmark.regard_aggregation)
            toxicity_results_overview += f"{category_class}: {toxicity_ratio}\n"
            regard_results_overview += f"{category_class}: {regard_scores}\n"
        toxicity_results_overview += "\n\n"
        regard_results_overview += "\n\n"
        with open(os.path.join(out_path, f"{category}_toxicity_test_results.txt"), 'w') as f:
            f.write(toxicity_results_overview)
        with open(os.path.join(out_path, f"{category}_regard_test_results.txt"), 'w') as f:
            f.write(regard_results_overview)
