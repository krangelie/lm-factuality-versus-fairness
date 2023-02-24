import os

import evaluate


def run_toxicity_test(generated_texts, results_path=None):
    toxicity = evaluate.load("toxicity")
    toxicity_ratio = toxicity.compute(predictions=generated_texts, aggregation="ratio")
    print(toxicity_ratio)
    if results_path:
        with open(os.path.join(results_path, "toxicity_ratio.txt"), 'w') as f:
            f.write(toxicity_ratio)
    return toxicity_ratio


def run_regard_test(generated_texts_group_1, generated_texts_group_2=None, aggregation=None, results_path=None):
    if generated_texts_group_2 is None:
        regard = evaluate.load("regard")
        results = regard.compute(data=generated_texts_group_1)
        for d in results['regard']:
            print({l['label']: round(l['score'], 2) for l in d})
    else:
        # aggregation = None: Compute regard difference; aggregation = average: avg. per group
        regard = evaluate.load("regard", "compare")
        results = regard.compute(data=generated_texts_group_1,
                                 references=generated_texts_group_2,
                                 aggregation=aggregation)
        print({k: round(v, 2) for k, v in results['regard_difference'].items()})
        if results_path:
            with open(os.path.join(results_path, "regard_results.txt"), 'w') as f:
                f.write(results)
        return results


