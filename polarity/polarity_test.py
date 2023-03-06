import os

import evaluate


def run_toxicity_test(generated_texts):
    toxicity = evaluate.load("toxicity")
    toxicity_ratio = toxicity.compute(predictions=generated_texts, aggregation="ratio")
    print("Toxicity ratio:", toxicity_ratio)
    return toxicity_ratio


def run_regard_test(generated_texts_group_1, generated_texts_group_2=None, aggregation=None):
    if generated_texts_group_2 is None:
        regard = evaluate.load("regard")
        results = regard.compute(data=generated_texts_group_1)
        for d in results['regard']:
            print("Regard scores:", {l['label']: round(l['score'], 2) for l in d})
    else:
        # aggregation = None: Compute regard difference; aggregation = average: avg. per group
        regard = evaluate.load("regard", "compare")
        results = regard.compute(data=generated_texts_group_1,
                                 references=generated_texts_group_2,
                                 aggregation=aggregation)
        print("Regard differences:", {k: round(v, 2) for k, v in results['regard_difference'].items()})

        return results


