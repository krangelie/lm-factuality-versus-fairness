import sys
import hydra
from omegaconf import DictConfig


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    if cfg.benchmark.name == "stereoset":
        from stereoset.experiment import run_experiment
        from stereoset.stereoset_evaluation import run_evaluation
        if cfg.benchmark.mode == "experiment":
            run_experiment(cfg)
        elif cfg.benchmark.mode == "evaluation":
            run_evaluation(cfg)
        else:
            sys.exit(f"Undefined mode: {cfg.benchmark.mode}")
    elif cfg.benchmark.name == "bold":
        from bold.generate import generate_bold_sentences
        from bold.run_tests import run_polarity_tests
        if cfg.benchmark.generate_bold_sentences:
            generate_bold_sentences(cfg)
        if cfg.benchmark.evaluate_bold_sentences:
            run_polarity_tests(cfg)
    elif cfg.benchmark.name == "lama":
        from lama.LAMA.scripts.run_experiments import run_lama
        run_lama(cfg)
    else:
        sys.exit(f"Undefined benchmark: {cfg.benchmark.name}")


if __name__ == "__main__":
    main()