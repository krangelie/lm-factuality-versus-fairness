import sys
import hydra
from omegaconf import DictConfig

from stereoset.experiment import run_experiment
from stereoset.stereoset_evaluation import run_evaluation


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    if cfg.benchmark.name == "stereoset":
        if cfg.benchmark.mode == "experiment":
            run_experiment(cfg)
        elif cfg.benchmark.mode == "evaluation":
            run_evaluation(cfg)
        else:
            sys.exit(f"Undefined mode: {cfg.benchmark.mode}")
    else:
        sys.exit(f"Undefined benchmark: {cfg.benchmark.name}")


if __name__ == "__main__":
    main()