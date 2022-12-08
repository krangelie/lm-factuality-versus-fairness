import sys
import hydra
from omegaconf import DictConfig

from stereoset import experiment, stereoset_evaluation


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    if cfg.benchmark.name == "stereoset":
        if cfg.benchmark.mode == "experiment":
            experiment.run_experiment(cfg)
        elif cfg.benchmark.mode == "evaluation":
            stereoset_evaluation.run_evaluation(cfg)
        else:
            sys.exit(f"Undefined mode: {cfg.benchmark.mode}")
    else:
        sys.exit(f"Undefined benchmark: {cfg.benchmark.name}")


if __name__ == "__main__":
    main()