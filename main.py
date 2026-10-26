import hydra
from omegaconf import DictConfig, OmegaConf

from data_loading.pipeline import run
from utils.schema.config import DataLoadingConfig


@hydra.main(version_base=None, config_path="config/data_loading", config_name="default")
def main(cfg: DictConfig) -> None:
    schema = OmegaConf.structured(DataLoadingConfig)
    # Hydra injects a reserved `hydra` node; strip it before merging into the schema.
    user_cfg = OmegaConf.create({k: v for k, v in cfg.items() if k != "hydra"})
    config = OmegaConf.to_object(OmegaConf.merge(schema, user_cfg))
    run(config)


if __name__ == "__main__":
    main()
