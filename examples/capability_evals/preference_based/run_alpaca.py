from __future__ import annotations

import asyncio
import dataclasses
import logging
from pathlib import Path

from safetytooling.utils import utils
from safetytooling.utils.experiment_utils import ExperimentConfigBase
from simple_parsing import ArgumentParser

from examples.capability_evals.preference_based import alpaca

LOGGER = logging.getLogger(__name__)


async def get_completions(
    cfg: ExperimentConfig,
    model_id: str,
    save_file: str,
):
    LOGGER.info(f"Getting completions on AlpacaEval dataset for {model_id}...")
    completions = await alpaca.get_completions_on_alpaca(
        model_id=model_id,
        progress_bar=True,
        n_rows=cfg.n_rows,
        api=cfg.api,
    )
    LOGGER.info("Got completions.")
    LOGGER.info(f"Total cost so far: {cfg.api.running_cost} USD")
    utils.save_jsonl(
        file_path=cfg.get_exp_subdir() / save_file,
        data=[x.model_dump() for x in completions],
    )
    cfg.log_api_cost(metadata=dict(model=model_id, purpose="alpaca-completions"))

    return completions


async def main(cfg: ExperimentConfig):
    completions_1 = await get_completions(
        cfg=cfg,
        model_id=cfg.model_id_1,
        save_file="alpaca-completions-1.jsonl",
    )
    completions_2 = await get_completions(
        cfg=cfg,
        model_id=cfg.model_id_2,
        save_file="alpaca-completions-2.jsonl",
    )

    LOGGER.info("Generating comparisons...")
    alpaca_comparisons = await alpaca.compare_all_completions(
        completions_1=completions_1,
        completions_2=completions_2,
        comparison_model_id=cfg.comparison_model_id,
        seed=cfg.seed,
        progress_bar=True,
        randomize_order=True,
        api=cfg.api,
    )
    LOGGER.info("Got comparisons.")
    LOGGER.info(f"Total cost so far: {cfg.api.running_cost} USD")
    utils.save_jsonl(
        file_path=cfg.get_exp_subdir() / "alpaca-comparisons.jsonl",
        data=[x.model_dump() for x in alpaca_comparisons],
    )
    cfg.log_api_cost(metadata=dict(model=cfg.comparison_model_id, purpose="alpaca-comparisons"))

    LOGGER.info("Finished experiment.")


@dataclasses.dataclass(kw_only=True)
class ExperimentConfig(ExperimentConfigBase):
    exp_name: str
    model_id_2: str
    model_id_1: str

    comparison_model_id: str = "gpt-4o"

    seed: int = 42
    n_rows: int | None = None  # Number of rows to process

    def get_exp_subdir(self) -> Path:
        return self.output_dir / self.exp_name


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(ExperimentConfig, dest="experiment_config")
    args = parser.parse_args()
    cfg: ExperimentConfig = args.experiment_config

    cfg.setup_experiment(log_file_prefix=cfg.exp_name)
    asyncio.run(main(cfg))
