import dataclasses
import random
from pathlib import Path

import jsonlines
from safetytooling.data_models.dataset import DatasetQuestion
from simple_parsing import ArgumentParser

from examples.capability_evals.multi_choice.dataset import SavedMultiChoiceDataset
from examples.capability_evals.multi_choice.datasets.aqua import AquaDataset
from examples.capability_evals.multi_choice.datasets.arc import (
    ArcDataset,
    TinyArcDataset,
)
from examples.capability_evals.multi_choice.datasets.commonsense import CommonsenseDataset
from examples.capability_evals.multi_choice.datasets.hellaswag import (
    HellaswagDataset,
    TinyHellaswagDataset,
)
from examples.capability_evals.multi_choice.datasets.logiqa import LogiqaDataset
from examples.capability_evals.multi_choice.datasets.ludwig import LudwigDataset
from examples.capability_evals.multi_choice.datasets.mmlu import (
    MMLUDataset,
    TinyMMLUDataset,
)
from examples.capability_evals.multi_choice.datasets.moral import MoralDataset
from examples.capability_evals.multi_choice.datasets.strategy import StrategyDataset
from examples.capability_evals.multi_choice.datasets.truthful import (
    TinyTruthfulDataset,
    TruthfulDataset,
)


@dataclasses.dataclass
class FormatConfig:
    output_file: Path  # specifies name of output file
    input_file: Path | None = None  # overwrites dataset if provided
    dataset: str | None = None  # available datasets in load.py
    seed: int = 42
    limit: int | None = None

    def __post_init__(self):
        if self.input_file is None and self.dataset is None:
            raise ValueError("Either input_file or dataset must be provided")


dataset_classes = {
    "aqua": AquaDataset,
    "arc": ArcDataset,
    "commonsense": CommonsenseDataset,
    "hellaswag": HellaswagDataset,
    "logiqa": LogiqaDataset,
    "ludwig": LudwigDataset,
    "mmlu": MMLUDataset,
    "moral": MoralDataset,
    "strategy": StrategyDataset,
    "truthful": TruthfulDataset,
    "tiny_mmlu": TinyMMLUDataset,
    "tiny_hellaswag": TinyHellaswagDataset,
    "tiny_truthful": TinyTruthfulDataset,
    "tiny_arc": TinyArcDataset,
}


def save_questions_to_disk(questions: list, input_file: str):
    path = Path(input_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    with jsonlines.open(path, "w") as writer:
        for question in questions:
            writer.write(question.model_dump())


def load_dataset_from_config(cfg: FormatConfig) -> list[DatasetQuestion]:
    assert cfg.dataset is not None

    if cfg.output_file is not None and Path(cfg.output_file).exists():
        return load_saved_dataset_from_config(cfg)

    random.seed(cfg.seed)
    if cfg.dataset in dataset_classes:
        dataset = dataset_classes[cfg.dataset]()
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")

    sliced_dataset = dataset.slice_dataset(limit=cfg.limit, seed=cfg.seed)
    questions = dataset.unpack(sliced_dataset)

    if cfg.output_file is not None:
        save_questions_to_disk(questions, cfg.output_file)

    return questions


def load_saved_dataset_from_config(cfg: FormatConfig) -> list[DatasetQuestion]:
    if cfg.input_file is None and cfg.dataset is None:
        raise ValueError("Either input_file or dataset must be provided")
    if cfg.input_file is None:
        cfg.input_file = f"/mnt/jailbreak-defense/exp/data/text_capability_evals/{cfg.dataset}.jsonl"

    dataset = SavedMultiChoiceDataset(cfg.input_file)
    sliced_dataset = dataset.slice_dataset(limit=cfg.limit, seed=cfg.seed)
    questions = dataset.unpack(sliced_dataset)
    return questions


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(FormatConfig, dest="load_config")
    args = parser.parse_args()
    cfg: FormatConfig = args.load_config
    load_dataset_from_config(cfg)
