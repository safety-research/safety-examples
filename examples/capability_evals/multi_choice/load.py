import logging
from pathlib import Path

from safetytooling.data_models.dataset import DatasetQuestion
from safetytooling.utils import utils

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

LOGGER = logging.getLogger(__name__)


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


def load_dataset_from_config(dataset: str, path_to_dataset: Path | None = None) -> list[DatasetQuestion]:
    if path_to_dataset is not None and Path(path_to_dataset).exists():
        dataset = SavedMultiChoiceDataset(path_to_dataset)
    else:
        if dataset in dataset_classes:
            dataset = dataset_classes[dataset]()
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

    questions = dataset.convert_to_questions(dataset.dataset)

    if path_to_dataset is not None:
        path_to_dataset.parent.mkdir(parents=True, exist_ok=True)
        questions = [question.model_dump() for question in questions]
        utils.save_jsonl(path_to_dataset, questions)

    return questions
