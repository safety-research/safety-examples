import json
from abc import ABC, abstractmethod

from safetytooling.data_models.dataset import DatasetQuestion


class Dataset(ABC):
    def __init__(self, dataset: list[dict]):
        self.dataset = dataset

    @abstractmethod
    def unpack_single(self, row: dict, index: int) -> DatasetQuestion:
        pass

    def convert_to_questions(self) -> list[DatasetQuestion]:
        return [self.unpack_single(row, i) for i, row in enumerate(self.dataset)]


class SavedMultiChoiceDataset(Dataset):
    def __init__(self, path_to_dataset: str):
        self.dataset = self.load_dataset_from_file(path_to_dataset)

    @staticmethod
    def load_dataset_from_file(file_path):
        with open(file_path, "r") as f:
            return [json.loads(line) for line in f]

    def unpack_single(self, item: dict, index: int) -> DatasetQuestion:
        return DatasetQuestion(
            question_id=index,
            question=item["question"],
            incorrect_answers=item["incorrect_answers"],
            correct_answer=item["correct_answer"],
        )
