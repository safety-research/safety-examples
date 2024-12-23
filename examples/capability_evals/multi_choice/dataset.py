import json
import random
from abc import ABC, abstractmethod

from safetytooling.data_models.dataset import DatasetQuestion


class Dataset(ABC):
    def slice_dataset(self, limit: int = None, seed: int = 42):
        if limit is not None:
            limit = int(limit)
            orig_length = len(self.dataset)
            limit = min(limit, orig_length)

            # Shuffle the dataset
            random.seed(seed)
            shuffled_dataset = random.sample(self.dataset, len(self.dataset))

            # Select the first 'limit' items
            sliced_dataset = shuffled_dataset[:limit]

            print(f"Dataset size sliced to {len(sliced_dataset)}/{orig_length}")
            return sliced_dataset
        else:
            return self.dataset

    @abstractmethod
    def unpack_single(self, row: dict, index: int) -> DatasetQuestion:
        pass

    def unpack(self, dataset: list) -> list[DatasetQuestion]:
        return [self.unpack_single(row, i) for i, row in enumerate(dataset)]


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
            audio_file_path=item["audio_file_path"] if "audio_file_path" in item else None,
        )
