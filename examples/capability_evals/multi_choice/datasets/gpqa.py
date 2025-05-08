# from datasets import load_dataset
from datasets import load_dataset
from safetytooling.data_models.dataset import DatasetQuestion

from examples.capability_evals.multi_choice.dataset import Dataset


class GpqaDataset(Dataset):
    def __init__(self, dataset_split: str = "train", dataset_subset: str = "gpqa_main"):
        dataset = load_dataset("Idavidrein/gpqa", dataset_subset, split=dataset_split)
        self.dataset = dataset

    def unpack_single(self, item: dict, index: int) -> DatasetQuestion:
        question = item["Question"]
        correct_answer = item["Correct Answer"]
        incorrect_answers = [item[f"Incorrect Answer {i}"] for i in [1, 2, 3]]

        if correct_answer in incorrect_answers:
            incorrect_answers.remove(correct_answer)

        return DatasetQuestion(
            question_id=index,
            question=question,
            incorrect_answers=incorrect_answers,
            correct_answer=correct_answer,
        )
