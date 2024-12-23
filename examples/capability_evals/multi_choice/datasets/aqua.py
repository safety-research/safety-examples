from datasets import load_dataset
from safetytooling.data_models.dataset import DatasetQuestion

from examples.capability_evals.multi_choice.dataset import Dataset


class AquaDataset(Dataset):
    def __init__(self, dataset_split: str = "test"):
        dataset = load_dataset("aqua_rat")
        self.dataset = dataset[dataset_split]

    @staticmethod
    def clean_answers(candidates: list):
        # remove suffix A), B), etc.
        return [c[2:] for c in candidates]

    def unpack_single(self, item: dict, index: int) -> DatasetQuestion:
        question = item["question"]
        item["options"] = self.clean_answers(item["options"])

        answer_key = ord(item["correct"]) - ord("A") if item["correct"].isalpha() else int(item["correct"]) - 1
        correct_answer = item["options"][answer_key]
        incorrect_answers = [item["options"][i] for i in range(len(item["options"])) if i != answer_key]

        if correct_answer in incorrect_answers:
            incorrect_answers.remove(correct_answer)

        return DatasetQuestion(
            question_id=index,
            question=question,
            incorrect_answers=incorrect_answers,
            correct_answer=correct_answer,
        )
