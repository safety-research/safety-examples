from datasets import load_dataset
from safetytooling.data_models.dataset import DatasetQuestion

from examples.capability_evals.multi_choice.dataset import Dataset


class MoralDataset(Dataset):
    def __init__(self, dataset_split: str = "validation"):
        dataset = load_dataset("tasksource/bigbench", "moral_permissibility")
        self.dataset = dataset[dataset_split]

    @staticmethod
    def raw_to_question(raw):
        question = raw["inputs"].rstrip("\n A:")
        base_question = f"""{question}"""
        return base_question

    def unpack_single(self, item: dict, index: int) -> DatasetQuestion:
        question = self.raw_to_question(item)
        correct_answer = item["targets"][0]
        incorrect_answers = [x for x in item["multiple_choice_targets"] if x != correct_answer]

        if correct_answer in incorrect_answers:
            incorrect_answers.remove(correct_answer)

        return DatasetQuestion(
            question_id=index,
            question=question,
            incorrect_answers=incorrect_answers,
            correct_answer=correct_answer,
        )
