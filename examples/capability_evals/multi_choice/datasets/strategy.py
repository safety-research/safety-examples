from datasets import load_dataset
from numpy import argmax
from safetytooling.data_models.dataset import DatasetQuestion

from examples.capability_evals.multi_choice.dataset import Dataset


class StrategyDataset(Dataset):
    def __init__(self, dataset_split: str = "validation"):
        dataset = load_dataset("tasksource/bigbench", "strategyqa")
        self.dataset = dataset[dataset_split]

    @staticmethod
    def raw_to_question(raw):
        sentence = raw["inputs"].rstrip("A:\n")
        base_question = f"""{sentence}"""
        return base_question

    def unpack_single(self, item: dict, index: int) -> DatasetQuestion:
        question = self.raw_to_question(item)

        answer_key = argmax(item["multiple_choice_scores"])
        correct_answer = item["multiple_choice_targets"][answer_key]
        incorrect_answers = [
            item["multiple_choice_targets"][i] for i in range(len(item["multiple_choice_targets"])) if i != answer_key
        ]

        if correct_answer in incorrect_answers:
            incorrect_answers.remove(correct_answer)

        return DatasetQuestion(
            question_id=index,
            question=question,
            incorrect_answers=incorrect_answers,
            correct_answer=correct_answer,
        )
