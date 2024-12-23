from datasets import load_dataset
from safetytooling.data_models.dataset import DatasetQuestion

from examples.capability_evals.multi_choice.dataset import Dataset


class TruthfulDataset(Dataset):
    def __init__(self, dataset_split: str = "validation"):
        dataset = load_dataset("EleutherAI/truthful_qa_mc", "multiple_choice")
        self.dataset = dataset[dataset_split]

    @staticmethod
    def raw_to_question(raw):
        sentence = raw["question"].rstrip("\n")
        base_question = f"""{sentence}"""
        return base_question

    def unpack_single(self, item: dict, index: int) -> DatasetQuestion:
        question = self.raw_to_question(item)
        answer_key = item["label"]
        correct_answer = item["choices"][answer_key]

        incorrect_answers = [item["choices"][i] for i in range(len(item["choices"])) if i != answer_key]

        if correct_answer in incorrect_answers:
            incorrect_answers.remove(correct_answer)

        return DatasetQuestion(
            question_id=index,
            question=question,
            incorrect_answers=incorrect_answers,
            correct_answer=correct_answer,
        )


class TinyTruthfulDataset(TruthfulDataset):
    def __init__(self, dataset_split: str = "validation"):
        self.dataset = load_dataset("tinyBenchmarks/tinyTruthfulQA", split=dataset_split)
        self.dataset = self.dataset.shuffle(seed=42)

    def unpack_single(self, item: dict, index: int) -> DatasetQuestion:
        question = self.raw_to_question(item)

        # Use mc2_targets for more comprehensive choices
        choices = item["mc2_targets"]["choices"]
        labels = item["mc2_targets"]["labels"]

        correct_answers = [choice for choice, label in zip(choices, labels) if label == 1]
        incorrect_answers = [choice for choice, label in zip(choices, labels) if label == 0]

        # If there are multiple correct answers, join them
        correct_answer = " / ".join(correct_answers)

        return DatasetQuestion(
            question_id=index,
            question=question,
            incorrect_answers=incorrect_answers,
            correct_answer=correct_answer,
        )
