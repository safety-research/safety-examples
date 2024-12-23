from datasets import load_dataset

from examples.capability_evals.multi_choice.dataset import Dataset
from safetytooling.safetytooling.data_models.dataset import DatasetQuestion


class CommonsenseDataset(Dataset):
    def __init__(self, dataset_split: str = "validation"):
        dataset = load_dataset("commonsense_qa")
        self.dataset = dataset[dataset_split]

    @staticmethod
    def raw_to_question(raw):
        sentence = raw["question"].rstrip("A:\n")
        base_question = f"""{sentence}"""
        return base_question

    def unpack_single(self, item: dict, index: int) -> DatasetQuestion:
        question = self.raw_to_question(item)

        answer_key = ord(item["answerKey"]) - ord("A") if item["answerKey"].isalpha() else int(item["answerKey"]) - 1
        correct_answer = item["choices"]["text"][answer_key]
        incorrect_answers = [item["choices"]["text"][i] for i in range(len(item["choices"]["text"])) if i != answer_key]

        if correct_answer in incorrect_answers:
            incorrect_answers.remove(correct_answer)

        return DatasetQuestion(
            question_id=index,
            question=question,
            incorrect_answers=incorrect_answers,
            correct_answer=correct_answer,
        )
