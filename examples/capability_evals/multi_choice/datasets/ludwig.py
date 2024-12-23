from datasets import load_dataset
from safetytooling.data_models.dataset import DatasetQuestion

from examples.capability_evals.multi_choice.dataset import Dataset


class LudwigDataset(Dataset):
    def __init__(self, dataset_split: str = "validation"):
        dataset = load_dataset("UCL-DARK/ludwig", "5-shot")
        self.dataset = dataset[dataset_split]

    @staticmethod
    def raw_to_question(raw):
        base_question = f"""Does the following response to the question imply yes or no?
Question: {raw['utterance']}
Response: {raw['response']}
"""
        return base_question

    def unpack_single(self, item: dict, index: int) -> DatasetQuestion:
        question = self.raw_to_question(item)
        correct_answer = item["implicature"]
        incorrect_answers = [item["incoherent_implicature"]]

        return DatasetQuestion(
            question_id=index,
            question=question,
            incorrect_answers=incorrect_answers,
            correct_answer=correct_answer,
        )
