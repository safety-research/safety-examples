# from datasets import load_dataset
from datasets import load_dataset

from examples.capability_evals.multi_choice.dataset import Dataset
from safetytooling.safetytooling.data_models.dataset import DatasetQuestion


class ArcDataset(Dataset):
    def __init__(self, dataset_split: str = "test"):
        dataset = load_dataset("ai2_arc", "ARC-Challenge")
        self.dataset = dataset[dataset_split]

    def unpack_single(self, item: dict, index: int) -> DatasetQuestion:
        question = item["question"]

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


class TinyArcDataset(ArcDataset):
    def __init__(self, dataset_split: str = "test"):
        dataset = load_dataset("tinyBenchmarks/tinyAI2_arc", split=dataset_split)
        self.dataset = dataset.shuffle(seed=42)
