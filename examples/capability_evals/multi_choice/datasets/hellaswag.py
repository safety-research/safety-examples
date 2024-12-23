from datasets import load_dataset

from examples.capability_evals.multi_choice.dataset import Dataset
from safetytooling.safetytooling.data_models.dataset import DatasetQuestion


class HellaswagDataset(Dataset):
    def __init__(self, dataset_split: str = "validation"):
        dataset = load_dataset("hellaswag")
        self.dataset = dataset[dataset_split]

    @staticmethod
    def raw_to_question(raw):
        base_question = f"""Which is the most natural completion of the following sentence?
    Sentence: {raw}
    Choose the best answer from the following options:
    """
        return base_question

    def unpack_single(self, item: dict, index: int) -> DatasetQuestion:
        question = self.raw_to_question(item["ctx"])
        if item["label"].isalpha():
            answer_key = ord(item["label"]) - ord("A")
        else:
            answer_key = int(item["label"])
        correct_answer = item["endings"][answer_key]

        incorrect_answers = [item["endings"][i] for i in range(len(item["endings"])) if i != answer_key]

        if correct_answer in incorrect_answers:
            incorrect_answers.remove(correct_answer)

        return DatasetQuestion(
            question_id=index,
            question=question,
            incorrect_answers=incorrect_answers,
            correct_answer=correct_answer,
        )


class TinyHellaswagDataset(HellaswagDataset):
    def __init__(self, dataset_split: str = "validation"):
        dataset = load_dataset("tinyBenchmarks/tinyHellaswag", split=dataset_split)
        self.dataset = dataset.shuffle(seed=42)
