from datasets import load_dataset
from safetytooling.data_models.dataset import DatasetQuestion

from examples.capability_evals.multi_choice.dataset import Dataset


class LogiqaDataset(Dataset):
    def __init__(self, dataset_split: str = "validation"):
        dataset = load_dataset("lucasmccabe/logiqa")
        self.dataset = dataset[dataset_split]

    @staticmethod
    def raw_to_question(raw):
        context = raw["context"].rstrip("\n")
        question = raw["query"].rstrip("\n")
        base_question = f"""Context: {context}\nQuestion: {question}"""
        return base_question

    def unpack_single(self, item: dict, index: int) -> DatasetQuestion:
        question = self.raw_to_question(item)
        answer_key = item["correct_option"]
        correct_answer = item["options"][answer_key]
        candidates = item["options"]
        incorrect_answers = [candidates[i] for i in range(len(candidates)) if i != answer_key]

        while correct_answer in incorrect_answers:
            incorrect_answers.remove(correct_answer)

        return DatasetQuestion(
            question_id=index,
            question=question,
            incorrect_answers=incorrect_answers,
            correct_answer=correct_answer,
        )
