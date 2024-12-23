import logging

from datasets import concatenate_datasets, load_dataset
from safetytooling.data_models.dataset import DatasetQuestion

from examples.capability_evals.multi_choice.dataset import Dataset

LOGGER = logging.getLogger(__name__)

TOPICS = [
    # "abstract_algebra",
    # "anatomy",
    # "astronomy",
    # "business_ethics",
    # "clinical_knowledge",
    # "college_biology",
    # "college_chemistry",
    # "college_computer_science",
    # "college_mathematics",
    # "college_medicine",
    # "college_physics",
    # "computer_security",
    # "conceptual_physics",
    # "econometrics",
    # "electrical_engineering",
    # "elementary_mathematics",
    # "formal_logic",
    # "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    # "human_aging",
    # "human_sexuality",
    # "international_law",
    # "jurisprudence",
    # "logical_fallacies",
    # "machine_learning",
    # "management",
    # "marketing",
    # "medical_genetics",
    # "miscellaneous",
    # "moral_disputes",
    # "moral_scenarios",
    # "nutrition",
    # "philosophy",
    # "prehistory",
    # "professional_accounting",
    # "professional_law",
    # "professional_medicine",
    # "professional_psychology",
    # "public_relations",
    # "security_studies",
    # "sociology",
    # "us_foreign_policy",
    # "virology",
    # "world_religions",
]


class MMLUDataset(Dataset):
    def __init__(
        self,
        dataset_split: str = "test",
        topics: list = TOPICS,
        topic_limit: int = 20,
    ):
        datasets = []
        for topic in topics:
            print(f"Loading MMLU topic: {topic}")
            dataset = load_dataset("tasksource/mmlu", topic)
            dataset = dataset[dataset_split]
            orig_length = len(dataset)
            topic_limit = min(topic_limit, orig_length)
            dataset_sliced = dataset.shuffle(seed=42).select(range(topic_limit))
            datasets.append(dataset_sliced)
        self.dataset = concatenate_datasets(datasets)

    @staticmethod
    def raw_to_question(raw: dict):
        question = raw["question"].rstrip("\n A:")
        base_question = f"""{question}"""
        return base_question

    def unpack_single(self, item: dict, index: int) -> DatasetQuestion:
        question = "Can you answer the following question as accurately as possible?\n"
        question += self.raw_to_question(item)

        correct_key = int(item["answer"])
        correct_answer = item["choices"][correct_key]
        incorrect_answers = [c for c in item["choices"] if c != correct_answer]

        if correct_answer in incorrect_answers:
            incorrect_answers.remove(correct_answer)

        return DatasetQuestion(
            question_id=index,
            question=question,
            incorrect_answers=incorrect_answers,
            correct_answer=correct_answer,
        )


class TinyMMLUDataset(MMLUDataset):
    def __init__(
        self,
        dataset_split: str = "test",
    ):

        LOGGER.info("Downloading tiny MMLU dataset and modifying")
        self.dataset = load_dataset("tinyBenchmarks/tinyMMLU", split=dataset_split)
        self.dataset = self.dataset.shuffle(seed=42)
