import logging
import random

from safetytooling.utils import utils
from safetytooling.utils.prompt_utils import get_prompt_template
from simple_parsing import ArgumentParser
from tqdm import tqdm

from examples.capability_evals.multi_choice.load import FormatConfig, load_dataset_from_config

LOGGER = logging.getLogger(__name__)


def format_evaluation(cfg: FormatConfig):
    user_template = get_prompt_template("capability-evals/user-prompt.jinja")

    if cfg.input_file is not None:
        input_data = utils.load_jsonl(cfg.input_file)
        dataset = cfg.input_file.stem
    else:
        input_data = load_dataset_from_config(cfg)
        input_data = [question.model_dump() for question in input_data]
        dataset = cfg.dataset

    cfg.output_file.parent.mkdir(parents=True, exist_ok=True)

    random.seed(cfg.seed)

    assert cfg.text_type == "simple"

    formatted_data = []

    for row in tqdm(input_data, desc="Formatting rows"):
        question_id = row["question_id"]
        question = row["question"]
        correct_answer = row["correct_answer"]
        incorrect_answers = row["incorrect_answers"]

        choices = [correct_answer] + incorrect_answers
        random.shuffle(choices)

        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        letters = letters[: len(choices)]
        choices_str = "\n".join(f"Letter {letters[i]}) {choice}" for i, choice in enumerate(choices))
        letters_str = ", ".join(letters)
        correct_letter = letters[choices.index(correct_answer)]

        system_prompt = "Please answer the following question"
        user_prompt = user_template.render(
            question=question,
            choices=choices_str,
            letters=letters_str,
        )
        formatted_row = {
            "question_id": question_id,
            "question": question,
            "correct_answer": correct_answer,
            "incorrect_answers": incorrect_answers,
            "dataset": dataset,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "choices": choices_str,
            "label": correct_letter,
            "letters": letters,
        }

        formatted_data.append(formatted_row)

    # Save formatted data
    utils.save_jsonl(cfg.output_file, formatted_data)

    LOGGER.info(f"Formatted evaluation saved to {cfg.output_file}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = ArgumentParser()
    parser.add_arguments(FormatConfig, dest="format_config")
    args = parser.parse_args()
    cfg: FormatConfig = args.format_config

    format_evaluation(cfg)
