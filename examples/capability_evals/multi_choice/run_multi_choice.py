import asyncio
import dataclasses
import logging
import random
import traceback
from pathlib import Path
from typing import Any

from safetytooling.apis import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from safetytooling.utils import utils
from safetytooling.utils.experiment_utils import ExperimentConfigBase
from safetytooling.utils.prompt_utils import get_prompt_template
from simple_parsing import ArgumentParser
from tqdm.asyncio import tqdm_asyncio
from tqdm.auto import tqdm

from examples.capability_evals.multi_choice.load import load_dataset_from_config
from examples.capability_evals.multi_choice.score import ScoreConfig, get_accuracy

LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass
class ExperimentConfig(ExperimentConfigBase):
    dataset: str
    path_to_dataset: Path | None = None
    path_to_output: Path | None = None
    model: str = "gpt-3.5-turbo-1106"

    temperature: float = 0.0
    max_tokens: int = 10
    n_rows_to_process: int | None = None
    max_workers: int = 100
    prefill: bool = False


async def process_row(
    row_obj: dict[str, Any],
    api: InferenceAPI,
    cfg: ExperimentConfig,
    dataset: str,
    lock: asyncio.Lock,
) -> dict[str, Any]:
    try:
        question_id = row_obj["question_id"]
        question = row_obj["question"]
        correct_answer = row_obj["correct_answer"]
        incorrect_answers = row_obj["incorrect_answers"]
        assert correct_answer not in incorrect_answers, f"{correct_answer} in {incorrect_answers}"
        choices = [correct_answer] + incorrect_answers

        # must shuffle choices in a thread-safe way and with a consistent seed based on question_id
        async with lock:
            random.seed(question_id + cfg.seed)
            random.shuffle(choices)

        # put in form A: choice1 B: choice2 etc
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        letters = letters[: len(choices)]
        choices_str = [f"{letters[i]}) {choice}" for i, choice in enumerate(choices)]
        choices_str = "\n".join(choices_str)
        letters_str = ", ".join(letters)
        correct_letter = letters[choices.index(correct_answer)]

        system_template = get_prompt_template("capability-evals/system-prompt.jinja")
        user_template = get_prompt_template("capability-evals/user-prompt.jinja")
        system_prompt = system_template.render()
        user_prompt = user_template.render(
            question=question,
            choices=choices_str,
            letters=letters_str,
        )

        messages = [
            ChatMessage(role=MessageRole.system, content=system_prompt),
            ChatMessage(role=MessageRole.user, content=user_prompt),
        ]

        prompt = Prompt(messages=messages)

        if cfg.prefill:
            prompt = prompt.add_assistant_message("Answer: ")

        response = await api(
            model_id=cfg.model,
            prompt=prompt,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )
        answer = response[0].completion

        return {
            **row_obj,
            "success": True,
            "answer": answer,
            "choices": choices_str,
            "label": correct_letter,
            "letters": letters,
            "dataset": dataset,
            "model": cfg.model,
        }
    except RuntimeError:
        return {
            **row_obj,
            "success": False,
            "error_msg": traceback.format_exc(),
        }


async def main(cfg: ExperimentConfig):
    if cfg.path_to_dataset is not None:
        assert cfg.dataset in str(cfg.path_to_dataset), "dataset must be in path_to_dataset"
    else:
        cfg.path_to_dataset = cfg.output_dir / f"{cfg.dataset}.jsonl"

    LOGGER.info(f"Loading dataset {cfg.dataset}...")
    if cfg.path_to_dataset.exists():
        input_objs = utils.load_jsonl(cfg.path_to_dataset)
    else:
        input_objs = load_dataset_from_config(
            dataset=cfg.dataset,
            path_to_dataset=cfg.path_to_dataset,
        )
        input_objs = [obj.model_dump() for obj in input_objs]

    # shuffle and slice dataset
    random.seed(cfg.seed)
    input_objs = random.sample(input_objs, len(input_objs))
    input_objs = input_objs[: cfg.n_rows_to_process]

    incomplete_objs = [obj for obj in input_objs if not obj.get("success", False)]
    complete_objs = [obj for obj in input_objs if obj.get("success", False)]

    LOGGER.info(f"Constructing tasks {cfg.dataset} (incomplete {len(incomplete_objs)}/{len(input_objs)})...")
    lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(cfg.max_workers)

    async def process_with_semaphore(row_obj):
        async with semaphore:
            return await process_row(row_obj, cfg.api, cfg, cfg.dataset, lock)

    tasks = [process_with_semaphore(row_obj) for row_obj in tqdm(incomplete_objs)]

    LOGGER.info(f"Processing rows {cfg.dataset}...")
    output_objs = await tqdm_asyncio.gather(*tasks)
    output_objs = complete_objs + output_objs

    LOGGER.info(f"Running cost: ${cfg.api.running_cost:.3f}")

    LOGGER.info(f"Writing output {cfg.dataset} to {cfg.path_to_output}...")
    if cfg.path_to_output is None:
        cfg.path_to_output = cfg.output_dir / f"{cfg.dataset}-{cfg.model}.jsonl"
    utils.save_jsonl(cfg.path_to_output, output_objs)

    cfg.log_api_cost(metadata={"dataset": cfg.dataset, "model": cfg.model})

    get_accuracy(
        ScoreConfig(
            input=cfg.path_to_output,
            results_file=cfg.output_dir / "results.jsonl",
            verbose=True,
        )
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(ExperimentConfig, dest="experiment_config")
    args = parser.parse_args()
    cfg: ExperimentConfig = args.experiment_config

    cfg.setup_experiment(log_file_prefix="multi-choice-eval")
    asyncio.run(main(cfg))
