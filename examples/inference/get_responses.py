import asyncio
import dataclasses
import logging
import pathlib
import traceback

import pandas as pd
from simple_parsing import ArgumentParser
from tqdm.auto import tqdm

from safetytooling.safetytooling.apis import InferenceAPI
from safetytooling.safetytooling.utils import defence_utils, utils
from safetytooling.safetytooling.utils.experiment_utils import ExperimentConfigBase

LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass
class ExperimentConfig(ExperimentConfigBase):
    """Get responses to requests."""

    input_file: pathlib.Path
    model_ids: tuple[str, ...] = ("gpt-3.5-turbo-1106",)
    n_samples: int = 1
    temperature: float = 1.0
    max_tokens: int | None = None

    request_tag: str = "request"
    requests_tag: str = "requests"
    file_name: str = "responses.jsonl"


async def get_model_response(
    input_obj: dict,
    model_id: str,
    api: InferenceAPI,
    request_tag: str = "request",
    n_samples: int = 1,
    temperature: float = 1.0,
    max_tokens: int | None = None,
    add_punctuation: bool = False,
    print_prompt_and_response: bool = False,
) -> dict:
    new_data = {
        "temperature": temperature,
        "model_id": model_id,
    }
    # Get model response
    try:
        assert request_tag in input_obj, f"Tag {request_tag} not in input_obj"
        request = input_obj[request_tag]

        responses = await api.ask_single_question(
            model_ids=model_id,
            question=request,
            n=n_samples,
            temperature=temperature,
            max_tokens=max_tokens,
            print_prompt_and_response=print_prompt_and_response,
        )
        LOGGER.info(f"Running cost: ${api.running_cost:.3f}")

        output_obj = (
            input_obj
            | new_data
            | {
                "responses": responses,
            }
        )

        return output_obj

    except RuntimeError:
        LOGGER.error("Error processing input_obj")
        LOGGER.error(traceback.format_exc())

        output_obj = (
            input_obj
            | new_data
            | {
                "responses": None,
            }
        )

        return output_obj


def get_short_model_name(model_id: str) -> str:
    if "::" in model_id:
        return model_id.split("::")[1]

    if "-" in model_id:
        return model_id.replace("-turbo", "t").replace("-preview", "p")

    return model_id


async def main(
    cfg: ExperimentConfig,
    request_objs: list[dict] | None = None,
):
    if request_objs is None:
        request_objs = utils.load_jsonl(cfg.input_file)

    df = pd.DataFrame(request_objs)
    if cfg.requests_tag in df.columns:
        # Explode the requests_tag column into multiple rows
        # This will duplicate the other information in the row for each entry in requests_tag
        df = df.explode(cfg.requests_tag)
        df = df.rename(columns={cfg.requests_tag: cfg.request_tag})
    if "response" in df.columns:
        # some testsets have a response column so we remove it
        df = df.drop(columns="response")
    if cfg.filter_by_input_classifier:
        assert "input_classifier_output" in df.columns, "input_classifier_output not in df"
        df = defence_utils.process_input_classifier_output(df)

    # dedupe based on request_tag
    num_duplicates = df.duplicated(subset=[cfg.request_tag]).sum()
    if num_duplicates > 0:
        LOGGER.warning(f"Found {num_duplicates} duplicates in request data")
        df = df.drop_duplicates(subset=[cfg.request_tag])
    request_objs = df.to_dict("records")

    # Get responses
    responses: list[dict] = await tqdm.gather(
        *[
            get_model_response(
                input_obj=obj,
                model_id=model_id,
                api=cfg.api,
                request_tag=cfg.request_tag,
                n_samples=cfg.n_samples,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                print_prompt_and_response=cfg.print_prompt_and_response,
            )
            for obj in request_objs
            for model_id in cfg.model_ids
        ],
        desc=f"{cfg.input_file.stem} {','.join([get_short_model_name(x) for x in cfg.model_ids])}",
    )
    LOGGER.info("Writing attack responses...")
    utils.save_jsonl(cfg.output_dir / cfg.file_name, responses)

    LOGGER.info(f"Running cost: ${cfg.api.running_cost:.3f}")

    return responses


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(ExperimentConfig, dest="experiment_config")
    args = parser.parse_args()
    cfg: ExperimentConfig = args.experiment_config

    cfg.setup_experiment(log_file_prefix="get_responses")
    asyncio.run(main(cfg))
