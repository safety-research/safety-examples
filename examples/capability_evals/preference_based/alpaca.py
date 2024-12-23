"""Convenience functions for evaluating models according to AlpacaEval."""

import asyncio
from typing import Awaitable, Callable, Iterable

import datasets
import numpy as np
import pydantic
import scipy.special
from safetytooling.apis import InferenceAPI
from safetytooling.apis.utils import binary_response_logit
from safetytooling.data_models import ChatMessage, LLMResponse, MessageRole, Prompt
from safetytooling.utils.prompt_utils import get_prompt_template
from tqdm.asyncio import tqdm_asyncio
from typing_extensions import Self


class AlpacaEvalCompletion(pydantic.BaseModel):
    dataset: str  # Name of the dataset the instruction is from
    instruction: str

    generator: str  # Model used to generate the output
    output: str


def get_gpt4_alpaca_eval_completions() -> list[AlpacaEvalCompletion]:
    eval_ds: datasets.Dataset = datasets.load_dataset(
        path="tatsu-lab/alpaca_eval",
        name="alpaca_eval_gpt4_baseline",
        revision="2955709b93b0e08df511a752a3874b0a89b8910b",
        trust_remote_code=True,
    )["eval"]

    return [AlpacaEvalCompletion.model_validate(row | dict(generator="gpt-4-1106-preview")) for row in eval_ds]


async def get_completions_on_alpaca(
    model_id: str,
    completion_fn: Callable[[str], Awaitable[str]] | None = None,
    api: InferenceAPI | None = None,
    progress_bar: bool = False,
    n_rows: int | None = None,
) -> list[AlpacaEvalCompletion]:
    """
    If completion_fn is None, will attempt to call model_id using api.
    Otherwise model_id serves only as a label.
    """
    if api is None:
        api = InferenceAPI.get_default_global_api()

    async def process_row(row: AlpacaEvalCompletion) -> AlpacaEvalCompletion:
        output: str
        if completion_fn is not None:
            output = await completion_fn(row.instruction)
        else:
            output = (await api.ask_single_question(model_ids=model_id, question=row.instruction))[0]

        return AlpacaEvalCompletion(
            dataset=row.dataset,
            instruction=row.instruction,
            generator=model_id,
            output=output,
        )

    ds = get_gpt4_alpaca_eval_completions()
    if n_rows is not None:
        ds = ds[:n_rows]

    async_lib = tqdm_asyncio if progress_bar else asyncio
    return await async_lib.gather(*[process_row(row) for row in ds])


class AlpacaEvalPairwiseComparison(pydantic.BaseModel):
    judge_summary: str
    judge_id: str

    model_id_1: str
    model_id_2: str
    output1_shown_first: bool = True

    # Probability that output1 is better than output2 (in range [0, 1])
    # None if the judge_summary was unable to be parsed.
    prob_1_gt_2: float | None = None
    logit_1_gt_2: float | None = None

    model_config = pydantic.ConfigDict(protected_namespaces=())

    @classmethod
    def from_llm_response(
        cls,
        response: LLMResponse,
        judge_id: str,
        output1_tokens: Iterable[str],
        output2_tokens: Iterable[str],
        model_id_1: str,
        model_id_2: str,
        token_idx: int = -1,
    ) -> Self:
        """Requires that response has logprobs."""
        logit_1_gt_2 = binary_response_logit(
            response=response,
            tokens1=output1_tokens,
            tokens2=output2_tokens,
            token_idx=token_idx,
        )

        if logit_1_gt_2 is None:
            return cls(
                judge_summary=response.completion,
                judge_id=judge_id,
                model_id_1=model_id_1,
                model_id_2=model_id_2,
            )

        return cls(
            judge_summary=response.completion,
            judge_id=judge_id,
            model_id_1=model_id_1,
            model_id_2=model_id_2,
            prob_1_gt_2=scipy.special.expit(logit_1_gt_2),
            logit_1_gt_2=logit_1_gt_2,
        )

    def flip_order(self) -> Self:
        return self.__class__(
            judge_summary=self.judge_summary,
            judge_id=self.judge_id,
            model_id_1=self.model_id_2,
            model_id_2=self.model_id_1,
            output1_shown_first=not self.output1_shown_first,
            prob_1_gt_2=None if self.prob_1_gt_2 is None else 1 - self.prob_1_gt_2,
            logit_1_gt_2=None if self.logit_1_gt_2 is None else -self.logit_1_gt_2,
        )


async def compare_via_alpaca_eval_template(
    instruction: str,
    output1: str,
    output2: str,
    model_id_1: str,
    model_id_2: str,
    judge_id: str = "gpt-4-1106-preview",
    max_tokens: int = 300,
    temperature: float = 1.0,
    api: InferenceAPI | None = None,
) -> AlpacaEvalPairwiseComparison:
    """
    The settings of model_id, max_tokens, and temperature are set to match
    the default settings here:
    https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/evaluators_configs/weighted_alpaca_eval_gpt4_turbo/configs.yaml

    By default output1 is shown to the grading model before output2.
    If swap_order is True, this is flipped.

    model_id_1 and model_id_2 are just used as labels.
    """
    if api is None:
        api = InferenceAPI.get_default_global_api()

    system_prompt = get_prompt_template("alpaca-eval/default-system-prompt.jinja").render()
    user_prompt = get_prompt_template("alpaca-eval/default-user-prompt.jinja").render(
        instruction=instruction, output_m=output1, output_M=output2
    )

    prompt = Prompt(
        messages=[
            ChatMessage(role=MessageRole.system, content=system_prompt),
            ChatMessage(
                role=MessageRole.user,
                content=user_prompt,
            ),
        ]
    )

    (judge_response,) = await api.__call__(
        model_ids=judge_id,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        logprobs=5,
    )

    return AlpacaEvalPairwiseComparison.from_llm_response(
        response=judge_response,
        judge_id=judge_id,
        model_id_1=model_id_1,
        model_id_2=model_id_2,
        output1_tokens=["m"],
        output2_tokens=["M"],
    )


async def compare_all_completions(
    completions_1: list[AlpacaEvalCompletion],
    completions_2: list[AlpacaEvalCompletion] | None = None,
    comparison_model_id: str = "gpt-4-1106-preview",
    randomize_order: bool = True,
    seed: int | None = None,
    progress_bar: bool = False,
    api: InferenceAPI | None = None,
) -> list[AlpacaEvalPairwiseComparison]:
    if completions_2 is None:
        completions_2 = get_gpt4_alpaca_eval_completions()

    assert len(completions_1) == len(completions_2)
    assert all(
        c1.dataset == c2.dataset and c1.instruction == c2.instruction for c1, c2 in zip(completions_1, completions_2)
    )

    rng = np.random.default_rng(seed)

    def should_swap() -> bool:
        if randomize_order:
            return rng.choice([False, True])
        return False

    async def _compare_pair(c1: AlpacaEvalCompletion, c2: AlpacaEvalCompletion, swap: bool):
        if swap:
            c1, c2 = c2, c1

        ret = await compare_via_alpaca_eval_template(
            judge_id=comparison_model_id,
            instruction=c1.instruction,
            model_id_1=c1.generator,
            model_id_2=c2.generator,
            output1=c1.output,
            output2=c2.output,
            api=api,
        )
        if swap:
            ret = ret.flip_order()

        return ret

    async_lib = tqdm_asyncio if progress_bar else asyncio
    return await async_lib.gather(
        *[_compare_pair(c1, c2, should_swap()) for c1, c2 in zip(completions_1, completions_2)]
    )
