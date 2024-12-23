import dataclasses
import pathlib
import re

import jsonlines
import pandas as pd
from simple_parsing import ArgumentParser


@dataclasses.dataclass
class ScoreConfig:
    input: pathlib.Path
    debug: bool = False
    verbose: bool = False
    results_file: pathlib.Path = None
    answer_key: str | None = None


def find_answer(text, letter):
    if text.strip() == letter:
        return True
    if f"Answer: {letter}" in text:
        return True
    pattern = re.compile(f"^(?:Final )?Answer: (<{letter}>|{letter})(?:\n|$)", re.MULTILINE)
    matches = pattern.findall(text)
    found_match = len(matches) > 0
    return found_match


def find_inconclusive(text):
    pattern = re.compile(
        r"^Answer: (Insufficient|Unable|Unclear|Can\'t determine|\?|Inconclusive|I do not have sufficient evidence).*?",
        re.MULTILINE,
    )
    matches = pattern.findall(text)
    found_match = len(matches) > 0
    return found_match


def func_correct(row, debug=False) -> tuple[bool, bool]:
    # returns (correct, unknown output)
    answer = row["answer"].strip()
    label = row["label"].strip()
    if find_answer(answer, label):
        return True, False

    # rest of code is for understanding if there are LLM outputs we don't expect
    if "letters" in row:
        for letter in row["letters"]:
            if label == letter:
                continue
            if find_answer(answer, letter):
                return False, False

    if find_inconclusive(answer):
        return False, False

    if debug:
        print(answer)
        print("======================\n\n")
        user_input = input("Do you want to see more? (y/n)")
        if user_input == "n":
            raise StopIteration("User stopped")

    return False, True


def get_complete_rows(df, complete_col, ensure_complete=True):
    if complete_col not in df.columns:
        raise ValueError(f"no complete column {complete_col}")

    len_before = len(df)
    df = df[df[complete_col]]
    len_after = len(df)

    if df.empty:
        raise ValueError("no complete rows")

    if len_before != len_after:
        warning_string = f"{len_before - len_after} incomplete"
        print(f"WARNING: {warning_string}")

        if ensure_complete:
            raise ValueError(warning_string)
    return df


def get_accuracy(cfg: ScoreConfig) -> tuple[float, int, int, pd.DataFrame]:
    if not isinstance(cfg.input, pd.DataFrame):
        df = pd.read_json(path_or_buf=cfg.input, lines=True)
    else:
        df = cfg.input
    if cfg.answer_key is not None:
        df["answer"] = df[cfg.answer_key]
    assert df.model.unique().size == 1, "Multiple models in input data"
    assert df.dataset.unique().size == 1, "Multiple datasets in input data"
    model = df.model.unique()[0]
    dataset = df.dataset.unique()[0]

    num_samples = len(df)
    df = get_complete_rows(df, "success")
    new_cols = df.apply(func_correct, debug=cfg.debug, axis=1)

    # calculate accuracy
    df["correct"] = new_cols.apply(lambda x: x[0])
    accuracy = df["correct"].sum() / num_samples
    # calculate unknown outputs from the LLM
    df["unknown"] = new_cols.apply(lambda x: x[1])
    count_unknown = df["unknown"].sum()

    if cfg.verbose:
        print(f"Accuracy: {accuracy}")

    results = {
        "model": model,
        "dataset": dataset,
        "accuracy": accuracy,
        "count_unknown": int(count_unknown),
        "num_samples": int(num_samples),
    }

    # save results to file if specified and append as jsonl
    if cfg.results_file is not None:
        with jsonlines.open(cfg.results_file, "a") as writer:
            writer.write(results)

    return accuracy, count_unknown, num_samples, df


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(ScoreConfig, dest="score_config")
    args = parser.parse_args()
    cfg: ScoreConfig = args.score_config

    get_accuracy(cfg)
