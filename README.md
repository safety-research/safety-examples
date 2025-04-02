# safety-examples

This repository uses `safety-tooling` as a submodule and showcases how to use the LLM api, experiment utils, prompt utils and environment setup. The repo is set up to have core code in `examples` and lightweight scripts that call that code in `experiments/<exp_name>`. We recommend forking this repository to get started with your project since it has pre-commit hooks and the submodule setup already. Even if you think you won’t need any of this tooling, we suggest forking anyway since it will be easy for you to add your own cool tooling to `safety-tooling` in the future if you would like.

## Set-up

1. First pull the submodules:

```bash
git submodule update --init --recursive
```

2. Follow the instructions in safety-tooling/README.md to set up the environment. Ensure to make a `.env` file with your API keys. This `.env` file can be in the root of the repo or in the safety-tooling submodule (safetytooling.utils.setup_environment() will check both places).

3. Install the requirements:

```bash
uv pip install -e safety-tooling
uv pip install -r safety-tooling/requirements_dev.txt
uv pip install -r requirements.txt # once you have added any other dependencies
```


## Run tests
To run tests, cd into the safety-tooling directory and run:
```bash
python -m pytest -n 6
```

## Add the submodule to your pythonpath
The submodule must be in your pythonpath for you to be able to use it. There are a few ways to do this:

1. Recommended: Install the submodule with `uv pip install -e safety-tooling`
2. Add the submodule to your pythonpath in the `<main_module>/__init__.py` of your main module (e.g. see `examples/__init__.py`). You then have to call your code like `python -m <main_module>.<other_module>`.
3. Add the submodule to your pythonpath manually. E.g. in a notebook, you can use this function at the top of your notebooks:
    ```python
    import os
    import pathlib
    import sys


    def put_submodule_in_python_path(submodule_name: str):
        repo_root = pathlib.Path(os.getcwd())
        submodule_path = repo_root / submodule_name
        if submodule_path.exists():
            sys.path.append(str(submodule_path))

    put_submodule_in_python_path("safety-tooling")
    ```

## Examples

The repo has 4 examples:

- Getting responses from an LLM to harmful direct requests and classifying them with another LLM. This involves chaining two scripts together.
    - `experiments/examples/241223_running_examples/run_get_responses.sh`
    - Stage 1: Gets LLM responses with `examples.inference.get_responses`
        - This adds a new column to the input jsonl (`experiments/examples/241223_running_examples/direct_request.jsonl`) and saves it as a new output file. This workflow is nice because you get to keep all metadata along with the model output.
    - Stage 2: Gets the classifier output using the HarmBench grader with `examples.inference.run_classifier`
        - This also adds another column for the classifier decision and outputs a new jsonl.
        - Once finished you can now analyse the results in a notebook quickly with `pandas`.
- Best of N jailbreaking
    - `experiments/examples/241223_running_examples/run_bon_jailbreaking.sh`
    - Repeated sampling of new augmented harmful queries until one jailbreaks the model.
- PAIR attack
    - `experiments/examples/241223_running_examples/run_pair.sh`
    - Use an attacker LLM to adapt and rewrite harmful prompts until it jailbreaks the model.
- Multi choice benchmarks
    - `experiments/examples/241223_running_examples/run_capability_evals.sh`
    - Run benchmarks like MMLU and find the accuracy.
