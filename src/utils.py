import asyncio
import os
import pathlib
import random
import re
import sys
from pathlib import Path
from typing import List, Optional

from datasets import Dataset
from jinja2 import Environment, FileSystemLoader
from safetytooling.apis import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from tqdm.auto import tqdm


def add_to_python_path(*relative_paths: str) -> None:
    """
    Add multiple paths to Python path relative to the repository root.
    Repository root is assumed to be one level up from the notebooks directory.
    """
    notebook_dir = pathlib.Path(os.getcwd())
    repo_root = notebook_dir.parent if notebook_dir.name == "notebooks" else notebook_dir

    for path in relative_paths:
        full_path = repo_root / path
        if full_path.exists():
            if str(full_path) not in sys.path:
                sys.path.append(str(full_path))
        else:
            print(f"Warning: Path {full_path} does not exist")


def get_project_root() -> Path:
    """Returns the project root directory."""
    return Path(__file__).resolve().parent.parent


def load_prompt_file(prompt_path: str | Path, **template_vars) -> str:
    """Load system prompt from file relative to project root."""
    project_root = get_project_root()
    env = Environment(loader=FileSystemLoader(str(project_root)), autoescape=False)

    # Convert the full path to be relative to project root
    full_path = Path(prompt_path)
    if full_path.is_absolute():
        relative_path = full_path.relative_to(project_root)
    else:
        relative_path = full_path

    # Normalize path separators to forward slashes for Jinja2
    path_str = str(relative_path).replace("\\", "/")
    try:
        template = env.get_template(path_str)
        return template.render(**template_vars)
    except Exception as e:
        print(f"Error loading template from {path_str}: {str(e)}")
        print(f"Project root: {project_root}")
        print(f"Full path: {full_path}")
        print(f"Relative path: {relative_path}")
        print(f"Normalized path: {path_str}")
        raise
