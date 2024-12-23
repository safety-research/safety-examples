import asyncio
import copy
import itertools
from multiprocessing import Manager, Pool
from pathlib import Path

from safetytooling.apis.finetuning.run import Config as ExperimentConfigFinetuning
from safetytooling.apis.finetuning.run import main as finetuning_run_main
from safetytooling.utils import utils
from tqdm import tqdm


def send_ft_job(args):
    (
        train_file,
        val_file,
        model,
        dry_run,
        tag,
        output_dir,
        extra_config,
        n_epochs,
        lock,
    ) = args

    async def run_async(train_file, val_file, model, dry_run, tag, extra_config, n_epochs):
        if isinstance(train_file, str):
            train_file = Path(train_file)
        if isinstance(val_file, str):
            val_file = Path(val_file)
        cfg = ExperimentConfigFinetuning(
            train_file=train_file,
            val_file=val_file,
            model=model,
            n_epochs=n_epochs,
            dry_run=dry_run,
            tags=(tag,),
        )
        ft_job, cost = await finetuning_run_main(cfg, extra_config=extra_config, verbose=False)
        return ft_job, cost

    # add epochs to name
    extra_config = copy.deepcopy(extra_config)
    if "name" in extra_config:
        extra_config["name"] += f" {n_epochs}ep"
    else:
        extra_config["name"] = f"ft:{model} {n_epochs}ep"

    # Execute the asynchronous function in the event loop within the process (I know... it's nasty...)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    ft_job, cost = loop.run_until_complete(run_async(train_file, val_file, model, dry_run, tag, extra_config, n_epochs))
    loop.close()

    # Process the results as before
    fine_tuned_model = ft_job.fine_tuned_model if ft_job is not None else None
    data = {
        "fine_tuned_model": fine_tuned_model,
        "train_file": str(train_file),
        "n_epochs": n_epochs,
        "model": model,
        "cost": cost,
        **extra_config,
    }
    with lock:
        utils.append_jsonl(output_dir / "ft_models_async.jsonl", [data])
    return data


def run_many_ft_jobs(
    jobs: tuple[Path, dict],
    model: str,
    epochs: list[int],
    output_dir: Path,
    dry_run: bool,
    tag: str,
    val_file: Path,
):
    # Wandb doesn't work with asyncronous code, so we need to use a process pool
    manager = Manager()
    lock = manager.Lock()  # Use Manager's Lock for inter-process locking
    pool = Pool(processes=3)  # Control the number of concurrent processes

    tasks = [
        (
            file,
            val_file,
            model,
            dry_run,
            tag,
            output_dir,
            extra_config,
            n_epochs,
            lock,
        )
        for (file, extra_config), n_epochs in itertools.product(jobs, epochs)
    ]

    results = []
    for data in tqdm(pool.imap_unordered(send_ft_job, tasks), total=len(tasks)):
        results.append(data)

    pool.close()
    pool.join()

    utils.save_jsonl(output_dir / "ft_models.jsonl", results)

    total_cost = sum([x["cost"] for x in results])
    print(f"Total cost: ${total_cost:.2f}")

    return results
