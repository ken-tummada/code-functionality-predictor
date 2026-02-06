import ast
import os
import re
import contextlib
import io
import sys
import shutil
import json
import warnings

from dotenv import load_dotenv
from numpy import mean
from tqdm import tqdm
import datasets
from bert_score import BERTScorer

from src.backend import LLMBackend
from src.cli import CLIArgs
from src.dataset import get_dataset
from src.predictor.description import DescriptionLLMPredictor
from src.eval.desc_scorring import DescriptionScorrer


def main(args: CLIArgs):
    load_dotenv()
    warnings.filterwarnings("ignore", message="Pydantic serializer warnings")

    sample_size = args.sample_size
    batch_size = args.batch_size
    experiment_name = args.experiment_name
    dataset_name = args.dataset
    save_preds = args.save_outputs
    results_dir = f"./results/{experiment_name}"

    if batch_size == -1:
        batch_size = sample_size

    if sample_size < batch_size:
        print("Error: sample_size can't be less than batch_size")
        exit(1)

    if not os.path.exists("./results"):
        os.mkdir("./results")

    if os.path.exists(results_dir):
        if not args.override:
            print(f"Error: experiment with name {experiment_name} already exists.")
            exit(1)
        else:
            print("Info: removing old results.")
            shutil.rmtree(results_dir)

    ds = get_dataset(dataset_name, sample_size, batch_size)
    model = DescriptionLLMPredictor(model=args.llm)
    judge = LLMBackend("gpt-5-mini")
    evaluator = DescriptionScorrer(
        model,
        ds,
        judge,
        save_preds=save_preds,
        experiment_name=experiment_name,
    )

    evaluator.evaluate()
    os.mkdir(results_dir)

    print("LLM usage and costs")
    if isinstance(model, DescriptionLLMPredictor) and model.llm_model is not None:
        print("LLM predictor:")
        model.llm_model._print_stats()
    print("LLM judge:")
    judge._print_stats()

    with open(f"./results/{experiment_name}/summary.json", "w") as f:
        json.dump(evaluator.compute_metrics(), f)

    with open(f"./results/{experiment_name}/raw_metrics.json", "w") as f:
        json.dump(evaluator.results, f)

    print(f"Experiment {experiment_name} summary:\n{evaluator.compute_metrics()}")


if __name__ == "__main__":
    args = CLIArgs()  # type: ignore
    main(args)
