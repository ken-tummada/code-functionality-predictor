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

from src.cli import CLIArgs
from src.dataset import get_dataset
from src.predictor import BasePredictor, LLMPredictor


def main(args: CLIArgs):
    load_dotenv()
    warnings.filterwarnings("ignore", message="Pydantic serializer warnings")

    sample_size = args.sample_size
    batch_size = args.batch_size
    experiment_name = args.experiment_name
    dataset_name = args.dataset
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
    predictor = LLMPredictor(model_name=args.llm)
    scorer = BERTScorer(lang="en")
    f1_scores = []

    os.mkdir(results_dir)

    batch_number = 0
    for examples in tqdm(ds, desc="Generating predictions"):
        preds = predictor(examples["code"])
        results = []
        for i in range(len(preds)):
            results.append(
                {
                    "ref": examples["desc"][i].strip(),
                    "cand": preds[i].strip(),
                }
            )

        with open(
            f"./results/{experiment_name}/output_batch_{batch_number}.json",
            "w",
        ) as f:
            json.dump(results, f)

            _, _, F1 = scorer.score(preds, examples["desc"])
            f1_scores.append(F1.mean())
            results = []
            batch_number += 1

    if isinstance(predictor, LLMPredictor):
        predictor.llm_model._print_stats()

    mean_f1_score = mean(f1_scores)

    with open(f"./results/{experiment_name}/summary.txt", "w") as f:
        f.write(f"F1: {mean_f1_score}")

    print(f"Finished experiment {experiment_name}")
    print(f"F1:{mean_f1_score}")


if __name__ == "__main__":
    args = CLIArgs()
    main(args)
