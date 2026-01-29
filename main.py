import ast
import os
import re
import contextlib
import io
import json
import shutil
import warnings

from dotenv import load_dotenv
import litellm
from numpy import mean
from pydantic_core import PydanticSerializationUnexpectedValue
from tqdm import tqdm
import datasets
from bert_score import BERTScorer

from cli import CLIArgs
from predictor import BasePredictor, LLMPredictor


def load_dataset_CFFI(sample_size=100):
    ds_loc = "./data"
    ds = None
    try:
        ds = datasets.load_from_disk(ds_loc)
    except Exception:
        ds = datasets.load_dataset("m-a-p/CodeFeedback-Filtered-Instruction")

        def extract_code(example):
            regex = re.compile(
                rf"(?:```{re.escape(example['lang'])}\n)([\s\S]*?)(?:\n```)"
            )
            m = re.search(regex, example["answer"])
            if m:
                example["code"] = m.group(1)
            else:
                example["code"] = example["answer"]

            return example

        def invalid_code(example):
            with contextlib.redirect_stderr(io.StringIO()):
                try:
                    ast.parse(example["code"])
                    return True
                except SyntaxError:
                    return False

        ds = ds.filter(lambda example: example["lang"] == "python")
        ds = ds.filter(lambda example: example["answer"] != "")
        ds = ds.map(extract_code)
        ds = ds.filter(invalid_code)
        ds = ds.rename_column("query", "desc")
        ds = ds.remove_columns(["answer", "resource", "lang"])
        ds.save_to_disk(ds_loc)

    return ds["train"].shuffle(seed=42).select(range(sample_size))


def main(args: CLIArgs):
    load_dotenv()
    warnings.filterwarnings("ignore", message="Pydantic serializer warnings")

    sample_size = args.sample_size
    batch_size = args.batch_size
    experiment_name = args.experiment_name
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

    ds = load_dataset_CFFI(sample_size)
    predictor = LLMPredictor(model_name=args.model_name)
    scorer = BERTScorer(lang="en")
    results = {}
    f1_scores = []

    os.mkdir(results_dir)

    with tqdm(total=len(ds), desc="Prdictor inference") as pbar:
        i = 0
        for example in iter(ds):
            pred = predictor.predict(example["code"])
            results[i] = {"ref": example["desc"].strip(), "cand": pred.strip()}
            pbar.update(1)
            i += 1

            if i % batch_size == 0:
                with open(
                    f"./results/{experiment_name}/output_batch_{i // batch_size}.json",
                    "w",
                ) as f:
                    json.dump(results, f)

                cands = [r["cand"] for i, r in results.items()]
                refs = [r["ref"] for i, r in results.items()]
                _, _, F1 = scorer.score(cands, refs)
                f1_scores.append(F1.mean())
                results = {}

    if not not results:
        with open(
            f"./results/{experiment_name}/output_batch_{len(ds) // batch_size}.json",
            "w",
        ) as f:
            json.dump(results, f)

            cands = [r["cand"] for i, r in results.items()]
            refs = [r["ref"] for i, r in results.items()]
            _, _, F1 = scorer.score(cands, refs)
            f1_scores.append(F1.mean())
            results = {}

    if isinstance(predictor, LLMPredictor):
        predictor._print_stats()

    mean_f1_score = mean(f1_scores)

    with open(f"./results/{experiment_name}/summary.txt", "w") as f:
        f.write(f"F1: {mean_f1_score}")

    print(f"Finished experiment {experiment_name}")
    print(f"F1:{mean_f1_score}")


if __name__ == "__main__":
    args = CLIArgs()
    main(args)
