import re
import contextlib
import io
import ast
from pathlib import Path

import datasets


DATASET_MAPPING = {
    "CFFI": "m-a-p/CodeFeedback-Filtered-Instruction",
}


def preprocess_CFFI_dataset(ds):
    def extract_code(example):
        regex = re.compile(rf"(?:```{re.escape(example['lang'])}\n)([\s\S]*?)(?:\n```)")
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
    return ds


def get_dataset(name: str, sample_size: int = 100, batch_size: int = -1):
    DATASET_PREPROCESS_MAPPING = {"CFFI": preprocess_CFFI_dataset}

    ds_loc = f"./data/{name}"
    ds = None
    try:
        ds = datasets.load_from_disk(ds_loc)
        print(f"Loaded dataset {name} from {Path(ds_loc).absolute().as_posix()}")
    except Exception:
        fullname, preprocess = (
            DATASET_MAPPING[name],
            DATASET_PREPROCESS_MAPPING[name],
        )
        print(f"Downloading dataset {fullname}")
        ds = datasets.load_dataset(fullname)
        ds = preprocess(ds)
        ds.save_to_disk(ds_loc)
        print(f"Dataset saved to {Path(ds_loc).absolute().as_posix()}")

    if batch_size == -1:
        batch_size = sample_size

    return ds["train"].shuffle(seed=42).select(range(sample_size)).batch(batch_size)
