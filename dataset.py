import datasets
import re
import contextlib
import io
import ast


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


def get_dataset(name: str, sample_size: int = 100):
    dataset_mapping = {
        "CFFI": {
            "fullname": "m-a-p/CodeFeedback-Filtered-Instruction",
            "preprocess": preprocess_CFFI_dataset,
        },
    }

    ds_loc = "./data"
    ds = None
    try:
        ds = datasets.load_from_disk(ds_loc)
    except Exception:
        fullname, preprocess = (
            dataset_mapping[name]["fullname"],
            dataset_mapping[name]["preprocess"],
        )
        ds = datasets.load_dataset(fullname)
        ds = preprocess(ds)
        ds.save_to_disk(ds_loc)

    return ds["train"].shuffle(seed=42).select(range(sample_size))


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
