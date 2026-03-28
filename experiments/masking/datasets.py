import datasets


def _preprocess_mbpp(ds):
    ds = ds.rename_columns(
        {
            "task_id": "id",
            "prompt": "hint",
            "test_list": "tests",
        }
    )

    ds = ds["test"]
    ds = ds.map(
        lambda ex: {
            **ex,
            "code": f"```python\n{ex['code']}\n```",
            "tests": "\n".join(ex["tests"]),
        }
    )

    return ds


DATASET_REGISTRY = {
    "mbpp": {
        "path": "mbpp",
        "name": "sanitized",
        "preprocess": _preprocess_mbpp,
    },
}


def get_dataset(alias: str):
    config = DATASET_REGISTRY[alias]
    cache_path = f"data/masking/{alias}"
    try:
        raise Exception
        ds = datasets.load_from_disk(cache_path)

    except Exception:
        ds = datasets.load_dataset(config["path"], config["name"])
        ds = config["preprocess"](ds)
        ds.save_to_disk(cache_path)

    return ds
