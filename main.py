import re
import datasets
from bert_score import BERTScorer

from predictor import BasePredictor
    
def load_dataset_CFFI():
    ds_loc = "./data"
    ds = None
    try:
        ds = datasets.load_from_disk(ds_loc)
    except Exception:
        ds = datasets.load_dataset("m-a-p/CodeFeedback-Filtered-Instruction")

        def extract_code(example):
            regex = re.compile(rf"(?:```{re.escape(example['lang'])}\n)([\s\S]*?)(?:\n```)")
            m = re.search(regex, example["answer"])
            if m:
                example["code"] = m.group(1)
            else:
                example["code"] = ""

            return example

        ds = ds.map(extract_code)
        ds = ds.rename_column("query", "desc")
        ds.save_to_disk(ds_loc)

    return ds["train"]

def main():
    ds = load_dataset_CFFI()
    predictor = BasePredictor()
    scorer = BERTScorer(lang="en")
    cands = []
    refs = []

    for example in iter(ds):
        pred = predictor.predict(example["code"])
        refs.append(example["desc"].strip()) 
        cands.append(pred.strip())

    _, _, F1 = scorer.score(cands, refs)
    print(F1.mean())

if __name__ == "__main__":
    main()
