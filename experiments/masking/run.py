import os
import shutil

import click
from dotenv import load_dotenv

from experiments.masking.datasets import DATASET_REGISTRY, get_dataset
from experiments.masking.models import MODEL_REGISTRY, get_tokenizer, get_model
from experiments.masking.trial import Trial

masking_experiment = click.Group("masking")


@masking_experiment.command()
@click.option("--trial-name", "-n", required=True, help="Trial name")
@click.option(
    "--dataset",
    type=click.Choice(list(DATASET_REGISTRY.keys())),
    required=True,
    help="Dataset",
)
@click.option(
    "--model",
    type=click.Choice(list(MODEL_REGISTRY.keys())),
    required=True,
    help="Model",
)
@click.option("--p-mask", type=float, required=True, help="Masking probability")
@click.option("--hint/--no-hint", required=True, help="Allow hint to LLM")
@click.option("--token-gen", type=int, default=0, help="Allowed generations")
@click.option(
    "--override", is_flag=True, default=False, help="Override existing results"
)
@click.option("--no-eval", is_flag=True, default=False, help="Only generate samples")
def run(trial_name, dataset, model, p_mask, hint, token_gen, override, no_eval):
    load_dotenv()

    base_out_path = f"results/masking/{trial_name}"

    if os.path.exists(base_out_path):
        if not override:
            click.echo(
                f"Error: {base_out_path} already exists. Use --override to replace."
            )
            return
        shutil.rmtree(base_out_path)

    os.makedirs(base_out_path, exist_ok=True)

    tokenizer = get_tokenizer(model)
    model_instance = get_model(model).to("cuda")
    ds = get_dataset(dataset)
    trial = Trial(ds, base_out_path)

    trial.generate_samples(model_instance, tokenizer, p_mask, hint, token_gen)

    if not no_eval:
        trial.eval()


@masking_experiment.command()
@click.option(
    "--samples",
    type=click.Path(exists=True, resolve_path=True),
    required=True,
    help="Path to jsonl file",
)
@click.option(
    "--dataset",
    type=click.Choice(list(DATASET_REGISTRY.keys())),
    required=True,
    help="Dataset",
)
def eval(samples, dataset):
    ds = get_dataset(dataset)
    trial = Trial(ds, samples)
    trial.eval()


if __name__ == "__main__":
    masking_experiment()
