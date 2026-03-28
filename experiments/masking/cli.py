import click

from experiments.masking.models import MODEL_REGISTRY
from experiments.masking.datasets import DATASET_REGISTRY


@click.command()
@click.option("--trial-name", "-n", required=True, help="Trial name")
@click.option(
    "--dataset",
    "-d",
    default="mbpp",
    type=click.Choice(list(DATASET_REGISTRY.keys())),
    help="Dataset",
)
@click.option(
    "--model",
    "-m",
    default=list(MODEL_REGISTRY.keys())[0],
    type=click.Choice(list(MODEL_REGISTRY.keys())),
    help="Model",
)
@click.option("--p-mask", type=float, default=0.2, help="Masking probability")
@click.option("--allow-hint/--no-allow-hint", default=False, help="Allow hint")
@click.option("--allowed-gen", type=int, default=0, help="Allowed generations")
@click.option("--verbose", "-v", is_flag=True, default=False, help="Verbose mode")
@click.option("--override", "-o", is_flag=True, default=False, help="Override existing")
def cli(trial_name, dataset, model, p_mask, allow_hint, allowed_gen, verbose, override):
    """CLI for masking experiments."""
    pass
