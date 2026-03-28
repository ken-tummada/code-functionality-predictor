import click

from experiments.masking import run


@click.group()
def cli():
    pass


cli.add_command(run.masking_experiment)

if __name__ == "__main__":
    cli()
