from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict, CliImplicitFlag

from src.backend import MODEL_MAPPINGS
from src.dataset import DATASET_MAPPING


class CLIArgs(BaseSettings, cli_parse_args=True):
    experiment_name: str
    sample_size: int = 4
    batch_size: int = 2
    dataset: str = Field("CFFI", alias="ds", examples=list(DATASET_MAPPING.keys()))
    llm: str = Field("gpt-5-mini", examples=list(MODEL_MAPPINGS.keys()))

    verbose: bool = False
    override: bool = False
    save_outputs: bool = False

    model_config = SettingsConfigDict(
        cli_shortcuts={
            "verbose": "v",
            "override": "o",
            "save-outputs": "s",
            "experiment-name": ["name", "n"],
        },
        cli_implicit_flags=True,
        cli_kebab_case=True,
    )

    @field_validator("llm", mode="before")
    @classmethod
    def validate_llm_alias(cls, value: str) -> str:
        if value not in MODEL_MAPPINGS and value != "none":
            raise ValueError(
                f"{value} is not a valid model alias. Valid model alias are {list(MODEL_MAPPINGS.keys()) + ['none']}"
            )
        return value

    @field_validator("dataset", mode="before")
    @classmethod
    def validate_dataset_alias(cls, value: str) -> str:
        if value not in DATASET_MAPPING:
            raise ValueError(
                f"{value} is not a valid model alias. Valid model alias are {list(DATASET_MAPPING.keys())}"
            )
        return value
