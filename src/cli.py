from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.backend import MODEL_MAPPINGS
from src.dataset import DATASET_MAPPING


class CLIArgs(BaseSettings):
    model_config = SettingsConfigDict(cli_parse_args=True)

    batch_size: int = Field(2, alias="batch-size")
    experiment_name: str = Field(alias="name")
    llm: str = Field("gpt-5-mini", examples=list(MODEL_MAPPINGS.keys()))
    dataset: str = Field("CFFI", alias="ds", examples=list(DATASET_MAPPING.keys()))
    sample_size: int = Field(4, alias="sample-size")
    verbose: bool = Field(False, alias="v")
    override: bool = Field(False, alias="o")

    @field_validator("llm", mode="before")
    @classmethod
    def validate_llm_alias(cls, value: str) -> str:
        if value not in MODEL_MAPPINGS:
            raise ValueError(
                f"{value} is not a valid model alias. Valid model alias are {list(MODEL_MAPPINGS.keys())}"
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
