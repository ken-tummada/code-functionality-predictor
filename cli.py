from typing import Literal
from pydantic import BaseModel, Field
from predictor import LLMPredictor
from pydantic_settings import BaseSettings, SettingsConfigDict


def llm_model_name_validator(value: str) -> str:
    if value not in LLMPredictor.model_name_mapping:
        raise ValueError(
            f"{value} is not a valid model alias. Valid model alias are {list(LLMPredictor.model_name_mapping.keys())}"
        )

    return value


class CLIArgs(BaseSettings):
    model_config = SettingsConfigDict(cli_parse_args=True)

    batch_size: int = Field(-1, alias="batch-size")
    experiment_name: str = Field(alias="name")
    model_name: str = Field("gpt", alias="model")
    sample_size: int = Field(3, alias="sample-size")
    verbose: bool = Field(False, alias="v")
    override: bool = Field(False, alias="o")
