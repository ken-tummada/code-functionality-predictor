from typing import Any
import litellm
import pydantic
from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)
from tqdm import tqdm
from errors import (
    CostLimitExceededError,
    ContextWindowExceededError,
)

_MAX_RETRIES = 10
MODEL_MAPPINGS = {
    "sonnet-4.5": "bedrock/arn:aws:bedrock:us-east-1:288380904485:inference-profile/global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "gpt-5-mini": "gpt-5-mini",
    "deepseek-r1": "bedrock/arn:aws:bedrock:us-east-1:463470961764:inference-profile/us.deepseek.r1-v1:0",
    "llama-3-8b": "bedrock/arn:aws:bedrock:us-east-1:288380904485:inference-profile/us.meta.llama3-1-8b-instruct-v1:0",
}


class LLMBackend:
    class Stats(pydantic.BaseModel):
        total_cost: float = 0
        tokens_sent: int = 0
        tokens_received: int = 0
        api_calls: int = 0

    def __init__(self, model_name: str, *args, **kwargs) -> None:
        self.model_name = MODEL_MAPPINGS[model_name]
        self.model_max_input_tokens = 1_000
        self.host_url = kwargs.get("host_url", "")
        self.lm_provider = "openai"
        self.stats = self.Stats()

        self.total_cost_limit = kwargs.get("total_cost_limit", 40.0)

    def _setup_client(self) -> None:
        self.model_max_input_tokens = litellm.model_cost.get(self.model_name, {}).get(
            "max_input_tokens"
        )
        self.model_max_output_tokens = litellm.model_cost.get(self.model_name, {}).get(
            "max_output_tokens"
        )
        self.lm_provider = litellm.model_cost.get(self.model_name, {}).get(
            "litellm_provider"
        )
        if self.lm_provider is None and self.host_url is not None:
            print(
                f"Using a custom API base: {self.host_url}. Cost management and context length error checking will not work."
            )

    def _print_stats(self) -> None:
        tqdm.write(
            f"total_tokens_sent={self.stats.tokens_sent:,}, "
            f"total_tokens_received={self.stats.tokens_received:,}, "
            f"total_cost={self.stats.total_cost:.2f}, "
            f"total_api_calls={self.stats.api_calls:,}",
        )

    def _update_stats(
        self, input_tokens: int, output_tokens: int, cost: float = 0.0
    ) -> None:
        self.stats.total_cost += cost
        self.stats.tokens_sent += input_tokens
        self.stats.tokens_received += output_tokens
        self.stats.api_calls += 1

        if self.stats.api_calls % 50 == 0:
            self._print_stats()

        # Check whether total cost or instance cost limits have been exceeded
        if 0 < self.total_cost_limit <= self.stats.total_cost:
            print(
                f"Cost {self.stats.total_cost:.2f} exceeds limit {self.total_cost_limit:.2f}"
            )
            msg = "Total cost limit exceeded"
            raise CostLimitExceededError(msg)

    @retry(
        wait=wait_random_exponential(min=60, max=180),
        reraise=True,
        stop=stop_after_attempt(_MAX_RETRIES),
        retry=retry_if_not_exception_type(
            (
                CostLimitExceededError,
                RuntimeError,
                litellm.exceptions.UnsupportedParamsError,
                litellm.exceptions.NotFoundError,
                litellm.exceptions.PermissionDeniedError,
                litellm.exceptions.ContextWindowExceededError,
                litellm.exceptions.APIError,
            )
        ),
    )
    def query(self, messages: list[Any]) -> str:
        input_tokens: int = litellm.token_counter(
            messages=messages, model=self.model_name
        )

        if self.model_max_input_tokens is None:
            print(f"No max input tokens found for model {self.model_name!r}")
        elif input_tokens > self.model_max_input_tokens:
            msg = f"Input tokens {input_tokens} exceed max tokens {self.model_max_input_tokens}"
            raise ContextWindowExceededError(msg)
        extra_args = {}
        if self.host_url:
            extra_args["api_base"] = self.host_url

        # FIXME: idk what this does
        #
        # completion_kwargs = self.completion_kwargs
        # if self.lm_provider == "anthropic":
        # completion_kwargs["max_tokens"] = self.model_max_output_tokens

        try:
            response: litellm.types.utils.ModelResponse = litellm.completion(
                model=self.model_name,
                messages=messages,
                # FIXME: remove?
                # temperature=self.args.temperature,
                # top_p=self.args.top_p,
                # api_version=self.args.api_version,
                # **completion_kwargs,
                **extra_args,
            )
        except Exception as e:
            print("Error during LLM query:\n")
            print(e)
            raise
        choices: litellm.types.utils.Choices = response.choices  # type: ignore
        output: str = choices[0].message.content or ""

        # update stats
        cost = litellm.cost_calculator.completion_cost(response)
        output_tokens = litellm.utils.token_counter(text=output, model=self.model_name)
        self._update_stats(
            input_tokens=input_tokens, output_tokens=output_tokens, cost=cost
        )

        return output
