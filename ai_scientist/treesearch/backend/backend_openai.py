import json
import logging
import os
import time

from .utils import FunctionSpec, OutputType, opt_messages_to_list, backoff_create
from funcy import notnone, once, select_values
import openai
from rich import print

logger = logging.getLogger("ai-scientist")


OPENAI_TIMEOUT_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)

def get_ai_client(model: str, max_retries=2) -> openai.OpenAI:
    if model.startswith("ollama/"):
        client = openai.OpenAI(
            base_url="http://localhost:11434/v1", 
            max_retries=max_retries
        )
    else:
        client = openai.OpenAI(max_retries=max_retries)
    return client


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    client = get_ai_client(model_kwargs.get("model"), max_retries=0)
    filtered_kwargs: dict = select_values(notnone, model_kwargs)  # type: ignore

    messages = opt_messages_to_list(system_message, user_message)

    # Auto-detect whether to use OpenAI native function calling, or fall back
    # to JSON-mode for OpenAI-compatible proxies that don't support tools.
    # Trigger fallback when:
    #   - OPENAI_USE_JSON_FALLBACK=1 is set explicitly, or
    #   - a custom OPENAI_BASE_URL is in use (third-party proxy).
    use_json_fallback = (
        os.environ.get("OPENAI_USE_JSON_FALLBACK", "").strip() in ("1", "true", "True")
        or bool(os.environ.get("OPENAI_BASE_URL"))
    )
    print(f"[backend_openai] use_json_fallback={use_json_fallback} "
          f"OPENAI_BASE_URL={os.environ.get('OPENAI_BASE_URL', '<unset>')}")

    if func_spec is not None and not use_json_fallback:
        filtered_kwargs["tools"] = [func_spec.as_openai_tool_dict]
        # force the model to use the function
        filtered_kwargs["tool_choice"] = func_spec.openai_tool_choice_dict
    elif func_spec is not None:
        # JSON-mode fallback: request a JSON object matching the function
        # schema, prepend an instruction to the system message describing it.
        schema_str = json.dumps(func_spec.json_schema, ensure_ascii=False)
        instruction = (
            f"You MUST respond with a single valid JSON object only "
            f"(no prose, no markdown fences). The JSON object MUST match this "
            f"JSON schema for the function `{func_spec.name}` "
            f"(`{func_spec.description}`):\n{schema_str}"
        )
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = messages[0]["content"].rstrip() + "\n\n" + instruction
        else:
            messages.insert(0, {"role": "system", "content": instruction})
        filtered_kwargs["response_format"] = {"type": "json_object"}

    if filtered_kwargs.get("model", "").startswith("ollama/"):
       filtered_kwargs["model"] = filtered_kwargs["model"].replace("ollama/", "")

    t0 = time.time()
    completion = backoff_create(
        client.chat.completions.create,
        OPENAI_TIMEOUT_EXCEPTIONS,
        messages=messages,
        **filtered_kwargs,
    )
    req_time = time.time() - t0

    choice = completion.choices[0]

    if func_spec is None:
        output = choice.message.content
    elif use_json_fallback:
        # JSON-mode fallback path: parse the message content as JSON object.
        raw = choice.message.content or ""
        try:
            # tolerate ```json ... ``` fences just in case
            if raw.lstrip().startswith("```"):
                import re
                m = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
                if m:
                    raw = m.group(1)
            output = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.error(
                f"JSON-mode fallback: failed to parse model output as JSON: {raw!r}"
            )
            raise e
    else:
        assert (
            choice.message.tool_calls
        ), f"function_call is empty, it is not a function call: {choice.message}"
        assert (
            choice.message.tool_calls[0].function.name == func_spec.name
        ), "Function name mismatch"
        try:
            print(f"[cyan]Raw func call response: {choice}[/cyan]")
            output = json.loads(choice.message.tool_calls[0].function.arguments)
        except json.JSONDecodeError as e:
            logger.error(
                f"Error decoding the function arguments: {choice.message.tool_calls[0].function.arguments}"
            )
            raise e

    in_tokens = completion.usage.prompt_tokens
    out_tokens = completion.usage.completion_tokens

    info = {
        "system_fingerprint": completion.system_fingerprint,
        "model": completion.model,
        "created": completion.created,
    }

    return output, req_time, in_tokens, out_tokens, info
