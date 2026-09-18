"""Load project data from JSON files."""

import json

from src.model import FunctionDefinition, Prompt


def load_json(path: str):
    """Load JSON data from a file.

    Args:
        path: JSON file path.

    Returns:
        Parsed JSON data.

    Raises:
        ValueError: If JSON is invalid.
        FileNotFoundError: If the file does not exist.
    """
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON file: {path}"
        ) from error

    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"File not found: {path}"
        ) from error


def load_prompts(path: str) -> list[Prompt]:
    """Load user prompts from a JSON file.

    Args:
        path: Input JSON path.

    Returns:
        List of validated prompts.
    """
    data = load_json(path)

    if not isinstance(data, list):
        raise ValueError(
            f"Expected a JSON array in: {path}"
        )

    prompts = []

    for prompt_data in data:
        if not isinstance(prompt_data, dict):
            raise ValueError(
                "Each prompt must be a JSON object."
            )

        prompt = Prompt(
            prompt=prompt_data.get("prompt")
        )

        prompts.append(prompt)

    return prompts


def load_functions(
    path: str,
) -> dict[str, FunctionDefinition]:
    """Load function definitions from JSON.

    Args:
        path: Function-definition JSON path.

    Returns:
        Dictionary indexed by function name.
    """
    data = load_json(path)

    if not isinstance(data, list):
        raise ValueError(
            f"Expected a JSON array in: {path}"
        )

    functions: dict[str, FunctionDefinition] = {}

    for function_data in data:
        if not isinstance(function_data, dict):
            raise ValueError(
                "Each function must be a JSON object."
            )

        function = FunctionDefinition(
            name=function_data.get("name"),
            description=function_data.get("description"),
            parameters=function_data.get("parameters"),
            returns=function_data.get("returns"),
        )

        functions[function.name] = function

    return functions