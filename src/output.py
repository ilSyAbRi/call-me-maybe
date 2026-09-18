"""Write function-calling results to JSON."""

import json
import os
from typing import Any

from src.model import FunctionCallResult


def save_results(
    results: list[FunctionCallResult],
    output_path: str,
) -> None:
    """Save function-call results as a JSON array.

    Args:
        results: Generated function calls.
        output_path: Destination JSON file.

    Raises:
        OSError: If the output file cannot be created.
    """
    directory = os.path.dirname(output_path)

    if directory:
        os.makedirs(directory, exist_ok=True)

    data: list[dict[str, Any]] = []

    for result in results:
        data.append({
            "prompt": result.prompt,
            "name": result.name,
            "parameters": result.parameters,
        })

    try:
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(
                data,
                file,
                indent=2,
                ensure_ascii=False,
            )
    except OSError as error:
        raise OSError(
            f"Could not write output file: {output_path}"
        ) from error