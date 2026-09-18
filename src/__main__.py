"""Main entry point for the Call-Me-Maybe project."""

import argparse
import sys

from src.decoder import Constrain_decoder
from src.generator import FunctionCallGenerator
from src.output import save_results
from src.parser import load_prompts


DEFAULT_FUNCTIONS = (
    "data/input/functions_definition.json"
)

DEFAULT_INPUT = (
    "data/input/function_calling_tests.json"
)

DEFAULT_OUTPUT = (
    "data/output/function_calling_results.json"
)


def parse_arguments() -> argparse.Namespace:
    """Read command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Convert natural language into function calls."
    )

    parser.add_argument(
        "--functions_definition",
        default=DEFAULT_FUNCTIONS,
        help="Path to the function definitions JSON file.",
    )

    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Path to the input prompts JSON file.",
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Path to the output JSON file.",
    )

    return parser.parse_args()


def main() -> int:
    """Run the complete function-calling pipeline.

    Returns:
        0 when successful, 1 when an error occurs.
    """
    args = parse_arguments()

    try:
        decoder = Constrain_decoder(args.functions_definition)

        prompts = load_prompts(args.input)

        generator = FunctionCallGenerator(decoder)
        results = []

        for prompt in prompts:
            result = generator.generate(prompt)
            results.append(result)

        save_results(
            results,
            args.output,
        )

        print(f"Created: {args.output}")
        return 0

    except (ValueError, FileNotFoundError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    except Exception as error:
        print(
            f"Unexpected error: {error}",
            file=sys.stderr,
        )
        return 1


def _load_functions(path: str) -> dict:
    """Load function definitions from a custom path.

    Args:
        path: Function-definition JSON path.

    Returns:
        Dictionary of function definitions.
    """
    from src.parser import load_functions

    return load_functions(path)


if __name__ == "__main__":
    raise SystemExit(main())