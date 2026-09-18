"""Build complete function calls from user prompts."""

from typing import Any

from src.arguments import ArgumentExtractor
from src.decoder import Constrain_decoder
from src.model import FunctionCallResult, Prompt


class FunctionCallGenerator:
    """Generate complete function calls."""

    def __init__(self, decoder: Constrain_decoder) -> None:
        """Create a function-call generator.

        Args:
            decoder: Existing constrained decoder.
        """
        self.decoder = decoder
        self.arguments = ArgumentExtractor(decoder)

    def generate(self, prompt: Prompt) -> FunctionCallResult:
        """Generate one complete function call.

        Args:
            prompt: User request.

        Returns:
            A validated function call result.
        """
        function = self.decoder.get_func_name(prompt)

        parameters = self.arguments.extract(
            prompt,
            function,
        )

        self._validate_parameters(
            parameters,
            function,
        )

        return FunctionCallResult(
            prompt=prompt.prompt,
            name=function.name,
            parameters=parameters,
        )

    def _validate_parameters(
        self,
        parameters: dict[str, Any],
        function: Any,
    ) -> None:
        """Check that generated arguments match the function schema.

        Args:
            parameters: Generated arguments.
            function: Function definition.

        Raises:
            ValueError: If the arguments do not match the schema.
        """
        expected = function.parameters

        if set(parameters) != set(expected):
            raise ValueError(
                f"Wrong parameters for {function.name}"
            )

        for name, parameter in expected.items():
            value = parameters[name]

            if parameter.type == "number":
                self._check_number(name, value)

            elif parameter.type == "integer":
                self._check_integer(name, value)

            elif parameter.type == "string":
                self._check_string(name, value)

            elif parameter.type == "boolean":
                self._check_boolean(name, value)

    def _check_number(self, name: str, value: Any) -> None:
        """Check a number argument."""
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"Parameter '{name}' must be a number"
            )

    def _check_integer(self, name: str, value: Any) -> None:
        """Check an integer argument."""
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"Parameter '{name}' must be an integer"
            )

    def _check_string(self, name: str, value: Any) -> None:
        """Check a string argument."""
        if not isinstance(value, str):
            raise ValueError(
                f"Parameter '{name}' must be a string"
            )

    def _check_boolean(self, name: str, value: Any) -> None:
        """Check a boolean argument."""
        if not isinstance(value, bool):
            raise ValueError(
                f"Parameter '{name}' must be a boolean"
            )