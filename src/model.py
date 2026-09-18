"""Pydantic models used by the project."""

from typing import Literal

from pydantic import BaseModel


ParameterType = Literal[
    "number",
    "string",
    "boolean",
    "integer",
]


class Prompt(BaseModel):
    """Represent one user prompt."""

    prompt: str


class Parameter(BaseModel):
    """Represent one function parameter."""

    type: ParameterType


class ReturnType(BaseModel):
    """Represent a function return type."""

    type: ParameterType


class FunctionDefinition(BaseModel):
    """Represent one available function."""

    name: str
    description: str | None
    parameters: dict[str, Parameter]
    returns: ReturnType

    def __str__(self) -> str:
        """Return a readable function description."""
        params = ", ".join(
            [
                f"{name}: {parameter.type}"
                for name, parameter in self.parameters.items()
            ]
        )

        return (
            f"{self.name}({params}) -> "
            f"{self.returns.type}: {self.description}"
        )


class FunctionCallResult(BaseModel):
    """Represent one final function call."""

    prompt: str
    name: str
    parameters: dict