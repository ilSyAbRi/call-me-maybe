from pydantic import BaseModel, Field

from typing import Literal

ParameterType = Literal["number", "string", "boolean", "integer"]


class Prompt(BaseModel):
    prompt: str

class Parameter(BaseModel):
    type: ParameterType

class ReturnType(BaseModel):
    type: ParameterType

class FunctionDefinition(BaseModel):
    name: str
    description: str | None
    parameters: dict[str, Parameter]
    returns: ReturnType


    def __str__(self):
        params = ', '.join([f"{name}: {type.type}" for name, type in self.parameters.items()])
        return f"{self.name}({params}) -> {self.returns.type}: {self.description}"


class FunctionCallResult(BaseModel):
    prompt: str
    name: str
    parameters: dict


"func_name(param: type, ...) -> return_type: description"
"func_name(param: type, ...) -> return_type: description"
"func_name(param: type, ...) -> return_type: description"
"func_name(param: type, ...) -> return_type: description"